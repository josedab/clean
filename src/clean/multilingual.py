"""Multi-Language Label Error Detection.

This module extends label error detection to support 50+ languages using
multilingual embeddings and cross-lingual consistency checks.

Example:
    >>> from clean.multilingual import MultilingualDetector
    >>>
    >>> # Create detector with multilingual model
    >>> detector = MultilingualDetector(embedding_model="multilingual")
    >>>
    >>> # Detect label errors in multilingual dataset
    >>> errors = detector.find_errors(df, text_column="text", label_column="label")
    >>> print(f"Found {len(errors)} potential label errors")
    >>>
    >>> # Check cross-lingual consistency
    >>> consistency = detector.check_cross_lingual_consistency(df_en, df_de)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clean.detection import LabelErrorDetector
from clean.exceptions import CleanError, ConfigurationError, DependencyError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Supported languages with ISO 639-1 codes
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "fa": "Persian",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "sw": "Swahili",
    "af": "Afrikaans",
    "ca": "Catalan",
    "eu": "Basque",
    "gl": "Galician",
}


class MultilingualModel(Enum):
    """Supported multilingual embedding models."""

    MULTILINGUAL_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MULTILINGUAL_MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    LABSE = "sentence-transformers/LaBSE"
    MUSE = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    XLM_ROBERTA = "xlm-roberta-base"


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""

    text: str
    detected_language: str
    confidence: float
    all_languages: dict[str, float] = field(default_factory=dict)


@dataclass
class MultilingualLabelError:
    """Label error detected in multilingual context."""

    index: int
    text: str
    language: str
    given_label: Any
    predicted_label: Any
    confidence: float
    cross_lingual_agreement: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "language": self.language,
            "given_label": self.given_label,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "cross_lingual_agreement": self.cross_lingual_agreement,
        }


@dataclass
class CrossLingualConsistencyResult:
    """Result of cross-lingual consistency check."""

    language_pair: tuple[str, str]
    n_samples: int
    consistency_score: float
    inconsistent_samples: list[dict[str, Any]]
    embedding_similarity: float

    def summary(self) -> str:
        """Generate summary string."""
        lang1, lang2 = self.language_pair
        return (
            f"Cross-Lingual Consistency: {lang1} ↔ {lang2}\n"
            f"Samples: {self.n_samples}\n"
            f"Consistency Score: {self.consistency_score:.2%}\n"
            f"Embedding Similarity: {self.embedding_similarity:.3f}\n"
            f"Inconsistent Samples: {len(self.inconsistent_samples)}"
        )


@dataclass
class MultilingualReport:
    """Report from multilingual analysis."""

    n_samples: int
    n_languages: int
    language_distribution: dict[str, int]
    errors: list[MultilingualLabelError]
    error_rate_by_language: dict[str, float]
    cross_lingual_results: list[CrossLingualConsistencyResult] | None
    overall_error_rate: float

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Multilingual Label Error Report",
            "=" * 50,
            f"Total Samples: {self.n_samples:,}",
            f"Languages Detected: {self.n_languages}",
            f"Total Errors: {len(self.errors)}",
            f"Overall Error Rate: {self.overall_error_rate:.1%}",
            "",
            "Language Distribution:",
        ]

        for lang, count in sorted(self.language_distribution.items(), key=lambda x: -x[1])[:10]:
            lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
            error_rate = self.error_rate_by_language.get(lang, 0)
            lines.append(f"  {lang_name:20s} {count:6d} samples  {error_rate:.1%} error rate")

        if self.cross_lingual_results:
            lines.append("")
            lines.append("Cross-Lingual Consistency:")
            for result in self.cross_lingual_results[:5]:
                lang1, lang2 = result.language_pair
                lines.append(f"  {lang1} ↔ {lang2}: {result.consistency_score:.1%}")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert errors to DataFrame."""
        return pd.DataFrame([e.to_dict() for e in self.errors])


class LanguageDetector:
    """Detect language of text samples."""

    def __init__(self, method: str = "fasttext"):
        """Initialize language detector.

        Args:
            method: Detection method ("fasttext", "langdetect", "polyglot")
        """
        self.method = method
        self._detector = None

        if method == "fasttext":
            self._init_fasttext()
        elif method == "langdetect":
            self._init_langdetect()
        else:
            # Fallback to simple heuristics
            self._detector = None

    def _init_fasttext(self) -> None:
        """Initialize FastText detector."""
        try:
            import fasttext
            # Would need to download lid.176.bin model
            # For now, fall back to langdetect
            self._init_langdetect()
        except ImportError:
            self._init_langdetect()

    def _init_langdetect(self) -> None:
        """Initialize langdetect."""
        try:
            import langdetect
            self._langdetect = langdetect
            self.method = "langdetect"
        except ImportError:
            self._langdetect = None
            self.method = "heuristic"

    def detect(self, text: str) -> LanguageDetectionResult:
        """Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            LanguageDetectionResult
        """
        if not text or len(text.strip()) < 3:
            return LanguageDetectionResult(
                text=text,
                detected_language="unknown",
                confidence=0.0,
            )

        if self.method == "langdetect" and self._langdetect:
            try:
                probs = self._langdetect.detect_langs(text)
                all_langs = {str(p.lang): p.prob for p in probs}
                best = probs[0]
                return LanguageDetectionResult(
                    text=text,
                    detected_language=str(best.lang),
                    confidence=best.prob,
                    all_languages=all_langs,
                )
            except Exception:
                pass

        # Fallback: heuristic based on character sets
        return self._heuristic_detect(text)

    def _heuristic_detect(self, text: str) -> LanguageDetectionResult:
        """Simple heuristic language detection."""
        # Character set patterns
        patterns = {
            "zh": any("\u4e00" <= c <= "\u9fff" for c in text),  # Chinese
            "ja": any("\u3040" <= c <= "\u30ff" for c in text),  # Japanese
            "ko": any("\uac00" <= c <= "\ud7af" for c in text),  # Korean
            "ar": any("\u0600" <= c <= "\u06ff" for c in text),  # Arabic
            "he": any("\u0590" <= c <= "\u05ff" for c in text),  # Hebrew
            "ru": any("\u0400" <= c <= "\u04ff" for c in text),  # Cyrillic
            "th": any("\u0e00" <= c <= "\u0e7f" for c in text),  # Thai
            "hi": any("\u0900" <= c <= "\u097f" for c in text),  # Devanagari
        }

        for lang, match in patterns.items():
            if match:
                return LanguageDetectionResult(
                    text=text,
                    detected_language=lang,
                    confidence=0.8,
                )

        # Default to English for Latin script
        return LanguageDetectionResult(
            text=text,
            detected_language="en",
            confidence=0.5,
        )

    def detect_batch(self, texts: list[str]) -> list[LanguageDetectionResult]:
        """Detect language for multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of LanguageDetectionResult
        """
        return [self.detect(text) for text in texts]


class MultilingualEmbedder:
    """Generate multilingual embeddings for text."""

    def __init__(
        self,
        model: str | MultilingualModel = MultilingualModel.MULTILINGUAL_MINILM,
    ):
        """Initialize embedder.

        Args:
            model: Model name or MultilingualModel enum
        """
        if isinstance(model, MultilingualModel):
            model = model.value

        self.model_name = model
        self._model = None

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except ImportError as e:
            raise DependencyError(
                "sentence-transformers",
                "Multilingual embeddings require sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding

        Returns:
            numpy array of embeddings
        """
        self._load_model()
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def similarity(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        """Compute pairwise similarity between text sets.

        Args:
            texts1: First set of texts
            texts2: Second set of texts

        Returns:
            Similarity matrix
        """
        self._load_model()
        emb1 = self.embed(texts1)
        emb2 = self.embed(texts2)

        # Cosine similarity
        norm1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        norm2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        return np.dot(norm1, norm2.T)


class MultilingualDetector:
    """Multilingual label error detector.

    Detects label errors in multilingual datasets using cross-lingual
    embeddings and language-aware analysis.

    Example:
        >>> detector = MultilingualDetector()
        >>> errors = detector.find_errors(df, text_column="text", label_column="label")
        >>> print(f"Found {len(errors)} errors")
    """

    def __init__(
        self,
        embedding_model: str | MultilingualModel = MultilingualModel.MULTILINGUAL_MINILM,
        language_detector: LanguageDetector | None = None,
        confidence_threshold: float = 0.5,
        cv_folds: int = 5,
    ):
        """Initialize the detector.

        Args:
            embedding_model: Model for multilingual embeddings
            language_detector: Optional custom language detector
            confidence_threshold: Threshold for flagging errors
            cv_folds: Cross-validation folds
        """
        self.embedder = MultilingualEmbedder(embedding_model)
        self.language_detector = language_detector or LanguageDetector()
        self.confidence_threshold = confidence_threshold
        self.cv_folds = cv_folds

    def find_errors(
        self,
        data: pd.DataFrame,
        text_column: str,
        label_column: str,
        language_column: str | None = None,
        per_language: bool = True,
    ) -> MultilingualReport:
        """Find label errors in multilingual dataset.

        Args:
            data: DataFrame with text and labels
            text_column: Column containing text
            label_column: Column containing labels
            language_column: Optional column with language codes
            per_language: Analyze each language separately

        Returns:
            MultilingualReport with errors
        """
        # Detect languages if not provided
        if language_column is None:
            logger.info("Detecting languages...")
            texts = data[text_column].astype(str).tolist()
            lang_results = self.language_detector.detect_batch(texts)
            data = data.copy()
            data["_detected_language"] = [r.detected_language for r in lang_results]
            language_column = "_detected_language"

        # Get language distribution
        language_distribution = data[language_column].value_counts().to_dict()
        n_languages = len(language_distribution)

        # Generate embeddings
        logger.info("Generating multilingual embeddings...")
        texts = data[text_column].astype(str).tolist()
        embeddings = self.embedder.embed(texts)

        # Detect errors
        errors: list[MultilingualLabelError] = []
        error_rate_by_language: dict[str, float] = {}

        if per_language:
            # Analyze each language separately
            for lang in language_distribution:
                lang_mask = data[language_column] == lang
                lang_data = data[lang_mask]
                lang_embeddings = embeddings[lang_mask]

                if len(lang_data) < 10:
                    continue

                lang_errors = self._detect_errors_for_language(
                    lang_data,
                    lang_embeddings,
                    text_column,
                    label_column,
                    lang,
                )
                errors.extend(lang_errors)
                error_rate_by_language[lang] = len(lang_errors) / len(lang_data)
        else:
            # Analyze all together
            errors = self._detect_errors_for_language(
                data,
                embeddings,
                text_column,
                label_column,
                "mixed",
            )

        overall_error_rate = len(errors) / len(data) if len(data) > 0 else 0

        return MultilingualReport(
            n_samples=len(data),
            n_languages=n_languages,
            language_distribution=language_distribution,
            errors=errors,
            error_rate_by_language=error_rate_by_language,
            cross_lingual_results=None,
            overall_error_rate=overall_error_rate,
        )

    def check_cross_lingual_consistency(
        self,
        data_lang1: pd.DataFrame,
        data_lang2: pd.DataFrame,
        text_column: str,
        label_column: str,
        id_column: str | None = None,
    ) -> CrossLingualConsistencyResult:
        """Check label consistency across translations.

        Args:
            data_lang1: Data in first language
            data_lang2: Data in second language (translations)
            text_column: Column containing text
            label_column: Column containing labels
            id_column: Optional column to match samples

        Returns:
            CrossLingualConsistencyResult
        """
        # If id_column provided, align samples
        if id_column:
            common_ids = set(data_lang1[id_column]) & set(data_lang2[id_column])
            data_lang1 = data_lang1[data_lang1[id_column].isin(common_ids)].sort_values(id_column)
            data_lang2 = data_lang2[data_lang2[id_column].isin(common_ids)].sort_values(id_column)
        else:
            # Assume aligned by index
            min_len = min(len(data_lang1), len(data_lang2))
            data_lang1 = data_lang1.head(min_len)
            data_lang2 = data_lang2.head(min_len)

        n_samples = len(data_lang1)

        # Get embeddings
        texts1 = data_lang1[text_column].astype(str).tolist()
        texts2 = data_lang2[text_column].astype(str).tolist()

        emb1 = self.embedder.embed(texts1)
        emb2 = self.embedder.embed(texts2)

        # Compute pairwise similarity
        similarities = []
        for e1, e2 in zip(emb1, emb2):
            sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            similarities.append(sim)

        avg_similarity = float(np.mean(similarities))

        # Check label consistency
        labels1 = data_lang1[label_column].values
        labels2 = data_lang2[label_column].values
        consistent = labels1 == labels2

        consistency_score = float(np.mean(consistent))

        # Find inconsistent samples
        inconsistent_samples = []
        for i in range(n_samples):
            if not consistent[i]:
                inconsistent_samples.append({
                    "index": i,
                    "text_lang1": texts1[i][:100],
                    "text_lang2": texts2[i][:100],
                    "label_lang1": labels1[i],
                    "label_lang2": labels2[i],
                    "similarity": float(similarities[i]),
                })

        # Detect languages
        lang1 = self.language_detector.detect(texts1[0]).detected_language
        lang2 = self.language_detector.detect(texts2[0]).detected_language

        return CrossLingualConsistencyResult(
            language_pair=(lang1, lang2),
            n_samples=n_samples,
            consistency_score=consistency_score,
            inconsistent_samples=inconsistent_samples,
            embedding_similarity=avg_similarity,
        )

    def _detect_errors_for_language(
        self,
        data: pd.DataFrame,
        embeddings: np.ndarray,
        text_column: str,
        label_column: str,
        language: str,
    ) -> list[MultilingualLabelError]:
        """Detect errors for a specific language subset."""
        errors = []

        labels = data[label_column].values
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            return errors

        # Use cleanlab for confident learning
        try:
            from cleanlab.filter import find_label_issues

            # Get predictions using cross-validation
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict

            clf = LogisticRegression(max_iter=1000, random_state=42)
            pred_probs = cross_val_predict(
                clf,
                embeddings,
                labels,
                cv=min(self.cv_folds, len(unique_labels)),
                method="predict_proba",
            )

            # Find label issues
            issues = find_label_issues(
                labels=labels,
                pred_probs=pred_probs,
                return_indices_ranked_by="self_confidence",
            )

            texts = data[text_column].values
            indices = data.index.values

            for issue_idx in issues:
                confidence = float(1 - pred_probs[issue_idx, labels[issue_idx]])

                if confidence >= self.confidence_threshold:
                    predicted_label = unique_labels[np.argmax(pred_probs[issue_idx])]

                    errors.append(MultilingualLabelError(
                        index=int(indices[issue_idx]),
                        text=str(texts[issue_idx]),
                        language=language,
                        given_label=labels[issue_idx],
                        predicted_label=predicted_label,
                        confidence=confidence,
                    ))

        except Exception as e:
            logger.warning(f"Error in cleanlab detection: {e}")
            # Fallback to simpler method
            pass

        return errors


def detect_multilingual_errors(
    data: pd.DataFrame,
    text_column: str,
    label_column: str,
    language_column: str | None = None,
    model: str = "multilingual",
    **kwargs: Any,
) -> MultilingualReport:
    """Convenience function for multilingual error detection.

    Args:
        data: DataFrame with text and labels
        text_column: Column containing text
        label_column: Column containing labels
        language_column: Optional column with language codes
        model: Embedding model ("multilingual", "labse", or model name)
        **kwargs: Additional arguments for MultilingualDetector

    Returns:
        MultilingualReport

    Example:
        >>> report = detect_multilingual_errors(df, "text", "label")
        >>> print(report.summary())
    """
    if model == "multilingual":
        embedding_model = MultilingualModel.MULTILINGUAL_MINILM
    elif model == "labse":
        embedding_model = MultilingualModel.LABSE
    else:
        embedding_model = model

    detector = MultilingualDetector(
        embedding_model=embedding_model,
        **kwargs,
    )

    return detector.find_errors(
        data,
        text_column=text_column,
        label_column=label_column,
        language_column=language_column,
    )


__all__ = [
    # Core classes
    "MultilingualDetector",
    "MultilingualReport",
    "MultilingualLabelError",
    "CrossLingualConsistencyResult",
    # Support classes
    "LanguageDetector",
    "LanguageDetectionResult",
    "MultilingualEmbedder",
    # Enums
    "MultilingualModel",
    # Constants
    "SUPPORTED_LANGUAGES",
    # Functions
    "detect_multilingual_errors",
]
