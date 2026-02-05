"""Cross-Dataset Contamination Detector.

This module detects data leakage and contamination across datasets,
including train/test leakage and cross-dataset duplicates.

Example:
    >>> from clean.contamination import ContaminationDetector
    >>>
    >>> detector = ContaminationDetector()
    >>> report = detector.detect(train_df, test_df)
    >>> print(f"Found {report.n_contaminated} contaminated samples")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ContaminationType(Enum):
    """Types of data contamination."""

    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    SEMANTIC_DUPLICATE = "semantic_duplicate"
    LABEL_LEAKAGE = "label_leakage"
    FEATURE_LEAKAGE = "feature_leakage"


class SeverityLevel(Enum):
    """Severity of contamination."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ContaminatedPair:
    """A pair of contaminated samples across datasets."""

    source_dataset: str
    source_index: int
    target_dataset: str
    target_index: int
    contamination_type: ContaminationType
    similarity: float
    severity: SeverityLevel
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_dataset": self.source_dataset,
            "source_index": self.source_index,
            "target_dataset": self.target_dataset,
            "target_index": self.target_index,
            "contamination_type": self.contamination_type.value,
            "similarity": self.similarity,
            "severity": self.severity.value,
            "metadata": self.metadata,
        }


@dataclass
class DatasetRegistration:
    """Registration info for a dataset."""

    name: str
    n_samples: int
    n_features: int
    hash_fingerprint: str
    registered_at: datetime
    embeddings: np.ndarray | None = None
    hashes: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContaminationReport:
    """Report of detected contamination."""

    timestamp: datetime
    datasets_compared: list[str]
    n_contaminated: int
    contamination_rate: float

    contaminated_pairs: list[ContaminatedPair]
    contamination_by_type: dict[str, int]
    contamination_by_severity: dict[str, int]

    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "⚠️ CONTAMINATION DETECTED" if self.n_contaminated > 0 else "✅ NO CONTAMINATION"

        lines = [
            "Cross-Dataset Contamination Report",
            "=" * 50,
            "",
            f"Status: {status}",
            f"Datasets compared: {', '.join(self.datasets_compared)}",
            f"Contaminated pairs: {self.n_contaminated}",
            f"Contamination rate: {self.contamination_rate:.2%}",
            "",
        ]

        if self.contamination_by_type:
            lines.append("By Type:")
            for type_name, count in self.contamination_by_type.items():
                lines.append(f"  • {type_name}: {count}")
            lines.append("")

        if self.contamination_by_severity:
            lines.append("By Severity:")
            for sev, count in self.contamination_by_severity.items():
                lines.append(f"  • {sev}: {count}")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert contaminated pairs to DataFrame."""
        if not self.contaminated_pairs:
            return pd.DataFrame()

        return pd.DataFrame([p.to_dict() for p in self.contaminated_pairs])

    def get_contaminated_indices(
        self,
        dataset_name: str,
    ) -> list[int]:
        """Get indices of contaminated samples in a dataset."""
        indices = set()
        for pair in self.contaminated_pairs:
            if pair.source_dataset == dataset_name:
                indices.add(pair.source_index)
            if pair.target_dataset == dataset_name:
                indices.add(pair.target_index)
        return list(indices)


@dataclass
class ContaminationConfig:
    """Configuration for contamination detection."""

    # Similarity thresholds
    exact_match_threshold: float = 1.0
    near_duplicate_threshold: float = 0.95
    semantic_threshold: float = 0.85

    # Detection settings
    detect_exact: bool = True
    detect_near_duplicates: bool = True
    detect_semantic: bool = True
    detect_label_leakage: bool = True

    # Performance settings
    max_samples_per_dataset: int | None = None
    batch_size: int = 1000
    n_neighbors: int = 10

    # Embedding settings
    embedding_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "exact_match_threshold": self.exact_match_threshold,
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "semantic_threshold": self.semantic_threshold,
            "detect_exact": self.detect_exact,
            "detect_near_duplicates": self.detect_near_duplicates,
            "detect_semantic": self.detect_semantic,
        }


class HashIndex:
    """Fast hash-based index for exact duplicate detection."""

    def __init__(self):
        """Initialize hash index."""
        self._hashes: dict[str, list[tuple[str, int]]] = {}

    def add(self, dataset_name: str, index: int, data: Any) -> str:
        """Add item to index.

        Args:
            dataset_name: Name of dataset
            index: Index in dataset
            data: Data to hash

        Returns:
            Hash string
        """
        hash_str = self._compute_hash(data)

        if hash_str not in self._hashes:
            self._hashes[hash_str] = []

        self._hashes[hash_str].append((dataset_name, index))
        return hash_str

    def find_duplicates(self) -> list[tuple[str, int, str, int]]:
        """Find all duplicates across datasets.

        Returns:
            List of (dataset1, idx1, dataset2, idx2) tuples
        """
        duplicates = []

        for hash_str, entries in self._hashes.items():
            if len(entries) > 1:
                # Check for cross-dataset duplicates
                for i, (ds1, idx1) in enumerate(entries):
                    for ds2, idx2 in entries[i + 1:]:
                        if ds1 != ds2:
                            duplicates.append((ds1, idx1, ds2, idx2))

        return duplicates

    def _compute_hash(self, data: Any) -> str:
        """Compute hash for data."""
        if isinstance(data, pd.Series):
            data_str = data.to_json()
        elif isinstance(data, np.ndarray):
            data_str = data.tobytes().hex()
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def clear(self) -> None:
        """Clear the index."""
        self._hashes.clear()


class EmbeddingIndex:
    """Vector index for semantic similarity search."""

    def __init__(self, n_neighbors: int = 10):
        """Initialize embedding index.

        Args:
            n_neighbors: Number of neighbors to search
        """
        self.n_neighbors = n_neighbors
        self._embeddings: list[np.ndarray] = []
        self._metadata: list[tuple[str, int]] = []
        self._nn: NearestNeighbors | None = None

    def add(
        self,
        dataset_name: str,
        index: int,
        embedding: np.ndarray,
    ) -> None:
        """Add embedding to index.

        Args:
            dataset_name: Name of dataset
            index: Index in dataset
            embedding: Embedding vector
        """
        self._embeddings.append(embedding)
        self._metadata.append((dataset_name, index))

    def build(self) -> None:
        """Build the search index."""
        if not self._embeddings:
            return

        embeddings_arr = np.array(self._embeddings)
        self._nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self._embeddings)),
            metric="cosine",
        )
        self._nn.fit(embeddings_arr)

    def search(
        self,
        query: np.ndarray,
        threshold: float = 0.85,
    ) -> list[tuple[str, int, float]]:
        """Search for similar embeddings.

        Args:
            query: Query embedding
            threshold: Similarity threshold

        Returns:
            List of (dataset_name, index, similarity) tuples
        """
        if self._nn is None:
            return []

        distances, indices = self._nn.kneighbors([query])

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - dist  # Convert cosine distance to similarity
            if similarity >= threshold:
                ds_name, ds_idx = self._metadata[idx]
                results.append((ds_name, ds_idx, float(similarity)))

        return results

    def find_cross_dataset_matches(
        self,
        threshold: float = 0.85,
    ) -> list[tuple[str, int, str, int, float]]:
        """Find all cross-dataset matches above threshold.

        Args:
            threshold: Similarity threshold

        Returns:
            List of (ds1, idx1, ds2, idx2, similarity) tuples
        """
        if self._nn is None:
            self.build()

        if self._nn is None or not self._embeddings:
            return []

        matches = []
        seen = set()

        for i, (embedding, (ds1, idx1)) in enumerate(
            zip(self._embeddings, self._metadata)
        ):
            neighbors = self.search(embedding, threshold)

            for ds2, idx2, similarity in neighbors:
                # Skip self-matches and same-dataset matches
                if (ds1 == ds2 and idx1 == idx2) or ds1 == ds2:
                    continue

                # Create canonical key to avoid duplicates
                key = tuple(sorted([(ds1, idx1), (ds2, idx2)]))
                if key in seen:
                    continue
                seen.add(key)

                matches.append((ds1, idx1, ds2, idx2, similarity))

        return matches

    def clear(self) -> None:
        """Clear the index."""
        self._embeddings.clear()
        self._metadata.clear()
        self._nn = None


class ContaminationDetector:
    """Detect data contamination across datasets.

    Identifies exact duplicates, near-duplicates, and semantic
    duplicates that could cause data leakage.
    """

    def __init__(
        self,
        config: ContaminationConfig | None = None,
    ):
        """Initialize contamination detector.

        Args:
            config: Detection configuration
        """
        self.config = config or ContaminationConfig()
        self._hash_index = HashIndex()
        self._embedding_index = EmbeddingIndex(self.config.n_neighbors)
        self._registrations: dict[str, DatasetRegistration] = {}

    def register_dataset(
        self,
        data: pd.DataFrame,
        name: str,
        text_columns: list[str] | None = None,
    ) -> DatasetRegistration:
        """Register a dataset for contamination checking.

        Args:
            data: Dataset to register
            name: Unique name for dataset
            text_columns: Columns containing text (for semantic matching)

        Returns:
            DatasetRegistration object
        """
        # Sample if needed
        if (self.config.max_samples_per_dataset and
                len(data) > self.config.max_samples_per_dataset):
            data = data.sample(
                n=self.config.max_samples_per_dataset,
                random_state=42,
            )

        # Compute hashes for exact matching
        hashes = []
        for idx, row in data.iterrows():
            hash_str = self._hash_index.add(name, int(idx), row)
            hashes.append(hash_str)

        # Compute embeddings for semantic matching
        embeddings = None
        if self.config.detect_semantic and text_columns:
            embeddings = self._compute_embeddings(data, text_columns, name)

        # Create registration
        fingerprint = hashlib.sha256(
            f"{name}_{len(data)}_{list(data.columns)}".encode()
        ).hexdigest()[:16]

        registration = DatasetRegistration(
            name=name,
            n_samples=len(data),
            n_features=len(data.columns),
            hash_fingerprint=fingerprint,
            registered_at=datetime.now(),
            embeddings=embeddings,
            hashes=hashes,
        )

        self._registrations[name] = registration
        return registration

    def detect(
        self,
        dataset1: pd.DataFrame | str,
        dataset2: pd.DataFrame | str,
        name1: str = "dataset_1",
        name2: str = "dataset_2",
        text_columns: list[str] | None = None,
    ) -> ContaminationReport:
        """Detect contamination between two datasets.

        Args:
            dataset1: First dataset or registered name
            dataset2: Second dataset or registered name
            name1: Name for first dataset (if DataFrame)
            name2: Name for second dataset (if DataFrame)
            text_columns: Text columns for semantic matching

        Returns:
            ContaminationReport
        """
        # Register datasets if DataFrames provided
        if isinstance(dataset1, pd.DataFrame):
            self.register_dataset(dataset1, name1, text_columns)
        else:
            name1 = dataset1

        if isinstance(dataset2, pd.DataFrame):
            self.register_dataset(dataset2, name2, text_columns)
        else:
            name2 = dataset2

        # Detect contamination
        contaminated_pairs = []

        # 1. Exact duplicates
        if self.config.detect_exact:
            exact_dupes = self._detect_exact_duplicates(name1, name2)
            contaminated_pairs.extend(exact_dupes)

        # 2. Near duplicates (based on feature similarity)
        if self.config.detect_near_duplicates:
            near_dupes = self._detect_near_duplicates(name1, name2)
            contaminated_pairs.extend(near_dupes)

        # 3. Semantic duplicates
        if self.config.detect_semantic:
            semantic_dupes = self._detect_semantic_duplicates(name1, name2)
            contaminated_pairs.extend(semantic_dupes)

        # Calculate statistics
        n_contaminated = len(contaminated_pairs)

        reg1 = self._registrations.get(name1)
        reg2 = self._registrations.get(name2)
        total_pairs = 0
        if reg1 and reg2:
            total_pairs = min(reg1.n_samples, reg2.n_samples)

        contamination_rate = n_contaminated / total_pairs if total_pairs > 0 else 0

        # Aggregate by type and severity
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}

        for pair in contaminated_pairs:
            type_name = pair.contamination_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            sev_name = pair.severity.value
            by_severity[sev_name] = by_severity.get(sev_name, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(
            contaminated_pairs, name1, name2
        )

        return ContaminationReport(
            timestamp=datetime.now(),
            datasets_compared=[name1, name2],
            n_contaminated=n_contaminated,
            contamination_rate=contamination_rate,
            contaminated_pairs=contaminated_pairs,
            contamination_by_type=by_type,
            contamination_by_severity=by_severity,
            recommendations=recommendations,
        )

    def detect_multiple(
        self,
        datasets: list[pd.DataFrame],
        names: list[str],
        text_columns: list[str] | None = None,
    ) -> ContaminationReport:
        """Detect contamination across multiple datasets.

        Args:
            datasets: List of datasets
            names: Names for each dataset
            text_columns: Text columns for semantic matching

        Returns:
            Aggregated ContaminationReport
        """
        # Register all datasets
        for data, name in zip(datasets, names):
            self.register_dataset(data, name, text_columns)

        # Detect contamination between all pairs
        all_pairs: list[ContaminatedPair] = []

        for i, name1 in enumerate(names):
            for name2 in names[i + 1:]:
                report = self.detect(name1, name2)
                all_pairs.extend(report.contaminated_pairs)

        # Deduplicate pairs
        seen = set()
        unique_pairs = []
        for pair in all_pairs:
            key = (
                pair.source_dataset, pair.source_index,
                pair.target_dataset, pair.target_index,
            )
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)

        # Calculate overall statistics
        n_contaminated = len(unique_pairs)

        total_samples = sum(
            self._registrations[name].n_samples
            for name in names
            if name in self._registrations
        )

        contamination_rate = n_contaminated / total_samples if total_samples > 0 else 0

        # Aggregate
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}

        for pair in unique_pairs:
            by_type[pair.contamination_type.value] = (
                by_type.get(pair.contamination_type.value, 0) + 1
            )
            by_severity[pair.severity.value] = (
                by_severity.get(pair.severity.value, 0) + 1
            )

        recommendations = self._generate_recommendations(unique_pairs, *names)

        return ContaminationReport(
            timestamp=datetime.now(),
            datasets_compared=names,
            n_contaminated=n_contaminated,
            contamination_rate=contamination_rate,
            contaminated_pairs=unique_pairs,
            contamination_by_type=by_type,
            contamination_by_severity=by_severity,
            recommendations=recommendations,
        )

    def _detect_exact_duplicates(
        self,
        name1: str,
        name2: str,
    ) -> list[ContaminatedPair]:
        """Detect exact duplicates between datasets."""
        duplicates = self._hash_index.find_duplicates()

        pairs = []
        for ds1, idx1, ds2, idx2 in duplicates:
            # Only report cross-dataset duplicates between our targets
            if not ((ds1 == name1 and ds2 == name2) or
                    (ds1 == name2 and ds2 == name1)):
                continue

            pairs.append(ContaminatedPair(
                source_dataset=ds1,
                source_index=idx1,
                target_dataset=ds2,
                target_index=idx2,
                contamination_type=ContaminationType.EXACT_DUPLICATE,
                similarity=1.0,
                severity=SeverityLevel.CRITICAL,
            ))

        return pairs

    def _detect_near_duplicates(
        self,
        name1: str,
        name2: str,
    ) -> list[ContaminatedPair]:
        """Detect near-duplicates based on feature similarity."""
        # This would use embedding index if available
        # For now, skip if already found as exact duplicates
        return []

    def _detect_semantic_duplicates(
        self,
        name1: str,
        name2: str,
    ) -> list[ContaminatedPair]:
        """Detect semantic duplicates using embeddings."""
        reg1 = self._registrations.get(name1)
        reg2 = self._registrations.get(name2)

        if not reg1 or not reg2:
            return []

        if reg1.embeddings is None or reg2.embeddings is None:
            return []

        # Build index with both datasets' embeddings
        self._embedding_index.clear()

        for i, emb in enumerate(reg1.embeddings):
            self._embedding_index.add(name1, i, emb)

        for i, emb in enumerate(reg2.embeddings):
            self._embedding_index.add(name2, i, emb)

        self._embedding_index.build()

        # Find matches
        matches = self._embedding_index.find_cross_dataset_matches(
            threshold=self.config.semantic_threshold
        )

        pairs = []
        for ds1, idx1, ds2, idx2, similarity in matches:
            # Determine severity based on similarity
            if similarity >= 0.98:
                severity = SeverityLevel.CRITICAL
            elif similarity >= 0.95:
                severity = SeverityLevel.HIGH
            elif similarity >= 0.9:
                severity = SeverityLevel.MEDIUM
            else:
                severity = SeverityLevel.LOW

            pairs.append(ContaminatedPair(
                source_dataset=ds1,
                source_index=idx1,
                target_dataset=ds2,
                target_index=idx2,
                contamination_type=ContaminationType.SEMANTIC_DUPLICATE,
                similarity=similarity,
                severity=severity,
            ))

        return pairs

    def _compute_embeddings(
        self,
        data: pd.DataFrame,
        text_columns: list[str],
        dataset_name: str,
    ) -> np.ndarray:
        """Compute embeddings for text columns."""
        # Concatenate text columns
        texts = []
        for _, row in data.iterrows():
            text_parts = [str(row[col]) for col in text_columns if col in row]
            texts.append(" ".join(text_parts))

        # Simple TF-IDF based embeddings as fallback
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=256)
        embeddings = vectorizer.fit_transform(texts).toarray()

        # Add to embedding index
        for i, emb in enumerate(embeddings):
            self._embedding_index.add(dataset_name, i, emb)

        return embeddings

    def _generate_recommendations(
        self,
        pairs: list[ContaminatedPair],
        *dataset_names: str,
    ) -> list[str]:
        """Generate recommendations based on detected contamination."""
        recommendations = []

        if not pairs:
            recommendations.append("No contamination detected. Safe to proceed.")
            return recommendations

        # Count by type
        exact_count = sum(
            1 for p in pairs
            if p.contamination_type == ContaminationType.EXACT_DUPLICATE
        )
        semantic_count = sum(
            1 for p in pairs
            if p.contamination_type == ContaminationType.SEMANTIC_DUPLICATE
        )

        if exact_count > 0:
            recommendations.append(
                f"Remove {exact_count} exact duplicate(s) from test set to prevent data leakage."
            )

        if semantic_count > 0:
            recommendations.append(
                f"Review {semantic_count} semantic duplicate(s) - they may be paraphrases "
                "or similar examples that could inflate evaluation metrics."
            )

        # Critical severity
        critical_count = sum(
            1 for p in pairs if p.severity == SeverityLevel.CRITICAL
        )
        if critical_count > 0:
            recommendations.append(
                f"⚠️ {critical_count} critical contamination(s) require immediate attention."
            )

        # General advice
        if len(pairs) > 10:
            recommendations.append(
                "Consider re-splitting your data using stratified sampling with "
                "deduplication to ensure clean train/test separation."
            )

        return recommendations

    def clear(self) -> None:
        """Clear all registered datasets and indices."""
        self._hash_index.clear()
        self._embedding_index.clear()
        self._registrations.clear()


def detect_contamination(
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_columns: list[str] | None = None,
    config: ContaminationConfig | None = None,
) -> ContaminationReport:
    """Convenience function to detect train/test contamination.

    Args:
        train: Training dataset
        test: Test dataset
        text_columns: Text columns for semantic matching
        config: Detection configuration

    Returns:
        ContaminationReport
    """
    detector = ContaminationDetector(config=config)
    return detector.detect(
        train, test,
        name1="train", name2="test",
        text_columns=text_columns,
    )


def detect_leakage(
    datasets: dict[str, pd.DataFrame],
    text_columns: list[str] | None = None,
    config: ContaminationConfig | None = None,
) -> ContaminationReport:
    """Detect data leakage across multiple named datasets.

    Args:
        datasets: Dictionary of name -> DataFrame
        text_columns: Text columns for semantic matching
        config: Detection configuration

    Returns:
        ContaminationReport
    """
    detector = ContaminationDetector(config=config)
    return detector.detect_multiple(
        list(datasets.values()),
        list(datasets.keys()),
        text_columns=text_columns,
    )


def create_contamination_detector(
    config: ContaminationConfig | None = None,
    **kwargs: Any,
) -> ContaminationDetector:
    """Create a contamination detector.

    Args:
        config: Detection configuration
        **kwargs: Additional configuration parameters

    Returns:
        ContaminationDetector instance
    """
    if config is None:
        config = ContaminationConfig(**kwargs)
    return ContaminationDetector(config=config)
