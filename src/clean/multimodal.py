"""Multi-modal data quality analysis.

This module provides quality analysis for multi-modal datasets:
- Image-text pair consistency
- Cross-modal alignment scoring
- Modality-specific quality checks

Example:
    >>> from clean.multimodal import MultiModalAnalyzer
    >>>
    >>> analyzer = MultiModalAnalyzer()
    >>> report = analyzer.analyze(df, image_col="image", text_col="caption")
    >>> print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class ModalityType(Enum):
    """Types of data modalities."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"


class AlignmentIssueType(Enum):
    """Types of cross-modal alignment issues."""

    MISALIGNMENT = "misalignment"
    MISSING_MODALITY = "missing_modality"
    LOW_QUALITY_TEXT = "low_quality_text"
    LOW_QUALITY_IMAGE = "low_quality_image"
    DUPLICATE_CROSS_MODAL = "duplicate_cross_modal"
    CAPTION_TOO_SHORT = "caption_too_short"
    CAPTION_TOO_GENERIC = "caption_too_generic"


@dataclass
class ModalityQuality:
    """Quality metrics for a single modality."""

    modality: ModalityType
    n_samples: int
    n_missing: int
    n_issues: int
    quality_score: float
    issues: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "modality": self.modality.value,
            "n_samples": self.n_samples,
            "n_missing": self.n_missing,
            "n_issues": self.n_issues,
            "quality_score": self.quality_score,
        }


@dataclass
class AlignmentIssue:
    """A cross-modal alignment issue."""

    index: int
    issue_type: AlignmentIssueType
    severity: str
    description: str
    confidence: float
    affected_modalities: list[ModalityType] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "issue_type": self.issue_type.value,
            "severity": self.severity,
            "description": self.description,
            "confidence": self.confidence,
            "affected_modalities": [m.value for m in self.affected_modalities],
        }


@dataclass
class MultiModalReport:
    """Complete multi-modal quality report."""

    n_samples: int
    modalities: list[ModalityType]
    modality_quality: dict[ModalityType, ModalityQuality]
    alignment_score: float
    alignment_issues: list[AlignmentIssue]
    overall_quality: float
    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Multi-Modal Quality Report",
            "=" * 50,
            f"Total Samples: {self.n_samples:,}",
            f"Modalities: {', '.join(m.value for m in self.modalities)}",
            f"Overall Quality: {self.overall_quality:.1f}/100",
            f"Alignment Score: {self.alignment_score:.1f}/100",
            "",
            "Per-Modality Quality:",
        ]

        for modality, quality in self.modality_quality.items():
            lines.append(
                f"  {modality.value}: {quality.quality_score:.1f}/100 "
                f"({quality.n_issues} issues, {quality.n_missing} missing)"
            )

        if self.alignment_issues:
            lines.extend([
                "",
                f"Alignment Issues ({len(self.alignment_issues)}):",
            ])
            for issue in self.alignment_issues[:5]:
                lines.append(f"  - [{issue.severity}] {issue.description}")
            if len(self.alignment_issues) > 5:
                lines.append(f"  ... and {len(self.alignment_issues) - 5} more")

        if self.recommendations:
            lines.extend([
                "",
                "Recommendations:",
            ])
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "modalities": [m.value for m in self.modalities],
            "modality_quality": {
                m.value: q.to_dict() for m, q in self.modality_quality.items()
            },
            "alignment_score": self.alignment_score,
            "alignment_issues": [i.to_dict() for i in self.alignment_issues],
            "overall_quality": self.overall_quality,
            "recommendations": self.recommendations,
        }


class MultiModalAnalyzer:
    """Analyzer for multi-modal dataset quality.

    Analyzes quality and consistency across different modalities
    such as image-text pairs, audio-text, etc.
    """

    def __init__(
        self,
        min_caption_length: int = 10,
        max_caption_length: int = 500,
        alignment_threshold: float = 0.3,
    ):
        """Initialize the analyzer.

        Args:
            min_caption_length: Minimum acceptable caption length
            max_caption_length: Maximum acceptable caption length
            alignment_threshold: Threshold for alignment scoring
        """
        self.min_caption_length = min_caption_length
        self.max_caption_length = max_caption_length
        self.alignment_threshold = alignment_threshold

    def analyze(
        self,
        data: pd.DataFrame,
        image_column: str | None = None,
        text_column: str | None = None,
        audio_column: str | None = None,
        check_alignment: bool = True,
        check_quality: bool = True,
    ) -> MultiModalReport:
        """Analyze multi-modal dataset quality.

        Args:
            data: DataFrame with multi-modal data
            image_column: Column containing image paths/data
            text_column: Column containing text/captions
            audio_column: Column containing audio paths/data
            check_alignment: Check cross-modal alignment
            check_quality: Check per-modality quality

        Returns:
            MultiModalReport with analysis results
        """
        modalities: list[ModalityType] = []
        modality_quality: dict[ModalityType, ModalityQuality] = {}
        alignment_issues: list[AlignmentIssue] = []

        # Detect and analyze each modality
        if text_column and text_column in data.columns:
            modalities.append(ModalityType.TEXT)
            if check_quality:
                modality_quality[ModalityType.TEXT] = self._analyze_text_quality(
                    data, text_column
                )

        if image_column and image_column in data.columns:
            modalities.append(ModalityType.IMAGE)
            if check_quality:
                modality_quality[ModalityType.IMAGE] = self._analyze_image_quality(
                    data, image_column
                )

        if audio_column and audio_column in data.columns:
            modalities.append(ModalityType.AUDIO)
            if check_quality:
                modality_quality[ModalityType.AUDIO] = self._analyze_audio_quality(
                    data, audio_column
                )

        # Check cross-modal alignment
        alignment_score = 100.0
        if check_alignment and len(modalities) >= 2:
            if ModalityType.TEXT in modalities and ModalityType.IMAGE in modalities:
                issues, score = self._check_image_text_alignment(
                    data, image_column, text_column
                )
                alignment_issues.extend(issues)
                alignment_score = score

        # Calculate overall quality
        quality_scores = [q.quality_score for q in modality_quality.values()]
        modality_avg = np.mean(quality_scores) if quality_scores else 100.0

        overall_quality = 0.6 * modality_avg + 0.4 * alignment_score

        # Generate recommendations
        recommendations = self._generate_recommendations(
            modality_quality, alignment_issues, alignment_score
        )

        return MultiModalReport(
            n_samples=len(data),
            modalities=modalities,
            modality_quality=modality_quality,
            alignment_score=alignment_score,
            alignment_issues=alignment_issues,
            overall_quality=overall_quality,
            recommendations=recommendations,
        )

    def _analyze_text_quality(
        self,
        data: pd.DataFrame,
        text_column: str,
    ) -> ModalityQuality:
        """Analyze text/caption quality."""
        issues = []
        n_missing = 0
        n_samples = len(data)

        for idx, row in data.iterrows():
            text = row.get(text_column)

            # Check for missing
            if pd.isna(text) or str(text).strip() == "":
                n_missing += 1
                issues.append({
                    "index": int(idx),
                    "type": "missing",
                    "description": "Missing text",
                })
                continue

            text = str(text)

            # Check length
            if len(text) < self.min_caption_length:
                issues.append({
                    "index": int(idx),
                    "type": "too_short",
                    "description": f"Text too short ({len(text)} chars)",
                })

            # Check for generic captions
            generic_patterns = [
                "image", "picture", "photo", "a photo of",
                "this is", "an image", "untitled",
            ]
            if any(text.lower().startswith(p) for p in generic_patterns):
                issues.append({
                    "index": int(idx),
                    "type": "generic",
                    "description": "Generic/uninformative caption",
                })

        n_issues = len(issues)
        quality_score = max(0, 100 - (n_issues / max(n_samples, 1)) * 100)

        return ModalityQuality(
            modality=ModalityType.TEXT,
            n_samples=n_samples,
            n_missing=n_missing,
            n_issues=n_issues,
            quality_score=quality_score,
            issues=issues,
        )

    def _analyze_image_quality(
        self,
        data: pd.DataFrame,
        image_column: str,
    ) -> ModalityQuality:
        """Analyze image quality (basic checks without loading images)."""
        issues = []
        n_missing = 0
        n_samples = len(data)

        for idx, row in data.iterrows():
            image = row.get(image_column)

            # Check for missing
            if pd.isna(image) or str(image).strip() == "":
                n_missing += 1
                issues.append({
                    "index": int(idx),
                    "type": "missing",
                    "description": "Missing image",
                })
                continue

            image_str = str(image)

            # Basic path validation
            valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
            if not any(image_str.lower().endswith(ext) for ext in valid_extensions):
                # Could be base64 or URL - do basic check
                if not (
                    image_str.startswith("data:image/") or
                    image_str.startswith("http://") or
                    image_str.startswith("https://")
                ):
                    issues.append({
                        "index": int(idx),
                        "type": "invalid_format",
                        "description": "Possibly invalid image format",
                    })

        n_issues = len(issues)
        quality_score = max(0, 100 - (n_issues / max(n_samples, 1)) * 100)

        return ModalityQuality(
            modality=ModalityType.IMAGE,
            n_samples=n_samples,
            n_missing=n_missing,
            n_issues=n_issues,
            quality_score=quality_score,
            issues=issues,
        )

    def _analyze_audio_quality(
        self,
        data: pd.DataFrame,
        audio_column: str,
    ) -> ModalityQuality:
        """Analyze audio quality (basic checks)."""
        issues = []
        n_missing = 0
        n_samples = len(data)

        for idx, row in data.iterrows():
            audio = row.get(audio_column)

            if pd.isna(audio) or str(audio).strip() == "":
                n_missing += 1
                issues.append({
                    "index": int(idx),
                    "type": "missing",
                    "description": "Missing audio",
                })

        n_issues = len(issues)
        quality_score = max(0, 100 - (n_issues / max(n_samples, 1)) * 100)

        return ModalityQuality(
            modality=ModalityType.AUDIO,
            n_samples=n_samples,
            n_missing=n_missing,
            n_issues=n_issues,
            quality_score=quality_score,
            issues=issues,
        )

    def _check_image_text_alignment(
        self,
        data: pd.DataFrame,
        image_column: str | None,
        text_column: str | None,
    ) -> tuple[list[AlignmentIssue], float]:
        """Check alignment between images and text captions."""
        issues = []

        if not image_column or not text_column:
            return issues, 100.0

        n_samples = len(data)
        n_aligned = 0

        for idx, row in data.iterrows():
            image = row.get(image_column)
            text = row.get(text_column)

            # Check for missing modality
            image_missing = pd.isna(image) or str(image).strip() == ""
            text_missing = pd.isna(text) or str(text).strip() == ""

            if image_missing and not text_missing:
                issues.append(AlignmentIssue(
                    index=int(idx),
                    issue_type=AlignmentIssueType.MISSING_MODALITY,
                    severity="high",
                    description="Image missing but text present",
                    confidence=1.0,
                    affected_modalities=[ModalityType.IMAGE],
                ))
            elif text_missing and not image_missing:
                issues.append(AlignmentIssue(
                    index=int(idx),
                    issue_type=AlignmentIssueType.MISSING_MODALITY,
                    severity="high",
                    description="Text missing but image present",
                    confidence=1.0,
                    affected_modalities=[ModalityType.TEXT],
                ))
            elif not image_missing and not text_missing:
                # Check for very short/generic captions
                text_str = str(text)
                if len(text_str) < 10:
                    issues.append(AlignmentIssue(
                        index=int(idx),
                        issue_type=AlignmentIssueType.CAPTION_TOO_SHORT,
                        severity="medium",
                        description=f"Caption too short ({len(text_str)} chars)",
                        confidence=0.9,
                        affected_modalities=[ModalityType.TEXT, ModalityType.IMAGE],
                    ))
                else:
                    n_aligned += 1

        alignment_score = (n_aligned / max(n_samples, 1)) * 100

        return issues, alignment_score

    def _generate_recommendations(
        self,
        modality_quality: dict[ModalityType, ModalityQuality],
        alignment_issues: list[AlignmentIssue],
        alignment_score: float,
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check modality-specific issues
        for modality, quality in modality_quality.items():
            if quality.n_missing > 0:
                recommendations.append(
                    f"Fill {quality.n_missing} missing {modality.value} entries"
                )

            if quality.quality_score < 70:
                recommendations.append(
                    f"Review {modality.value} quality - score is {quality.quality_score:.1f}/100"
                )

        # Check alignment issues
        if alignment_score < 80:
            recommendations.append(
                f"Improve cross-modal alignment (current score: {alignment_score:.1f}/100)"
            )

        missing_issues = [
            i for i in alignment_issues
            if i.issue_type == AlignmentIssueType.MISSING_MODALITY
        ]
        if missing_issues:
            recommendations.append(
                f"Address {len(missing_issues)} samples with missing modalities"
            )

        short_caption_issues = [
            i for i in alignment_issues
            if i.issue_type == AlignmentIssueType.CAPTION_TOO_SHORT
        ]
        if short_caption_issues:
            recommendations.append(
                f"Improve {len(short_caption_issues)} captions that are too short"
            )

        if not recommendations:
            recommendations.append("Multi-modal quality is good. Continue monitoring.")

        return recommendations


def analyze_multimodal(
    data: pd.DataFrame,
    image_column: str | None = None,
    text_column: str | None = None,
    **kwargs: Any,
) -> MultiModalReport:
    """Analyze multi-modal dataset quality.

    Args:
        data: DataFrame with multi-modal data
        image_column: Column containing image paths
        text_column: Column containing text/captions
        **kwargs: Additional arguments for MultiModalAnalyzer

    Returns:
        MultiModalReport with analysis results
    """
    analyzer = MultiModalAnalyzer(**kwargs)
    return analyzer.analyze(data, image_column=image_column, text_column=text_column)
