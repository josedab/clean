"""Tests for multi-modal analysis module."""

from __future__ import annotations

import pandas as pd
import pytest

from clean.multimodal import (
    AlignmentIssueType,
    ModalityType,
    MultiModalAnalyzer,
    MultiModalReport,
    analyze_multimodal,
)


class TestMultiModalAnalyzer:
    """Tests for MultiModalAnalyzer class."""

    @pytest.fixture
    def image_text_data(self) -> pd.DataFrame:
        """Create sample image-text dataset."""
        return pd.DataFrame({
            "image": [
                "img1.jpg",
                "img2.png",
                "img3.jpg",
                "",  # Missing
                "img5.jpg",
            ],
            "caption": [
                "A beautiful sunset over the ocean",
                "A cat sitting on a windowsill",
                "",  # Missing
                "This text has no image",
                "Short",  # Too short
            ],
        })

    @pytest.fixture
    def good_data(self) -> pd.DataFrame:
        """Create high-quality image-text dataset."""
        return pd.DataFrame({
            "image": [f"img{i}.jpg" for i in range(10)],
            "caption": [
                "A detailed description of image one with relevant context",
                "Another comprehensive caption describing the second image",
                "The third image shows an interesting scene with details",
                "A well-written caption for the fourth photograph",
                "Descriptive text explaining what appears in image five",
                "Caption six provides context for the accompanying image",
                "The seventh image is described thoroughly here",
                "An informative caption for the eighth photograph",
                "Detailed description of what image nine contains",
                "The final image is accompanied by this description",
            ],
        })

    def test_analyzer_init(self) -> None:
        analyzer = MultiModalAnalyzer()
        assert analyzer is not None
        assert analyzer.min_caption_length == 10

    def test_analyze_image_text(self, image_text_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            image_text_data,
            image_column="image",
            text_column="caption",
        )

        assert isinstance(report, MultiModalReport)
        assert ModalityType.IMAGE in report.modalities
        assert ModalityType.TEXT in report.modalities

    def test_detects_missing_image(self, image_text_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            image_text_data,
            image_column="image",
            text_column="caption",
        )

        # Should detect missing image
        assert ModalityType.IMAGE in report.modality_quality
        image_quality = report.modality_quality[ModalityType.IMAGE]
        assert image_quality.n_missing >= 1

    def test_detects_missing_text(self, image_text_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            image_text_data,
            image_column="image",
            text_column="caption",
        )

        text_quality = report.modality_quality[ModalityType.TEXT]
        assert text_quality.n_missing >= 1

    def test_alignment_issues_detected(self, image_text_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            image_text_data,
            image_column="image",
            text_column="caption",
        )

        # Should detect alignment issues
        assert len(report.alignment_issues) > 0

    def test_short_caption_detected(self, image_text_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            image_text_data,
            image_column="image",
            text_column="caption",
        )

        short_issues = [
            i for i in report.alignment_issues
            if i.issue_type == AlignmentIssueType.CAPTION_TOO_SHORT
        ]
        assert len(short_issues) >= 1

    def test_good_data_high_quality(self, good_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            good_data,
            image_column="image",
            text_column="caption",
        )

        assert report.overall_quality > 70
        assert report.alignment_score > 80

    def test_modality_quality_scores(self, image_text_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            image_text_data,
            image_column="image",
            text_column="caption",
        )

        for modality, quality in report.modality_quality.items():
            assert 0 <= quality.quality_score <= 100

    def test_report_summary(self, image_text_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            image_text_data,
            image_column="image",
            text_column="caption",
        )

        summary = report.summary()
        assert "Multi-Modal Quality Report" in summary
        assert "Alignment Score" in summary

    def test_report_to_dict(self, image_text_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            image_text_data,
            image_column="image",
            text_column="caption",
        )

        d = report.to_dict()
        assert "modalities" in d
        assert "alignment_score" in d
        assert "overall_quality" in d

    def test_recommendations_generated(self, image_text_data: pd.DataFrame) -> None:
        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(
            image_text_data,
            image_column="image",
            text_column="caption",
        )

        assert len(report.recommendations) > 0

    def test_custom_thresholds(self) -> None:
        analyzer = MultiModalAnalyzer(
            min_caption_length=20,
            max_caption_length=100,
        )
        assert analyzer.min_caption_length == 20


class TestAnalyzeMultimodal:
    """Tests for analyze_multimodal convenience function."""

    def test_function_works(self) -> None:
        df = pd.DataFrame({
            "image": ["img1.jpg", "img2.jpg"],
            "caption": ["A detailed caption here", "Another good caption"],
        })

        report = analyze_multimodal(df, image_column="image", text_column="caption")
        assert isinstance(report, MultiModalReport)


class TestSingleModality:
    """Tests for single modality analysis."""

    def test_text_only(self) -> None:
        df = pd.DataFrame({
            "text": ["Some text content", "More text here", ""],
        })

        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(df, text_column="text")

        assert ModalityType.TEXT in report.modalities
        assert ModalityType.IMAGE not in report.modalities

    def test_image_only(self) -> None:
        df = pd.DataFrame({
            "image": ["img1.jpg", "img2.png", ""],
        })

        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(df, image_column="image")

        assert ModalityType.IMAGE in report.modalities
        assert ModalityType.TEXT not in report.modalities


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"image": [], "caption": []})

        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(df, image_column="image", text_column="caption")

        assert report.n_samples == 0

    def test_all_missing(self) -> None:
        df = pd.DataFrame({
            "image": ["", "", ""],
            "caption": ["", "", ""],
        })

        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(df, image_column="image", text_column="caption")

        assert report.modality_quality[ModalityType.IMAGE].n_missing == 3
        assert report.modality_quality[ModalityType.TEXT].n_missing == 3

    def test_none_values(self) -> None:
        df = pd.DataFrame({
            "image": ["img1.jpg", None, "img3.jpg"],
            "caption": [None, "A caption", "Another caption"],
        })

        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(df, image_column="image", text_column="caption")

        assert report is not None

    def test_no_modality_columns(self) -> None:
        df = pd.DataFrame({"other": [1, 2, 3]})

        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(df)

        assert len(report.modalities) == 0

    def test_generic_captions_detected(self) -> None:
        df = pd.DataFrame({
            "image": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "caption": [
                "image of something",
                "a photo of a thing",
                "A detailed and specific description of the scene",
            ],
        })

        analyzer = MultiModalAnalyzer()
        report = analyzer.analyze(df, image_column="image", text_column="caption")

        text_quality = report.modality_quality[ModalityType.TEXT]
        generic_issues = [i for i in text_quality.issues if i.get("type") == "generic"]
        assert len(generic_issues) >= 1
