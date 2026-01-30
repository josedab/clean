"""Tests for compliance report generation module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pytest

from clean.compliance import (
    ComplianceFramework,
    ComplianceReport,
    ComplianceReportGenerator,
    ComplianceStatus,
    RiskLevel,
    generate_compliance_report,
)


@dataclass
class MockQualityScore:
    """Mock quality score for testing."""

    overall: float = 75.0
    label_quality: float = 80.0
    duplicate_quality: float = 90.0
    outlier_quality: float = 85.0
    imbalance_quality: float = 70.0
    bias_quality: float = 75.0


@dataclass
class MockQualityReport:
    """Mock quality report for testing."""

    quality_score: MockQualityScore = None

    def __post_init__(self):
        if self.quality_score is None:
            self.quality_score = MockQualityScore()


class TestComplianceReportGenerator:
    """Tests for ComplianceReportGenerator class."""

    @pytest.fixture
    def mock_report(self) -> MockQualityReport:
        """Create mock quality report."""
        return MockQualityReport()

    @pytest.fixture
    def good_report(self) -> MockQualityReport:
        """Create high-quality mock report."""
        return MockQualityReport(
            quality_score=MockQualityScore(
                overall=95.0,
                label_quality=95.0,
                duplicate_quality=98.0,
                outlier_quality=96.0,
                imbalance_quality=90.0,
                bias_quality=92.0,
            )
        )

    @pytest.fixture
    def poor_report(self) -> MockQualityReport:
        """Create low-quality mock report."""
        return MockQualityReport(
            quality_score=MockQualityScore(
                overall=40.0,
                label_quality=35.0,
                duplicate_quality=50.0,
                outlier_quality=45.0,
                imbalance_quality=30.0,
                bias_quality=40.0,
            )
        )

    def test_generator_init(self) -> None:
        generator = ComplianceReportGenerator()
        assert generator is not None
        assert generator.framework == ComplianceFramework.EU_AI_ACT

    def test_generator_with_framework_string(self) -> None:
        generator = ComplianceReportGenerator(framework="nist_ai_rmf")
        assert generator.framework == ComplianceFramework.NIST_AI_RMF

    def test_generator_with_risk_level(self) -> None:
        generator = ComplianceReportGenerator(risk_level=RiskLevel.HIGH)
        assert generator.risk_level == RiskLevel.HIGH

    def test_generate_eu_ai_act_report(self, mock_report: MockQualityReport) -> None:
        generator = ComplianceReportGenerator(framework=ComplianceFramework.EU_AI_ACT)
        report = generator.generate(mock_report, dataset_name="Test Dataset")

        assert isinstance(report, ComplianceReport)
        assert report.framework == ComplianceFramework.EU_AI_ACT
        assert len(report.requirements) > 0

    def test_generate_nist_report(self, mock_report: MockQualityReport) -> None:
        generator = ComplianceReportGenerator(framework=ComplianceFramework.NIST_AI_RMF)
        report = generator.generate(mock_report, dataset_name="Test Dataset")

        assert report.framework == ComplianceFramework.NIST_AI_RMF
        assert len(report.requirements) > 0

    def test_generate_custom_report(self, mock_report: MockQualityReport) -> None:
        custom_reqs = [
            {"id": "CUSTOM-001", "title": "Custom Requirement", "score": 80},
        ]
        generator = ComplianceReportGenerator(
            framework=ComplianceFramework.CUSTOM,
            custom_requirements=custom_reqs,
        )
        report = generator.generate(mock_report)

        assert len(report.requirements) == 1
        assert report.requirements[0].requirement_id == "CUSTOM-001"

    def test_good_report_compliant(self, good_report: MockQualityReport) -> None:
        generator = ComplianceReportGenerator()
        report = generator.generate(good_report)

        # High quality should result in mostly compliant status
        assert report.overall_score > 70
        compliant_count = sum(
            1 for r in report.requirements
            if r.status == ComplianceStatus.COMPLIANT
        )
        assert compliant_count > 0

    def test_poor_report_non_compliant(self, poor_report: MockQualityReport) -> None:
        generator = ComplianceReportGenerator()
        report = generator.generate(poor_report)

        # Low quality should result in non-compliant findings
        assert report.overall_score < 70
        non_compliant_count = sum(
            1 for r in report.requirements
            if r.status == ComplianceStatus.NON_COMPLIANT
        )
        assert non_compliant_count > 0

    def test_report_has_executive_summary(self, mock_report: MockQualityReport) -> None:
        generator = ComplianceReportGenerator()
        report = generator.generate(mock_report)

        assert report.executive_summary is not None
        assert len(report.executive_summary) > 0

    def test_report_has_audit_trail(self, mock_report: MockQualityReport) -> None:
        generator = ComplianceReportGenerator()
        report = generator.generate(mock_report)

        assert len(report.audit_trail) > 0
        assert "timestamp" in report.audit_trail[0]

    def test_additional_context(self, mock_report: MockQualityReport) -> None:
        generator = ComplianceReportGenerator()
        context = {
            "documentation_provided": True,
            "origin_documented": True,
        }
        report = generator.generate(mock_report, additional_context=context)

        # With documentation provided, those requirements should score higher
        assert report is not None


class TestComplianceReport:
    """Tests for ComplianceReport class."""

    @pytest.fixture
    def sample_report(self) -> ComplianceReport:
        """Create sample compliance report."""
        generator = ComplianceReportGenerator()
        mock = MockQualityReport()
        return generator.generate(mock, dataset_name="Sample Dataset")

    def test_report_summary(self, sample_report: ComplianceReport) -> None:
        summary = sample_report.summary()

        assert "COMPLIANCE ASSESSMENT REPORT" in summary
        assert "Sample Dataset" in summary
        assert "EU AI ACT" in summary

    def test_report_to_dict(self, sample_report: ComplianceReport) -> None:
        d = sample_report.to_dict()

        assert "framework" in d
        assert "overall_score" in d
        assert "requirements" in d
        assert isinstance(d["requirements"], list)

    def test_report_to_markdown(self, sample_report: ComplianceReport) -> None:
        md = sample_report.to_markdown()

        assert "# Compliance Assessment Report" in md
        assert "| ID | Requirement | Status | Score |" in md

    def test_report_to_html(self, sample_report: ComplianceReport) -> None:
        html = sample_report.to_html()

        assert "<!DOCTYPE html>" in html
        assert "Sample Dataset" in html


class TestGenerateComplianceReport:
    """Tests for generate_compliance_report convenience function."""

    def test_function_works(self) -> None:
        mock = MockQualityReport()
        report = generate_compliance_report(mock, framework="eu_ai_act")

        assert isinstance(report, ComplianceReport)

    def test_function_with_nist(self) -> None:
        mock = MockQualityReport()
        report = generate_compliance_report(mock, framework="nist_ai_rmf")

        assert report.framework == ComplianceFramework.NIST_AI_RMF


class TestComplianceStatus:
    """Tests for compliance status determination."""

    def test_score_to_status_compliant(self) -> None:
        generator = ComplianceReportGenerator()
        status = generator._score_to_status(85)
        assert status == ComplianceStatus.COMPLIANT

    def test_score_to_status_partial(self) -> None:
        generator = ComplianceReportGenerator()
        status = generator._score_to_status(65)
        assert status == ComplianceStatus.PARTIALLY_COMPLIANT

    def test_score_to_status_non_compliant(self) -> None:
        generator = ComplianceReportGenerator()
        status = generator._score_to_status(40)
        assert status == ComplianceStatus.NON_COMPLIANT


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_quality_score(self) -> None:
        class EmptyReport:
            pass

        generator = ComplianceReportGenerator()
        report = generator.generate(EmptyReport())

        assert report is not None

    def test_none_values(self) -> None:
        @dataclass
        class PartialReport:
            quality_score: None = None

        generator = ComplianceReportGenerator()
        report = generator.generate(PartialReport())

        assert report is not None

    def test_empty_custom_requirements(self) -> None:
        generator = ComplianceReportGenerator(
            framework=ComplianceFramework.CUSTOM,
            custom_requirements=[],
        )
        mock = MockQualityReport()
        report = generator.generate(mock)

        assert len(report.requirements) == 0
