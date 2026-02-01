"""Compliance report generation for regulatory requirements.

This module generates compliance documentation for:
- EU AI Act data quality requirements
- NIST AI Risk Management Framework
- Custom compliance frameworks

Example:
    >>> from clean.compliance import ComplianceReportGenerator
    >>>
    >>> generator = ComplianceReportGenerator(framework="eu_ai_act")
    >>> report = generator.generate(quality_report, dataset_info)
    >>> report.export("compliance_report.pdf")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    EU_AI_ACT = "eu_ai_act"
    NIST_AI_RMF = "nist_ai_rmf"
    ISO_IEC_22989 = "iso_iec_22989"
    CUSTOM = "custom"


class RiskLevel(Enum):
    """AI system risk levels (EU AI Act)."""

    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


class ComplianceStatus(Enum):
    """Compliance status for a requirement."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class ComplianceRequirement:
    """A single compliance requirement."""

    requirement_id: str
    title: str
    description: str
    framework: ComplianceFramework
    status: ComplianceStatus
    evidence: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    score: float = 0.0  # 0-100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requirement_id": self.requirement_id,
            "title": self.title,
            "description": self.description,
            "framework": self.framework.value,
            "status": self.status.value,
            "evidence": self.evidence,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "score": self.score,
        }


@dataclass
class ComplianceReport:
    """Complete compliance assessment report."""

    framework: ComplianceFramework
    generated_at: datetime
    dataset_name: str
    risk_level: RiskLevel
    overall_status: ComplianceStatus
    overall_score: float
    requirements: list[ComplianceRequirement]
    executive_summary: str
    audit_trail: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        status_emoji = {
            ComplianceStatus.COMPLIANT: "✅",
            ComplianceStatus.PARTIALLY_COMPLIANT: "⚠️",
            ComplianceStatus.NON_COMPLIANT: "❌",
            ComplianceStatus.NOT_APPLICABLE: "➖",
            ComplianceStatus.REQUIRES_REVIEW: "🔍",
        }

        lines = [
            f"{'=' * 60}",
            f"COMPLIANCE ASSESSMENT REPORT",
            f"{'=' * 60}",
            f"Framework: {self.framework.value.upper().replace('_', ' ')}",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset: {self.dataset_name}",
            f"Risk Level: {self.risk_level.value.upper()}",
            f"",
            f"Overall Status: {status_emoji.get(self.overall_status, '')} {self.overall_status.value.upper()}",
            f"Compliance Score: {self.overall_score:.1f}/100",
            f"",
            f"{'=' * 60}",
            f"EXECUTIVE SUMMARY",
            f"{'=' * 60}",
            self.executive_summary,
            f"",
            f"{'=' * 60}",
            f"REQUIREMENTS ASSESSMENT ({len(self.requirements)} items)",
            f"{'=' * 60}",
        ]

        # Group by status
        by_status: dict[ComplianceStatus, list[ComplianceRequirement]] = {}
        for req in self.requirements:
            by_status.setdefault(req.status, []).append(req)

        for status in [
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.PARTIALLY_COMPLIANT,
            ComplianceStatus.REQUIRES_REVIEW,
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.NOT_APPLICABLE,
        ]:
            reqs = by_status.get(status, [])
            if reqs:
                lines.append(f"\n{status_emoji.get(status, '')} {status.value.upper()} ({len(reqs)})")
                for req in reqs[:5]:
                    lines.append(f"  • [{req.requirement_id}] {req.title}")
                if len(reqs) > 5:
                    lines.append(f"  ... and {len(reqs) - 5} more")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "framework": self.framework.value,
            "generated_at": self.generated_at.isoformat(),
            "dataset_name": self.dataset_name,
            "risk_level": self.risk_level.value,
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
            "requirements": [r.to_dict() for r in self.requirements],
            "executive_summary": self.executive_summary,
            "audit_trail": self.audit_trail,
        }

    def to_markdown(self) -> str:
        """Export report as Markdown."""
        lines = [
            f"# Compliance Assessment Report",
            f"",
            f"**Framework:** {self.framework.value.upper().replace('_', ' ')}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Dataset:** {self.dataset_name}",
            f"**Risk Level:** {self.risk_level.value}",
            f"**Overall Status:** {self.overall_status.value}",
            f"**Compliance Score:** {self.overall_score:.1f}/100",
            f"",
            f"## Executive Summary",
            f"",
            self.executive_summary,
            f"",
            f"## Requirements Assessment",
            f"",
            f"| ID | Requirement | Status | Score |",
            f"|---|---|---|---|",
        ]

        for req in self.requirements:
            lines.append(
                f"| {req.requirement_id} | {req.title} | {req.status.value} | {req.score:.0f} |"
            )

        lines.extend([
            f"",
            f"## Detailed Findings",
            f"",
        ])

        for req in self.requirements:
            if req.findings or req.recommendations:
                lines.append(f"### {req.requirement_id}: {req.title}")
                lines.append(f"**Status:** {req.status.value}")
                if req.findings:
                    lines.append(f"\n**Findings:**")
                    for finding in req.findings:
                        lines.append(f"- {finding}")
                if req.recommendations:
                    lines.append(f"\n**Recommendations:**")
                    for rec in req.recommendations:
                        lines.append(f"- {rec}")
                lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Export report as HTML."""
        md = self.to_markdown()
        # Simple markdown to HTML conversion
        html_content = md.replace("# ", "<h1>").replace("\n## ", "</h1>\n<h2>")
        html_content = html_content.replace("\n### ", "</h2>\n<h3>")
        html_content = html_content.replace("**", "<strong>").replace("**", "</strong>")
        html_content = html_content.replace("\n- ", "\n<li>")
        html_content = html_content.replace("\n\n", "</p>\n<p>")

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Compliance Report - {self.dataset_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .compliant {{ color: green; }}
        .non_compliant {{ color: red; }}
        .partially_compliant {{ color: orange; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""


class ComplianceReportGenerator:
    """Generator for compliance assessment reports.

    Evaluates data quality against regulatory requirements and
    generates comprehensive compliance documentation.
    """

    # EU AI Act requirements
    EU_AI_ACT_REQUIREMENTS = [
        {
            "id": "EU-DQ-001",
            "title": "Data Quality Standards",
            "description": "Training data shall be subject to appropriate data governance and management practices",
            "article": "Article 10.2",
        },
        {
            "id": "EU-DQ-002",
            "title": "Relevant and Representative Data",
            "description": "Training data shall be relevant, representative, free of errors and complete",
            "article": "Article 10.3",
        },
        {
            "id": "EU-DQ-003",
            "title": "Statistical Properties",
            "description": "Training data shall have the appropriate statistical properties",
            "article": "Article 10.3",
        },
        {
            "id": "EU-DQ-004",
            "title": "Bias Examination",
            "description": "Training data shall be examined for possible biases",
            "article": "Article 10.2(f)",
        },
        {
            "id": "EU-DQ-005",
            "title": "Data Documentation",
            "description": "Appropriate documentation of training data shall be maintained",
            "article": "Article 10.2",
        },
        {
            "id": "EU-DQ-006",
            "title": "Gap Identification",
            "description": "Any gaps or shortcomings in data shall be identified and addressed",
            "article": "Article 10.2(g)",
        },
        {
            "id": "EU-DQ-007",
            "title": "Data Origin",
            "description": "Data origin and collection methods shall be documented",
            "article": "Article 11.1",
        },
        {
            "id": "EU-DQ-008",
            "title": "Labeling Quality",
            "description": "Labels shall be accurate and validated through quality control",
            "article": "Article 10.3",
        },
    ]

    # NIST AI RMF requirements
    NIST_AI_RMF_REQUIREMENTS = [
        {
            "id": "NIST-MAP-1.1",
            "title": "Data Characteristics",
            "description": "Document data characteristics including quality, provenance, and limitations",
            "category": "MAP",
        },
        {
            "id": "NIST-MAP-1.5",
            "title": "Bias Assessment",
            "description": "Assess and document potential biases in training data",
            "category": "MAP",
        },
        {
            "id": "NIST-MEASURE-2.3",
            "title": "Data Quality Metrics",
            "description": "Establish and track data quality metrics",
            "category": "MEASURE",
        },
        {
            "id": "NIST-MEASURE-2.6",
            "title": "Fairness Assessment",
            "description": "Measure fairness across demographic groups",
            "category": "MEASURE",
        },
        {
            "id": "NIST-MANAGE-2.1",
            "title": "Risk Documentation",
            "description": "Document identified risks and mitigation strategies",
            "category": "MANAGE",
        },
        {
            "id": "NIST-GOVERN-1.5",
            "title": "Data Governance",
            "description": "Establish data governance policies and procedures",
            "category": "GOVERN",
        },
    ]

    def __init__(
        self,
        framework: ComplianceFramework | str = ComplianceFramework.EU_AI_ACT,
        risk_level: RiskLevel | str = RiskLevel.HIGH,
        custom_requirements: list[dict] | None = None,
    ):
        """Initialize the report generator.

        Args:
            framework: Compliance framework to use
            risk_level: Risk level of the AI system
            custom_requirements: Custom requirements for CUSTOM framework
        """
        if isinstance(framework, str):
            framework = ComplianceFramework(framework)
        if isinstance(risk_level, str):
            risk_level = RiskLevel(risk_level)

        self.framework = framework
        self.risk_level = risk_level
        self.custom_requirements = custom_requirements or []

    def generate(
        self,
        quality_report: Any,
        dataset_name: str = "Untitled Dataset",
        additional_context: dict[str, Any] | None = None,
    ) -> ComplianceReport:
        """Generate a compliance assessment report.

        Args:
            quality_report: QualityReport from DatasetCleaner.analyze()
            dataset_name: Name of the dataset being assessed
            additional_context: Additional context for assessment

        Returns:
            ComplianceReport with assessment results
        """
        context = additional_context or {}
        requirements = []

        # Get requirements based on framework
        if self.framework == ComplianceFramework.EU_AI_ACT:
            requirements = self._assess_eu_ai_act(quality_report, context)
        elif self.framework == ComplianceFramework.NIST_AI_RMF:
            requirements = self._assess_nist_ai_rmf(quality_report, context)
        elif self.framework == ComplianceFramework.CUSTOM:
            requirements = self._assess_custom(quality_report, context)

        # Calculate overall status and score
        scores = [r.score for r in requirements if r.status != ComplianceStatus.NOT_APPLICABLE]
        overall_score = sum(scores) / len(scores) if scores else 0

        non_compliant_count = sum(
            1 for r in requirements if r.status == ComplianceStatus.NON_COMPLIANT
        )
        partial_count = sum(
            1 for r in requirements if r.status == ComplianceStatus.PARTIALLY_COMPLIANT
        )

        if non_compliant_count > 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif partial_count > 0:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.COMPLIANT

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            quality_report, requirements, overall_status, overall_score
        )

        # Create audit trail
        audit_trail = [
            {
                "timestamp": datetime.now().isoformat(),
                "action": "compliance_assessment_generated",
                "framework": self.framework.value,
                "dataset": dataset_name,
            }
        ]

        return ComplianceReport(
            framework=self.framework,
            generated_at=datetime.now(),
            dataset_name=dataset_name,
            risk_level=self.risk_level,
            overall_status=overall_status,
            overall_score=overall_score,
            requirements=requirements,
            executive_summary=executive_summary,
            audit_trail=audit_trail,
        )

    def _assess_eu_ai_act(
        self,
        quality_report: Any,
        context: dict[str, Any],
    ) -> list[ComplianceRequirement]:
        """Assess compliance with EU AI Act requirements."""
        requirements = []

        # Extract metrics from quality report
        quality_score = getattr(quality_report, 'quality_score', None)
        overall_score = quality_score.overall if quality_score else 50
        label_quality = quality_score.label_quality if quality_score else 50
        bias_quality = quality_score.bias_quality if quality_score else 50
        duplicate_quality = quality_score.duplicate_quality if quality_score else 50

        for req_def in self.EU_AI_ACT_REQUIREMENTS:
            req = self._create_requirement(
                req_def["id"],
                req_def["title"],
                req_def["description"],
                ComplianceFramework.EU_AI_ACT,
            )

            # Assess each requirement
            if req_def["id"] == "EU-DQ-001":
                # Data governance
                req.score = min(100, overall_score + 20)  # Assume basic governance if analyzed
                req.status = self._score_to_status(req.score)
                req.evidence.append("Data quality analysis performed")
                req.findings.append(f"Overall quality score: {overall_score:.1f}/100")

            elif req_def["id"] == "EU-DQ-002":
                # Relevant and representative
                req.score = overall_score
                req.status = self._score_to_status(req.score)
                if overall_score < 70:
                    req.findings.append("Data quality below recommended threshold")
                    req.recommendations.append("Review and improve data collection process")

            elif req_def["id"] == "EU-DQ-003":
                # Statistical properties
                req.score = duplicate_quality
                req.status = self._score_to_status(req.score)
                req.evidence.append("Duplicate analysis performed")

            elif req_def["id"] == "EU-DQ-004":
                # Bias examination
                req.score = bias_quality
                req.status = self._score_to_status(req.score)
                req.evidence.append("Bias detection analysis performed")
                if bias_quality < 80:
                    req.findings.append("Potential bias detected in dataset")
                    req.recommendations.append("Review bias metrics and implement mitigation")

            elif req_def["id"] == "EU-DQ-005":
                # Documentation
                doc_provided = context.get("documentation_provided", False)
                req.score = 100 if doc_provided else 50
                req.status = self._score_to_status(req.score)
                if not doc_provided:
                    req.recommendations.append("Provide comprehensive data documentation")

            elif req_def["id"] == "EU-DQ-006":
                # Gap identification
                req.score = overall_score
                req.status = self._score_to_status(req.score)
                req.evidence.append("Gap analysis through quality detection")

            elif req_def["id"] == "EU-DQ-007":
                # Data origin
                origin_documented = context.get("origin_documented", False)
                req.score = 100 if origin_documented else 40
                req.status = self._score_to_status(req.score)
                if not origin_documented:
                    req.recommendations.append("Document data sources and collection methods")

            elif req_def["id"] == "EU-DQ-008":
                # Labeling quality
                req.score = label_quality
                req.status = self._score_to_status(req.score)
                req.evidence.append("Label error detection performed")
                if label_quality < 80:
                    req.findings.append("Label quality issues detected")
                    req.recommendations.append("Review and correct identified label errors")

            requirements.append(req)

        return requirements

    def _assess_nist_ai_rmf(
        self,
        quality_report: Any,
        context: dict[str, Any],
    ) -> list[ComplianceRequirement]:
        """Assess compliance with NIST AI RMF requirements."""
        requirements = []

        quality_score = getattr(quality_report, 'quality_score', None)
        overall_score = quality_score.overall if quality_score else 50
        bias_quality = quality_score.bias_quality if quality_score else 50

        for req_def in self.NIST_AI_RMF_REQUIREMENTS:
            req = self._create_requirement(
                req_def["id"],
                req_def["title"],
                req_def["description"],
                ComplianceFramework.NIST_AI_RMF,
            )

            if req_def["id"] == "NIST-MAP-1.1":
                req.score = min(100, overall_score + 10)
                req.status = self._score_to_status(req.score)
                req.evidence.append("Data characteristics documented through analysis")

            elif req_def["id"] == "NIST-MAP-1.5":
                req.score = bias_quality
                req.status = self._score_to_status(req.score)
                req.evidence.append("Bias assessment performed")

            elif req_def["id"] == "NIST-MEASURE-2.3":
                req.score = overall_score
                req.status = self._score_to_status(req.score)
                req.evidence.append("Quality metrics computed")

            elif req_def["id"] == "NIST-MEASURE-2.6":
                req.score = bias_quality
                req.status = self._score_to_status(req.score)
                req.evidence.append("Fairness metrics assessed")

            elif req_def["id"] == "NIST-MANAGE-2.1":
                doc_provided = context.get("risk_documented", False)
                req.score = 100 if doc_provided else 50
                req.status = self._score_to_status(req.score)

            elif req_def["id"] == "NIST-GOVERN-1.5":
                governance = context.get("governance_established", False)
                req.score = 100 if governance else 40
                req.status = self._score_to_status(req.score)

            requirements.append(req)

        return requirements

    def _assess_custom(
        self,
        quality_report: Any,
        context: dict[str, Any],
    ) -> list[ComplianceRequirement]:
        """Assess compliance with custom requirements."""
        requirements = []

        for req_def in self.custom_requirements:
            req = self._create_requirement(
                req_def.get("id", "CUSTOM-001"),
                req_def.get("title", "Custom Requirement"),
                req_def.get("description", ""),
                ComplianceFramework.CUSTOM,
            )

            # Use provided score or default
            req.score = req_def.get("score", 50)
            req.status = self._score_to_status(req.score)
            requirements.append(req)

        return requirements

    def _create_requirement(
        self,
        req_id: str,
        title: str,
        description: str,
        framework: ComplianceFramework,
    ) -> ComplianceRequirement:
        """Create a new compliance requirement."""
        return ComplianceRequirement(
            requirement_id=req_id,
            title=title,
            description=description,
            framework=framework,
            status=ComplianceStatus.REQUIRES_REVIEW,
            score=0.0,
        )

    def _score_to_status(self, score: float) -> ComplianceStatus:
        """Convert score to compliance status."""
        if score >= 80:
            return ComplianceStatus.COMPLIANT
        elif score >= 50:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _generate_executive_summary(
        self,
        quality_report: Any,
        requirements: list[ComplianceRequirement],
        overall_status: ComplianceStatus,
        overall_score: float,
    ) -> str:
        """Generate executive summary for the report."""
        framework_name = self.framework.value.upper().replace("_", " ")

        compliant = sum(1 for r in requirements if r.status == ComplianceStatus.COMPLIANT)
        partial = sum(1 for r in requirements if r.status == ComplianceStatus.PARTIALLY_COMPLIANT)
        non_compliant = sum(1 for r in requirements if r.status == ComplianceStatus.NON_COMPLIANT)

        summary_parts = [
            f"This report assesses compliance with {framework_name} requirements.",
            f"",
            f"Of {len(requirements)} assessed requirements:",
            f"- {compliant} are fully compliant",
            f"- {partial} are partially compliant",
            f"- {non_compliant} are non-compliant",
            f"",
            f"Overall compliance score: {overall_score:.1f}/100",
        ]

        if overall_status == ComplianceStatus.COMPLIANT:
            summary_parts.append(
                "\nThe dataset meets compliance requirements. Maintain current practices."
            )
        elif overall_status == ComplianceStatus.PARTIALLY_COMPLIANT:
            summary_parts.append(
                "\nSome requirements need attention. Review findings and implement recommendations."
            )
        else:
            summary_parts.append(
                "\nSignificant compliance gaps identified. Immediate action required."
            )

        return "\n".join(summary_parts)


def generate_compliance_report(
    quality_report: Any,
    framework: str = "eu_ai_act",
    dataset_name: str = "Dataset",
    **kwargs: Any,
) -> ComplianceReport:
    """Generate a compliance report from a quality analysis.

    Args:
        quality_report: QualityReport from DatasetCleaner
        framework: Compliance framework ('eu_ai_act', 'nist_ai_rmf')
        dataset_name: Name of the dataset
        **kwargs: Additional arguments for generator

    Returns:
        ComplianceReport with assessment results
    """
    generator = ComplianceReportGenerator(framework=framework, **kwargs)
    return generator.generate(quality_report, dataset_name)
