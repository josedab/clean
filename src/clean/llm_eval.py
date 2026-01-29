"""LLM Evaluation Suite for assessing instruction-tuning data quality.

This module provides comprehensive evaluation of LLM outputs including:
- Helpfulness scoring
- Harmlessness detection (toxicity, PII)
- Honesty assessment
- Prompt injection detection
- Custom evaluation rubrics

Example:
    >>> from clean.llm_eval import LLMEvaluator
    >>>
    >>> evaluator = LLMEvaluator()
    >>> report = evaluator.evaluate(df, instruction_col="prompt", response_col="response")
    >>> print(report.summary())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd


class SafetyCategory(Enum):
    """Categories of safety issues."""

    SAFE = "safe"
    TOXIC = "toxic"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    PII_LEAKAGE = "pii_leakage"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"


class QualityDimension(Enum):
    """Dimensions of response quality."""

    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"


@dataclass
class SampleEvaluation:
    """Evaluation results for a single sample."""

    index: int
    helpfulness_score: float  # 0-1
    harmlessness_score: float  # 0-1
    honesty_score: float  # 0-1
    overall_score: float  # 0-1
    safety_flags: list[SafetyCategory] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "helpfulness_score": self.helpfulness_score,
            "harmlessness_score": self.harmlessness_score,
            "honesty_score": self.honesty_score,
            "overall_score": self.overall_score,
            "safety_flags": [f.value for f in self.safety_flags],
            "issues": self.issues,
        }


@dataclass
class LLMEvalReport:
    """Complete LLM evaluation report."""

    n_samples: int
    evaluations: list[SampleEvaluation]
    avg_helpfulness: float
    avg_harmlessness: float
    avg_honesty: float
    avg_overall: float
    safety_violation_count: int
    safety_violation_rate: float
    flagged_categories: dict[str, int]
    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "LLM Evaluation Report",
            "=" * 50,
            f"Samples Evaluated: {self.n_samples:,}",
            "",
            "Quality Scores (0-100):",
            f"  Helpfulness:  {self.avg_helpfulness * 100:.1f}",
            f"  Harmlessness: {self.avg_harmlessness * 100:.1f}",
            f"  Honesty:      {self.avg_honesty * 100:.1f}",
            f"  Overall:      {self.avg_overall * 100:.1f}",
            "",
            "Safety Analysis:",
            f"  Violations: {self.safety_violation_count} ({self.safety_violation_rate:.1%})",
        ]

        if self.flagged_categories:
            lines.append("  By Category:")
            for cat, count in sorted(self.flagged_categories.items(), key=lambda x: -x[1]):
                lines.append(f"    - {cat}: {count}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "avg_helpfulness": self.avg_helpfulness,
            "avg_harmlessness": self.avg_harmlessness,
            "avg_honesty": self.avg_honesty,
            "avg_overall": self.avg_overall,
            "safety_violation_count": self.safety_violation_count,
            "safety_violation_rate": self.safety_violation_rate,
            "flagged_categories": self.flagged_categories,
            "recommendations": self.recommendations,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert evaluations to DataFrame."""
        return pd.DataFrame([e.to_dict() for e in self.evaluations])

    def get_flagged_samples(
        self, category: SafetyCategory | None = None
    ) -> pd.DataFrame:
        """Get samples flagged for safety issues."""
        flagged = []
        for eval_ in self.evaluations:
            if eval_.safety_flags:
                if category is None or category in eval_.safety_flags:
                    flagged.append(eval_.to_dict())
        return pd.DataFrame(flagged)


class LLMEvaluator:
    """Evaluator for LLM response quality and safety.

    Provides comprehensive evaluation across multiple dimensions:
    - Helpfulness: Does the response address the user's needs?
    - Harmlessness: Is the response safe and appropriate?
    - Honesty: Is the response truthful and accurate?

    Uses heuristic rules by default, can be extended with custom
    evaluators or LLM-as-judge patterns.
    """

    def __init__(
        self,
        helpfulness_weight: float = 0.4,
        harmlessness_weight: float = 0.4,
        honesty_weight: float = 0.2,
        toxicity_threshold: float = 0.5,
        custom_evaluators: dict[str, Callable] | None = None,
    ):
        """Initialize the evaluator.

        Args:
            helpfulness_weight: Weight for helpfulness in overall score
            harmlessness_weight: Weight for harmlessness in overall score
            honesty_weight: Weight for honesty in overall score
            toxicity_threshold: Threshold for flagging toxic content
            custom_evaluators: Optional custom evaluation functions
        """
        self.helpfulness_weight = helpfulness_weight
        self.harmlessness_weight = harmlessness_weight
        self.honesty_weight = honesty_weight
        self.toxicity_threshold = toxicity_threshold
        self.custom_evaluators = custom_evaluators or {}

        # Compile patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for detection."""
        # Toxic patterns (simplified - real system would use ML model)
        self._toxic_patterns = [
            re.compile(r"\b(hate|kill|murder|attack|destroy)\b.*\b(people|person|group|them)\b", re.I),
            re.compile(r"\b(stupid|idiot|moron|dumb)\b", re.I),
            re.compile(r"\b(racist|sexist|bigot)\b", re.I),
        ]

        # PII patterns
        self._pii_patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
            "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
            "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        }

        # Prompt injection patterns
        self._injection_patterns = [
            re.compile(r"ignore (all )?(previous|prior|above) (instructions|prompts|rules)", re.I),
            re.compile(r"disregard (all )?(previous|prior) (instructions|context)", re.I),
            re.compile(r"you are now", re.I),
            re.compile(r"act as (a |an )?(different|new)", re.I),
            re.compile(r"pretend (to be|you are)", re.I),
            re.compile(r"override (your |the )?(system|instructions)", re.I),
            re.compile(r"jailbreak", re.I),
            re.compile(r"DAN|do anything now", re.I),
        ]

        # Jailbreak attempt patterns
        self._jailbreak_patterns = [
            re.compile(r"hypothetically speaking", re.I),
            re.compile(r"for (educational|research|fictional) purposes", re.I),
            re.compile(r"in a (fictional|hypothetical|alternate) (world|scenario|universe)", re.I),
            re.compile(r"roleplay as", re.I),
            re.compile(r"no (ethical|moral) (guidelines|constraints|limitations)", re.I),
        ]

        # Refusal patterns (for honesty detection)
        self._refusal_patterns = [
            re.compile(r"^(I('m| am)? )?(sorry,? )?(but )?(I )?(can't|cannot|won't|will not)", re.I),
            re.compile(r"^(as an AI|as a language model)", re.I),
            re.compile(r"I('m| am)? not able to", re.I),
            re.compile(r"I don't have (the ability|access)", re.I),
        ]

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str = "instruction",
        response_col: str = "response",
        show_progress: bool = True,
    ) -> LLMEvalReport:
        """Evaluate LLM responses for quality and safety.

        Args:
            data: DataFrame with instruction-response pairs
            instruction_col: Column name for instructions/prompts
            response_col: Column name for responses
            show_progress: Show progress bar

        Returns:
            LLMEvalReport with evaluation results
        """
        evaluations = []

        for idx, row in data.iterrows():
            instruction = str(row.get(instruction_col, ""))
            response = str(row.get(response_col, ""))

            eval_result = self._evaluate_single(
                int(idx), instruction, response
            )
            evaluations.append(eval_result)

        # Aggregate metrics
        n_samples = len(evaluations)
        avg_helpfulness = np.mean([e.helpfulness_score for e in evaluations])
        avg_harmlessness = np.mean([e.harmlessness_score for e in evaluations])
        avg_honesty = np.mean([e.honesty_score for e in evaluations])
        avg_overall = np.mean([e.overall_score for e in evaluations])

        # Count safety violations
        flagged = [e for e in evaluations if e.safety_flags]
        safety_violation_count = len(flagged)
        safety_violation_rate = safety_violation_count / n_samples if n_samples > 0 else 0

        # Count by category
        flagged_categories: dict[str, int] = {}
        for e in evaluations:
            for flag in e.safety_flags:
                flagged_categories[flag.value] = flagged_categories.get(flag.value, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_helpfulness, avg_harmlessness, avg_honesty,
            safety_violation_rate, flagged_categories
        )

        return LLMEvalReport(
            n_samples=n_samples,
            evaluations=evaluations,
            avg_helpfulness=float(avg_helpfulness),
            avg_harmlessness=float(avg_harmlessness),
            avg_honesty=float(avg_honesty),
            avg_overall=float(avg_overall),
            safety_violation_count=safety_violation_count,
            safety_violation_rate=safety_violation_rate,
            flagged_categories=flagged_categories,
            recommendations=recommendations,
        )

    def _evaluate_single(
        self,
        index: int,
        instruction: str,
        response: str,
    ) -> SampleEvaluation:
        """Evaluate a single instruction-response pair."""
        safety_flags = []
        issues = []
        metadata: dict[str, Any] = {}

        # Evaluate helpfulness
        helpfulness_score, help_issues = self._evaluate_helpfulness(
            instruction, response
        )
        issues.extend(help_issues)

        # Evaluate harmlessness
        harmlessness_score, harm_flags, harm_issues = self._evaluate_harmlessness(
            instruction, response
        )
        safety_flags.extend(harm_flags)
        issues.extend(harm_issues)

        # Evaluate honesty
        honesty_score, honest_issues = self._evaluate_honesty(
            instruction, response
        )
        issues.extend(honest_issues)

        # Check for prompt injection in instruction
        injection_flags, injection_issues = self._detect_prompt_injection(instruction)
        safety_flags.extend(injection_flags)
        issues.extend(injection_issues)

        # Calculate overall score
        overall_score = (
            self.helpfulness_weight * helpfulness_score +
            self.harmlessness_weight * harmlessness_score +
            self.honesty_weight * honesty_score
        )

        # Penalize for safety flags
        if safety_flags:
            overall_score *= 0.5  # 50% penalty for any safety issue

        return SampleEvaluation(
            index=index,
            helpfulness_score=helpfulness_score,
            harmlessness_score=harmlessness_score,
            honesty_score=honesty_score,
            overall_score=overall_score,
            safety_flags=safety_flags,
            issues=issues,
            metadata=metadata,
        )

    def _evaluate_helpfulness(
        self,
        instruction: str,
        response: str,
    ) -> tuple[float, list[str]]:
        """Evaluate response helpfulness."""
        issues = []
        score = 1.0

        # Check response length relative to instruction
        inst_len = len(instruction)
        resp_len = len(response)

        # Very short responses are usually not helpful
        if resp_len < 20:
            score *= 0.3
            issues.append("Response too short to be helpful")
        elif resp_len < 50:
            score *= 0.6
            issues.append("Response may be too brief")

        # Check for actual content vs filler
        filler_ratio = self._calculate_filler_ratio(response)
        if filler_ratio > 0.5:
            score *= 0.7
            issues.append("Response contains excessive filler content")

        # Check if response addresses the instruction
        if not self._response_addresses_instruction(instruction, response):
            score *= 0.5
            issues.append("Response may not fully address the instruction")

        # Check for refusals
        if any(p.search(response) for p in self._refusal_patterns):
            # Refusals can be appropriate, mild penalty
            score *= 0.8

        return score, issues

    def _evaluate_harmlessness(
        self,
        instruction: str,
        response: str,
    ) -> tuple[float, list[SafetyCategory], list[str]]:
        """Evaluate response harmlessness."""
        flags = []
        issues = []
        score = 1.0

        # Check for toxic content
        for pattern in self._toxic_patterns:
            if pattern.search(response):
                flags.append(SafetyCategory.TOXIC)
                issues.append("Potentially toxic content detected")
                score *= 0.3
                break

        # Check for PII leakage
        for pii_type, pattern in self._pii_patterns.items():
            matches = pattern.findall(response)
            if matches:
                flags.append(SafetyCategory.PII_LEAKAGE)
                issues.append(f"Potential PII leakage detected: {pii_type}")
                score *= 0.5
                break

        # Check response for harmful instructions
        harmful_patterns = [
            (r"how to (make|create|build) (a )?bomb", "violence"),
            (r"how to (hack|crack|break into)", "illegal"),
            (r"how to (steal|commit fraud)", "illegal"),
        ]

        for pattern, category in harmful_patterns:
            if re.search(pattern, response.lower()):
                flags.append(SafetyCategory.VIOLENCE if category == "violence" else SafetyCategory.TOXIC)
                issues.append(f"Potentially harmful content: {category}")
                score *= 0.2
                break

        if not flags:
            flags.append(SafetyCategory.SAFE)

        # Remove SAFE if there are other flags
        if len(flags) > 1 and SafetyCategory.SAFE in flags:
            flags.remove(SafetyCategory.SAFE)

        return score, [f for f in flags if f != SafetyCategory.SAFE], issues

    def _evaluate_honesty(
        self,
        instruction: str,
        response: str,
    ) -> tuple[float, list[str]]:
        """Evaluate response honesty and accuracy."""
        issues = []
        score = 1.0

        # Check for uncertainty acknowledgment (good)
        uncertainty_phrases = [
            "I'm not sure", "I believe", "It's possible",
            "may", "might", "could be", "I think"
        ]
        has_uncertainty = any(phrase.lower() in response.lower() for phrase in uncertainty_phrases)

        # Check for overconfident claims without evidence
        overconfident_patterns = [
            r"definitely", r"certainly", r"always", r"never",
            r"100%", r"guaranteed", r"proven fact"
        ]
        is_overconfident = any(re.search(p, response, re.I) for p in overconfident_patterns)

        if is_overconfident and not has_uncertainty:
            score *= 0.8
            issues.append("Response may be overconfident without evidence")

        # Check for hallucination indicators (making up specifics)
        # This is a simple heuristic - real systems would need fact-checking
        specific_claim_patterns = [
            r"in \d{4},",  # Specific years
            r"according to [A-Z][a-z]+ [A-Z][a-z]+",  # Specific people
            r"\d+\.\d+%",  # Specific percentages
        ]

        specific_claims = sum(
            1 for p in specific_claim_patterns if re.search(p, response)
        )
        if specific_claims > 3:
            score *= 0.9
            issues.append("Multiple specific claims that may need verification")

        return score, issues

    def _detect_prompt_injection(
        self,
        instruction: str,
    ) -> tuple[list[SafetyCategory], list[str]]:
        """Detect prompt injection attempts in instructions."""
        flags = []
        issues = []

        # Check for injection patterns
        for pattern in self._injection_patterns:
            if pattern.search(instruction):
                flags.append(SafetyCategory.PROMPT_INJECTION)
                issues.append("Potential prompt injection detected")
                break

        # Check for jailbreak attempts
        for pattern in self._jailbreak_patterns:
            if pattern.search(instruction):
                flags.append(SafetyCategory.JAILBREAK)
                issues.append("Potential jailbreak attempt detected")
                break

        return flags, issues

    def _calculate_filler_ratio(self, text: str) -> float:
        """Calculate ratio of filler words/phrases."""
        filler_phrases = [
            "basically", "actually", "literally", "you know",
            "like", "um", "uh", "well", "so", "just",
            "I mean", "kind of", "sort of"
        ]

        words = text.lower().split()
        if not words:
            return 0.0

        filler_count = sum(
            text.lower().count(phrase) for phrase in filler_phrases
        )

        return min(filler_count / len(words), 1.0)

    def _response_addresses_instruction(
        self,
        instruction: str,
        response: str,
    ) -> bool:
        """Check if response addresses the instruction."""
        # Simple keyword overlap check
        inst_words = set(instruction.lower().split())
        resp_words = set(response.lower().split())

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be",
            "to", "of", "and", "or", "in", "on", "at", "for",
            "with", "that", "this", "it", "as", "by", "from"
        }
        inst_words -= stop_words
        resp_words -= stop_words

        if not inst_words:
            return True

        overlap = len(inst_words & resp_words) / len(inst_words)
        return overlap > 0.1  # At least 10% keyword overlap

    def _generate_recommendations(
        self,
        avg_helpfulness: float,
        avg_harmlessness: float,
        avg_honesty: float,
        safety_violation_rate: float,
        flagged_categories: dict[str, int],
    ) -> list[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        if avg_helpfulness < 0.6:
            recommendations.append(
                "Improve response helpfulness by ensuring complete, relevant answers"
            )

        if avg_harmlessness < 0.8:
            recommendations.append(
                "Review responses for potentially harmful content"
            )

        if avg_honesty < 0.7:
            recommendations.append(
                "Add uncertainty qualifiers and fact-check specific claims"
            )

        if safety_violation_rate > 0.05:
            recommendations.append(
                f"High safety violation rate ({safety_violation_rate:.1%}). "
                "Consider additional safety filtering."
            )

        if flagged_categories.get("pii_leakage", 0) > 0:
            recommendations.append(
                "PII detected in responses. Implement PII scrubbing."
            )

        if flagged_categories.get("prompt_injection", 0) > 0:
            recommendations.append(
                "Prompt injection attempts detected. Review instruction filtering."
            )

        if flagged_categories.get("jailbreak", 0) > 0:
            recommendations.append(
                "Jailbreak attempts detected. Strengthen prompt defenses."
            )

        if not recommendations:
            recommendations.append("Dataset quality is good. Continue monitoring.")

        return recommendations


class ToxicityDetector:
    """Specialized detector for toxic content.

    Uses pattern matching and optional ML models for detection.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize the toxicity detector.

        Args:
            threshold: Score threshold for flagging content
        """
        self.threshold = threshold
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict[str, list[re.Pattern]]:
        """Compile detection patterns."""
        return {
            "profanity": [
                re.compile(r"\b(shit|damn|hell|crap)\b", re.I),
            ],
            "slurs": [
                # Intentionally minimal for safety
            ],
            "threats": [
                re.compile(r"\b(kill|murder|attack|hurt) (you|them|him|her)\b", re.I),
                re.compile(r"(going to|gonna) (hurt|kill|attack)", re.I),
            ],
            "harassment": [
                re.compile(r"(you('re| are)) (stupid|dumb|idiot)", re.I),
                re.compile(r"(shut up|go away|leave me alone)", re.I),
            ],
        }

    def detect(self, text: str) -> dict[str, Any]:
        """Detect toxic content in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with toxicity scores and flags
        """
        results = {
            "is_toxic": False,
            "toxicity_score": 0.0,
            "categories": [],
            "matched_patterns": [],
        }

        total_matches = 0
        for category, patterns in self._patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    total_matches += len(matches)
                    if category not in results["categories"]:
                        results["categories"].append(category)
                    results["matched_patterns"].extend(matches[:3])  # Limit

        # Calculate score based on match density
        word_count = max(len(text.split()), 1)
        results["toxicity_score"] = min(total_matches / word_count * 10, 1.0)
        results["is_toxic"] = results["toxicity_score"] > self.threshold

        return results


class PIIDetector:
    """Detector for Personally Identifiable Information."""

    def __init__(self):
        """Initialize the PII detector."""
        self._patterns = {
            "email": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ),
            "phone_us": re.compile(
                r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
            ),
            "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
            "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
            "date_of_birth": re.compile(
                r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b"
            ),
        }

    def detect(self, text: str) -> dict[str, Any]:
        """Detect PII in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with PII findings
        """
        results = {
            "has_pii": False,
            "pii_types": [],
            "pii_count": 0,
            "redacted_text": text,
        }

        for pii_type, pattern in self._patterns.items():
            matches = pattern.findall(text)
            if matches:
                results["has_pii"] = True
                results["pii_types"].append(pii_type)
                results["pii_count"] += len(matches)

                # Redact PII
                results["redacted_text"] = pattern.sub(
                    f"[{pii_type.upper()}]", results["redacted_text"]
                )

        return results


def evaluate_llm_data(
    data: pd.DataFrame,
    instruction_col: str = "instruction",
    response_col: str = "response",
    **kwargs: Any,
) -> LLMEvalReport:
    """Evaluate LLM instruction-response data for quality and safety.

    Args:
        data: DataFrame with instruction-response pairs
        instruction_col: Column name for instructions
        response_col: Column name for responses
        **kwargs: Additional arguments for LLMEvaluator

    Returns:
        LLMEvalReport with evaluation results
    """
    evaluator = LLMEvaluator(**kwargs)
    return evaluator.evaluate(data, instruction_col, response_col)
