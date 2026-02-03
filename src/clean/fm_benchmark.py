"""Foundation Model Quality Benchmark Suite.

Standardized benchmarks for evaluating LLM training data quality with metrics
for instruction quality, diversity, safety, and more.

Example:
    >>> from clean.fm_benchmark import FMBenchmark, BenchmarkLeaderboard
    >>>
    >>> # Benchmark your dataset
    >>> benchmark = FMBenchmark()
    >>> result = benchmark.evaluate(df, instruction_col="prompt", response_col="completion")
    >>> print(result.summary())
    >>>
    >>> # Compare against reference datasets
    >>> comparison = benchmark.compare_to_references(result)
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkMetric(Enum):
    """Standard benchmark metrics for LLM data quality."""

    # Instruction Quality
    INSTRUCTION_CLARITY = "instruction_clarity"
    INSTRUCTION_SPECIFICITY = "instruction_specificity"
    INSTRUCTION_COMPLETENESS = "instruction_completeness"

    # Response Quality
    RESPONSE_RELEVANCE = "response_relevance"
    RESPONSE_COMPLETENESS = "response_completeness"
    RESPONSE_COHERENCE = "response_coherence"

    # Diversity
    LEXICAL_DIVERSITY = "lexical_diversity"
    SEMANTIC_DIVERSITY = "semantic_diversity"
    TASK_DIVERSITY = "task_diversity"

    # Safety
    TOXICITY_RATE = "toxicity_rate"
    PII_RATE = "pii_rate"
    BIAS_SCORE = "bias_score"

    # Data Quality
    DUPLICATE_RATE = "duplicate_rate"
    CONSISTENCY_SCORE = "consistency_score"
    NOISE_RATE = "noise_rate"

    # Overall
    OVERALL_QUALITY = "overall_quality"


class DatasetCategory(Enum):
    """Categories of instruction-tuning datasets."""

    GENERAL = "general"
    CODE = "code"
    MATH = "math"
    REASONING = "reasoning"
    CREATIVE = "creative"
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    SAFETY = "safety"
    MULTILINGUAL = "multilingual"


@dataclass
class MetricResult:
    """Result for a single metric evaluation."""

    metric: BenchmarkMetric
    value: float
    percentile: float | None = None  # Percentile vs reference datasets
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark evaluation result."""

    dataset_id: str
    dataset_name: str
    timestamp: datetime
    n_samples: int
    category: DatasetCategory
    metrics: dict[BenchmarkMetric, MetricResult]
    overall_score: float
    grade: str  # A, B, C, D, F
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Foundation Model Benchmark Results",
            "=" * 50,
            f"Dataset: {self.dataset_name}",
            f"ID: {self.dataset_id}",
            f"Samples: {self.n_samples:,}",
            f"Category: {self.category.value}",
            f"Timestamp: {self.timestamp.isoformat()}",
            "",
            f"Overall Score: {self.overall_score:.1f}/100 (Grade: {self.grade})",
            "",
            "Metric Scores:",
        ]

        for metric, result in sorted(
            self.metrics.items(), key=lambda x: x[1].value, reverse=True
        ):
            percentile_str = ""
            if result.percentile is not None:
                percentile_str = f" (P{result.percentile:.0f})"
            lines.append(f"  • {metric.value}: {result.value:.2f}{percentile_str}")

        if self.strengths:
            lines.append("")
            lines.append("Strengths:")
            for s in self.strengths[:5]:
                lines.append(f"  ✓ {s}")

        if self.weaknesses:
            lines.append("")
            lines.append("Weaknesses:")
            for w in self.weaknesses[:5]:
                lines.append(f"  ✗ {w}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for r in self.recommendations[:5]:
                lines.append(f"  → {r}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "n_samples": self.n_samples,
            "category": self.category.value,
            "metrics": {
                m.value: {"value": r.value, "percentile": r.percentile, "details": r.details}
                for m, r in self.metrics.items()
            },
            "overall_score": self.overall_score,
            "grade": self.grade,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
        }


class MetricEvaluator(ABC):
    """Base class for metric evaluators."""

    @property
    @abstractmethod
    def metric(self) -> BenchmarkMetric:
        """The metric this evaluator computes."""
        pass

    @abstractmethod
    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str,
        response_col: str,
    ) -> MetricResult:
        """Evaluate the metric on the dataset."""
        pass


class InstructionClarityEvaluator(MetricEvaluator):
    """Evaluates clarity of instructions."""

    @property
    def metric(self) -> BenchmarkMetric:
        return BenchmarkMetric.INSTRUCTION_CLARITY

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str,
        response_col: str,
    ) -> MetricResult:
        instructions = data[instruction_col].dropna().astype(str)

        scores = []
        for inst in instructions:
            score = self._score_clarity(inst)
            scores.append(score)

        avg_score = np.mean(scores) if scores else 0.0

        return MetricResult(
            metric=self.metric,
            value=float(avg_score),
            details={
                "n_evaluated": len(scores),
                "score_distribution": {
                    "min": float(np.min(scores)) if scores else 0,
                    "max": float(np.max(scores)) if scores else 0,
                    "std": float(np.std(scores)) if scores else 0,
                },
            },
        )

    def _score_clarity(self, instruction: str) -> float:
        """Score clarity of a single instruction (0-1)."""
        score = 1.0

        # Length check - too short or too long reduces clarity
        length = len(instruction)
        if length < 10:
            score *= 0.5
        elif length > 2000:
            score *= 0.8

        # Sentence structure - should have proper punctuation
        if not any(instruction.rstrip().endswith(c) for c in ".?!"):
            score *= 0.9

        # Question clarity - questions should have question marks
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        starts_with_question = any(
            instruction.lower().strip().startswith(w) for w in question_words
        )
        if starts_with_question and "?" not in instruction:
            score *= 0.85

        # Avoid vague language
        vague_terms = ["something", "stuff", "things", "etc", "whatever"]
        vague_count = sum(1 for term in vague_terms if term in instruction.lower())
        score *= max(0.5, 1 - vague_count * 0.1)

        return score


class LexicalDiversityEvaluator(MetricEvaluator):
    """Evaluates lexical diversity of the dataset."""

    @property
    def metric(self) -> BenchmarkMetric:
        return BenchmarkMetric.LEXICAL_DIVERSITY

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str,
        response_col: str,
    ) -> MetricResult:
        all_text = []
        all_text.extend(data[instruction_col].dropna().astype(str).tolist())
        all_text.extend(data[response_col].dropna().astype(str).tolist())

        # Compute type-token ratio
        all_tokens = []
        for text in all_text:
            tokens = text.lower().split()
            all_tokens.extend(tokens)

        if not all_tokens:
            return MetricResult(metric=self.metric, value=0.0)

        unique_tokens = set(all_tokens)
        ttr = len(unique_tokens) / len(all_tokens)

        # Normalize TTR (it decreases with corpus size)
        # Use root TTR for better normalization
        root_ttr = len(unique_tokens) / np.sqrt(len(all_tokens))

        # Scale to 0-1 range (root_ttr typically ranges from 5-50)
        normalized_score = min(1.0, root_ttr / 30)

        return MetricResult(
            metric=self.metric,
            value=float(normalized_score),
            details={
                "unique_tokens": len(unique_tokens),
                "total_tokens": len(all_tokens),
                "type_token_ratio": float(ttr),
                "root_ttr": float(root_ttr),
            },
        )


class ToxicityRateEvaluator(MetricEvaluator):
    """Evaluates toxicity rate in the dataset."""

    # Common toxic/profane terms (simplified)
    TOXIC_PATTERNS = [
        r"\b(hate|kill|die|stupid|idiot|dumb)\b",
        r"\b(racist|sexist|homophobic)\b",
    ]

    @property
    def metric(self) -> BenchmarkMetric:
        return BenchmarkMetric.TOXICITY_RATE

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str,
        response_col: str,
    ) -> MetricResult:
        import re

        toxic_count = 0
        total = 0

        for col in [instruction_col, response_col]:
            texts = data[col].dropna().astype(str)
            for text in texts:
                total += 1
                text_lower = text.lower()
                is_toxic = any(
                    re.search(pattern, text_lower)
                    for pattern in self.TOXIC_PATTERNS
                )
                if is_toxic:
                    toxic_count += 1

        toxicity_rate = toxic_count / max(total, 1)

        # Invert for score (lower toxicity = higher score)
        score = 1.0 - toxicity_rate

        return MetricResult(
            metric=self.metric,
            value=float(score),
            details={
                "toxic_samples": toxic_count,
                "total_samples": total,
                "toxicity_rate": float(toxicity_rate),
            },
        )


class DuplicateRateEvaluator(MetricEvaluator):
    """Evaluates duplicate rate in the dataset."""

    @property
    def metric(self) -> BenchmarkMetric:
        return BenchmarkMetric.DUPLICATE_RATE

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str,
        response_col: str,
    ) -> MetricResult:
        # Check for exact instruction duplicates
        instructions = data[instruction_col].dropna()
        instruction_dup_rate = instructions.duplicated().mean()

        # Check for exact response duplicates
        responses = data[response_col].dropna()
        response_dup_rate = responses.duplicated().mean()

        # Check for exact pair duplicates
        pairs = data[[instruction_col, response_col]].dropna()
        pair_dup_rate = pairs.duplicated().mean()

        # Overall duplicate rate (weighted)
        overall_dup_rate = (
            instruction_dup_rate * 0.4 +
            response_dup_rate * 0.2 +
            pair_dup_rate * 0.4
        )

        # Invert for score
        score = 1.0 - overall_dup_rate

        return MetricResult(
            metric=self.metric,
            value=float(score),
            details={
                "instruction_duplicate_rate": float(instruction_dup_rate),
                "response_duplicate_rate": float(response_dup_rate),
                "pair_duplicate_rate": float(pair_dup_rate),
                "overall_duplicate_rate": float(overall_dup_rate),
            },
        )


class ResponseCompletenessEvaluator(MetricEvaluator):
    """Evaluates completeness of responses."""

    @property
    def metric(self) -> BenchmarkMetric:
        return BenchmarkMetric.RESPONSE_COMPLETENESS

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str,
        response_col: str,
    ) -> MetricResult:
        responses = data[response_col].dropna().astype(str)

        scores = []
        empty_count = 0
        short_count = 0

        for resp in responses:
            length = len(resp.strip())

            if length == 0:
                scores.append(0.0)
                empty_count += 1
            elif length < 10:
                scores.append(0.3)
                short_count += 1
            elif length < 50:
                scores.append(0.6)
            elif length < 200:
                scores.append(0.85)
            else:
                scores.append(1.0)

        avg_score = np.mean(scores) if scores else 0.0

        return MetricResult(
            metric=self.metric,
            value=float(avg_score),
            details={
                "empty_responses": empty_count,
                "short_responses": short_count,
                "total_responses": len(responses),
                "avg_length": float(responses.str.len().mean()) if len(responses) > 0 else 0,
            },
        )


class TaskDiversityEvaluator(MetricEvaluator):
    """Evaluates diversity of tasks/instruction types."""

    TASK_PATTERNS = {
        "question_answering": ["what is", "what are", "who is", "when did", "where is"],
        "summarization": ["summarize", "summary", "brief", "tldr"],
        "translation": ["translate", "translation", "in english", "in spanish"],
        "code": ["write code", "function", "implement", "debug", "code that"],
        "creative": ["write a story", "write a poem", "creative", "imagine"],
        "analysis": ["analyze", "analysis", "evaluate", "assess"],
        "explanation": ["explain", "how does", "why does", "describe how"],
        "instruction": ["how to", "steps to", "guide", "tutorial"],
        "comparison": ["compare", "difference between", "versus", "vs"],
        "classification": ["classify", "categorize", "which category"],
    }

    @property
    def metric(self) -> BenchmarkMetric:
        return BenchmarkMetric.TASK_DIVERSITY

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str,
        response_col: str,
    ) -> MetricResult:
        instructions = data[instruction_col].dropna().astype(str)

        task_counts: dict[str, int] = {task: 0 for task in self.TASK_PATTERNS}
        task_counts["other"] = 0

        for inst in instructions:
            inst_lower = inst.lower()
            matched = False

            for task, patterns in self.TASK_PATTERNS.items():
                if any(p in inst_lower for p in patterns):
                    task_counts[task] += 1
                    matched = True
                    break

            if not matched:
                task_counts["other"] += 1

        # Compute diversity score using normalized entropy
        total = sum(task_counts.values())
        if total == 0:
            return MetricResult(metric=self.metric, value=0.0)

        probs = [c / total for c in task_counts.values() if c > 0]
        if len(probs) <= 1:
            entropy = 0.0
        else:
            entropy = -sum(p * np.log2(p) for p in probs)
            max_entropy = np.log2(len(self.TASK_PATTERNS) + 1)
            entropy = entropy / max_entropy  # Normalize to 0-1

        return MetricResult(
            metric=self.metric,
            value=float(entropy),
            details={
                "task_distribution": task_counts,
                "n_task_types": sum(1 for c in task_counts.values() if c > 0),
            },
        )


class PIIRateEvaluator(MetricEvaluator):
    """Evaluates PII leakage rate."""

    import re

    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    @property
    def metric(self) -> BenchmarkMetric:
        return BenchmarkMetric.PII_RATE

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str,
        response_col: str,
    ) -> MetricResult:
        import re

        pii_counts: dict[str, int] = {pii_type: 0 for pii_type in self.PII_PATTERNS}
        total_samples = 0

        for col in [instruction_col, response_col]:
            texts = data[col].dropna().astype(str)
            for text in texts:
                total_samples += 1
                for pii_type, pattern in self.PII_PATTERNS.items():
                    if re.search(pattern, text):
                        pii_counts[pii_type] += 1

        total_pii = sum(pii_counts.values())
        pii_rate = total_pii / max(total_samples, 1)

        # Invert for score
        score = 1.0 - min(1.0, pii_rate * 10)  # Scale up small rates

        return MetricResult(
            metric=self.metric,
            value=float(score),
            details={
                "pii_counts": pii_counts,
                "total_pii": total_pii,
                "pii_rate": float(pii_rate),
            },
        )


class FMBenchmark:
    """Foundation Model Benchmark for LLM training data quality."""

    # Reference percentiles for common metrics (based on typical datasets)
    REFERENCE_PERCENTILES = {
        BenchmarkMetric.INSTRUCTION_CLARITY: [0.5, 0.6, 0.7, 0.8, 0.9],
        BenchmarkMetric.LEXICAL_DIVERSITY: [0.3, 0.4, 0.5, 0.6, 0.7],
        BenchmarkMetric.TOXICITY_RATE: [0.9, 0.93, 0.96, 0.98, 0.99],
        BenchmarkMetric.DUPLICATE_RATE: [0.7, 0.8, 0.9, 0.95, 0.99],
        BenchmarkMetric.RESPONSE_COMPLETENESS: [0.5, 0.65, 0.75, 0.85, 0.92],
        BenchmarkMetric.TASK_DIVERSITY: [0.3, 0.45, 0.55, 0.65, 0.75],
        BenchmarkMetric.PII_RATE: [0.8, 0.9, 0.95, 0.98, 0.99],
    }

    def __init__(self, evaluators: list[MetricEvaluator] | None = None):
        """Initialize the benchmark.

        Args:
            evaluators: Custom evaluators. If None, uses default set.
        """
        if evaluators is None:
            self.evaluators = [
                InstructionClarityEvaluator(),
                LexicalDiversityEvaluator(),
                ToxicityRateEvaluator(),
                DuplicateRateEvaluator(),
                ResponseCompletenessEvaluator(),
                TaskDiversityEvaluator(),
                PIIRateEvaluator(),
            ]
        else:
            self.evaluators = evaluators

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str = "instruction",
        response_col: str = "response",
        dataset_name: str = "unnamed",
        category: DatasetCategory = DatasetCategory.GENERAL,
    ) -> BenchmarkResult:
        """Run full benchmark evaluation.

        Args:
            data: DataFrame with instruction-response pairs
            instruction_col: Name of instruction column
            response_col: Name of response column
            dataset_name: Name of the dataset
            category: Category of the dataset

        Returns:
            BenchmarkResult with all metrics
        """
        logger.info("Running FM benchmark on dataset: %s (%d samples)", dataset_name, len(data))

        # Generate dataset ID
        dataset_id = self._generate_dataset_id(data, dataset_name)

        # Run all evaluators
        metrics: dict[BenchmarkMetric, MetricResult] = {}
        for evaluator in self.evaluators:
            try:
                result = evaluator.evaluate(data, instruction_col, response_col)
                result.percentile = self._compute_percentile(result.metric, result.value)
                metrics[evaluator.metric] = result
                logger.debug("Evaluated %s: %.3f", evaluator.metric.value, result.value)
            except Exception as e:
                logger.warning("Failed to evaluate %s: %s", evaluator.metric.value, e)

        # Compute overall score
        overall_score = self._compute_overall_score(metrics)
        grade = self._compute_grade(overall_score)

        # Generate insights
        strengths = self._identify_strengths(metrics)
        weaknesses = self._identify_weaknesses(metrics)
        recommendations = self._generate_recommendations(metrics, weaknesses)

        return BenchmarkResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            timestamp=datetime.now(),
            n_samples=len(data),
            category=category,
            metrics=metrics,
            overall_score=overall_score,
            grade=grade,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    def _generate_dataset_id(self, data: pd.DataFrame, name: str) -> str:
        """Generate unique dataset identifier."""
        content_hash = hashlib.md5(
            f"{name}_{len(data)}_{data.columns.tolist()}".encode()
        ).hexdigest()[:12]
        return f"ds_{content_hash}"

    def _compute_percentile(self, metric: BenchmarkMetric, value: float) -> float | None:
        """Compute percentile rank for a metric value."""
        if metric not in self.REFERENCE_PERCENTILES:
            return None

        percentiles = self.REFERENCE_PERCENTILES[metric]
        for i, threshold in enumerate(percentiles):
            if value < threshold:
                return (i / len(percentiles)) * 100
        return 100.0

    def _compute_overall_score(self, metrics: dict[BenchmarkMetric, MetricResult]) -> float:
        """Compute overall quality score."""
        if not metrics:
            return 0.0

        # Weighted average with importance weights
        weights = {
            BenchmarkMetric.INSTRUCTION_CLARITY: 1.5,
            BenchmarkMetric.RESPONSE_COMPLETENESS: 1.5,
            BenchmarkMetric.TOXICITY_RATE: 2.0,
            BenchmarkMetric.PII_RATE: 2.0,
            BenchmarkMetric.DUPLICATE_RATE: 1.0,
            BenchmarkMetric.LEXICAL_DIVERSITY: 0.8,
            BenchmarkMetric.TASK_DIVERSITY: 0.8,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for metric, result in metrics.items():
            weight = weights.get(metric, 1.0)
            weighted_sum += result.value * weight * 100
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _compute_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _identify_strengths(self, metrics: dict[BenchmarkMetric, MetricResult]) -> list[str]:
        """Identify dataset strengths."""
        strengths = []

        for metric, result in metrics.items():
            if result.value >= 0.9:
                strengths.append(f"Excellent {metric.value.replace('_', ' ')}")
            elif result.value >= 0.8:
                strengths.append(f"Good {metric.value.replace('_', ' ')}")

        return strengths

    def _identify_weaknesses(self, metrics: dict[BenchmarkMetric, MetricResult]) -> list[str]:
        """Identify dataset weaknesses."""
        weaknesses = []

        for metric, result in metrics.items():
            if result.value < 0.5:
                weaknesses.append(f"Poor {metric.value.replace('_', ' ')}")
            elif result.value < 0.7:
                weaknesses.append(f"Below average {metric.value.replace('_', ' ')}")

        return weaknesses

    def _generate_recommendations(
        self,
        metrics: dict[BenchmarkMetric, MetricResult],
        weaknesses: list[str],
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        for metric, result in metrics.items():
            if result.value < 0.7:
                if metric == BenchmarkMetric.INSTRUCTION_CLARITY:
                    recommendations.append(
                        "Improve instruction clarity by adding specific details and context"
                    )
                elif metric == BenchmarkMetric.DUPLICATE_RATE:
                    recommendations.append(
                        "Run deduplication to remove redundant instruction-response pairs"
                    )
                elif metric == BenchmarkMetric.TOXICITY_RATE:
                    recommendations.append(
                        "Review and filter samples flagged for toxic content"
                    )
                elif metric == BenchmarkMetric.TASK_DIVERSITY:
                    recommendations.append(
                        "Add more diverse task types to improve coverage"
                    )
                elif metric == BenchmarkMetric.PII_RATE:
                    recommendations.append(
                        "Scan and redact PII from the dataset"
                    )
                elif metric == BenchmarkMetric.RESPONSE_COMPLETENESS:
                    recommendations.append(
                        "Review and expand short or incomplete responses"
                    )
                elif metric == BenchmarkMetric.LEXICAL_DIVERSITY:
                    recommendations.append(
                        "Increase vocabulary diversity by varying response styles"
                    )

        return recommendations


@dataclass
class LeaderboardEntry:
    """Entry in the benchmark leaderboard."""

    rank: int
    dataset_id: str
    dataset_name: str
    category: DatasetCategory
    overall_score: float
    grade: str
    n_samples: int
    submitted_at: datetime
    metrics_summary: dict[str, float]


class BenchmarkLeaderboard:
    """Leaderboard for comparing dataset quality benchmarks."""

    def __init__(self, storage_path: Path | str | None = None):
        """Initialize leaderboard.

        Args:
            storage_path: Path to store leaderboard data (JSON)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._entries: list[LeaderboardEntry] = []
        self._load()

    def _load(self) -> None:
        """Load leaderboard from storage."""
        if self.storage_path and self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    self._entries = [
                        LeaderboardEntry(
                            rank=e["rank"],
                            dataset_id=e["dataset_id"],
                            dataset_name=e["dataset_name"],
                            category=DatasetCategory(e["category"]),
                            overall_score=e["overall_score"],
                            grade=e["grade"],
                            n_samples=e["n_samples"],
                            submitted_at=datetime.fromisoformat(e["submitted_at"]),
                            metrics_summary=e["metrics_summary"],
                        )
                        for e in data.get("entries", [])
                    ]
            except Exception as e:
                logger.warning("Failed to load leaderboard: %s", e)

    def _save(self) -> None:
        """Save leaderboard to storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(
                    {
                        "entries": [
                            {
                                "rank": e.rank,
                                "dataset_id": e.dataset_id,
                                "dataset_name": e.dataset_name,
                                "category": e.category.value,
                                "overall_score": e.overall_score,
                                "grade": e.grade,
                                "n_samples": e.n_samples,
                                "submitted_at": e.submitted_at.isoformat(),
                                "metrics_summary": e.metrics_summary,
                            }
                            for e in self._entries
                        ]
                    },
                    f,
                    indent=2,
                )

    def submit(self, result: BenchmarkResult) -> LeaderboardEntry:
        """Submit a benchmark result to the leaderboard.

        Args:
            result: Benchmark result to submit

        Returns:
            LeaderboardEntry with assigned rank
        """
        metrics_summary = {
            m.value: r.value for m, r in result.metrics.items()
        }

        entry = LeaderboardEntry(
            rank=0,  # Will be computed
            dataset_id=result.dataset_id,
            dataset_name=result.dataset_name,
            category=result.category,
            overall_score=result.overall_score,
            grade=result.grade,
            n_samples=result.n_samples,
            submitted_at=datetime.now(),
            metrics_summary=metrics_summary,
        )

        # Remove existing entry for same dataset
        self._entries = [e for e in self._entries if e.dataset_id != result.dataset_id]

        # Add and re-rank
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.overall_score, reverse=True)

        for i, e in enumerate(self._entries):
            e.rank = i + 1

        entry.rank = next(
            e.rank for e in self._entries if e.dataset_id == result.dataset_id
        )

        self._save()
        return entry

    def get_top(
        self,
        n: int = 10,
        category: DatasetCategory | None = None,
    ) -> list[LeaderboardEntry]:
        """Get top N entries.

        Args:
            n: Number of entries to return
            category: Filter by category

        Returns:
            List of top entries
        """
        entries = self._entries
        if category:
            entries = [e for e in entries if e.category == category]
        return entries[:n]

    def get_percentile(self, score: float) -> float:
        """Get percentile rank for a score.

        Args:
            score: Overall score

        Returns:
            Percentile (0-100)
        """
        if not self._entries:
            return 100.0

        better_count = sum(1 for e in self._entries if e.overall_score < score)
        return (better_count / len(self._entries)) * 100

    def format_leaderboard(
        self,
        n: int = 10,
        category: DatasetCategory | None = None,
    ) -> str:
        """Format leaderboard as string table.

        Args:
            n: Number of entries
            category: Filter by category

        Returns:
            Formatted leaderboard string
        """
        entries = self.get_top(n, category)

        lines = [
            "Foundation Model Quality Leaderboard",
            "=" * 70,
            f"{'Rank':<6}{'Dataset':<25}{'Score':<10}{'Grade':<8}{'Samples':<12}{'Category':<15}",
            "-" * 70,
        ]

        for entry in entries:
            lines.append(
                f"{entry.rank:<6}{entry.dataset_name[:24]:<25}"
                f"{entry.overall_score:>6.1f}   {entry.grade:<8}"
                f"{entry.n_samples:>10,}  {entry.category.value:<15}"
            )

        if not entries:
            lines.append("No entries yet")

        return "\n".join(lines)


def benchmark_dataset(
    data: pd.DataFrame,
    instruction_col: str = "instruction",
    response_col: str = "response",
    dataset_name: str = "unnamed",
    category: str = "general",
) -> BenchmarkResult:
    """Convenience function to benchmark a dataset.

    Args:
        data: DataFrame with instruction-response pairs
        instruction_col: Name of instruction column
        response_col: Name of response column
        dataset_name: Name of the dataset
        category: Category of the dataset

    Returns:
        BenchmarkResult with all metrics
    """
    benchmark = FMBenchmark()
    return benchmark.evaluate(
        data=data,
        instruction_col=instruction_col,
        response_col=response_col,
        dataset_name=dataset_name,
        category=DatasetCategory(category),
    )


def compare_datasets(
    datasets: dict[str, pd.DataFrame],
    instruction_col: str = "instruction",
    response_col: str = "response",
) -> pd.DataFrame:
    """Compare multiple datasets on benchmark metrics.

    Args:
        datasets: Dictionary of {name: DataFrame}
        instruction_col: Name of instruction column
        response_col: Name of response column

    Returns:
        DataFrame with comparison results
    """
    benchmark = FMBenchmark()
    results = []

    for name, data in datasets.items():
        result = benchmark.evaluate(
            data=data,
            instruction_col=instruction_col,
            response_col=response_col,
            dataset_name=name,
        )

        row = {
            "dataset": name,
            "samples": result.n_samples,
            "overall_score": result.overall_score,
            "grade": result.grade,
        }

        for metric, metric_result in result.metrics.items():
            row[metric.value] = metric_result.value

        results.append(row)

    return pd.DataFrame(results)
