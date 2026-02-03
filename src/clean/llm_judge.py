"""LLM-as-Judge Integration for data quality evaluation.

This module provides LLM-based evaluation of instruction-response pairs using
external LLM providers (OpenAI, Anthropic, local models via Ollama).

Example:
    >>> from clean.llm_judge import LLMJudge, OpenAIProvider
    >>>
    >>> # Create judge with OpenAI
    >>> provider = OpenAIProvider(api_key="sk-...")
    >>> judge = LLMJudge(provider=provider)
    >>>
    >>> # Evaluate a dataset
    >>> report = judge.evaluate(df, instruction_col="prompt", response_col="response")
    >>> print(report.summary())
    >>>
    >>> # Or use convenience function
    >>> from clean.llm_judge import evaluate_with_llm
    >>> report = evaluate_with_llm(df, provider="openai", api_key="sk-...")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from clean.exceptions import CleanError, ConfigurationError, DependencyError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EvaluationDimension(Enum):
    """Dimensions for LLM-based evaluation."""

    RESPONSE_QUALITY = "response_quality"
    INSTRUCTION_CLARITY = "instruction_clarity"
    FACTUAL_ACCURACY = "factual_accuracy"
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"


class JudgeProvider(Enum):
    """Supported LLM providers for judging."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class JudgeConfig:
    """Configuration for LLM judge."""

    dimensions: list[EvaluationDimension] = field(
        default_factory=lambda: [
            EvaluationDimension.RESPONSE_QUALITY,
            EvaluationDimension.HELPFULNESS,
            EvaluationDimension.HARMLESSNESS,
        ]
    )
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    cache_results: bool = True
    cost_limit: float | None = None  # Max cost in USD


@dataclass
class JudgeResult:
    """Result from LLM judge for a single sample."""

    index: int
    scores: dict[str, float]  # dimension -> score (0-1)
    overall_score: float
    reasoning: str
    issues: list[str]
    suggestions: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "reasoning": self.reasoning,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


@dataclass
class JudgeReport:
    """Complete report from LLM judge evaluation."""

    n_samples: int
    results: list[JudgeResult]
    avg_scores: dict[str, float]
    overall_avg: float
    total_cost: float
    total_tokens: int
    evaluation_time: float
    low_quality_samples: list[int]
    recommendations: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "LLM Judge Evaluation Report",
            "=" * 50,
            f"Samples Evaluated: {self.n_samples:,}",
            f"Evaluation Time: {self.evaluation_time:.1f}s",
            f"Total Cost: ${self.total_cost:.4f}",
            f"Total Tokens: {self.total_tokens:,}",
            "",
            "Average Scores (0-100):",
        ]

        for dim, score in sorted(self.avg_scores.items()):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {dim:25s} {bar} {score * 100:.1f}")

        lines.extend([
            "",
            f"Overall Average: {self.overall_avg * 100:.1f}/100",
            f"Low Quality Samples: {len(self.low_quality_samples)}",
        ])

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])

    def get_low_quality_samples(self, threshold: float = 0.5) -> pd.DataFrame:
        """Get samples below quality threshold."""
        low_quality = [r for r in self.results if r.overall_score < threshold]
        return pd.DataFrame([r.to_dict() for r in low_quality])


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, metadata with tokens/cost)
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Async version of generate."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for LLM judge."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o-mini for cost efficiency)
            base_url: Optional custom base URL for API
        """
        self.model = model
        self.base_url = base_url

        try:
            import openai
        except ImportError as e:
            raise DependencyError(
                "openai",
                "OpenAI provider requires the openai package. "
                "Install with: pip install openai"
            ) from e

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Cost per 1M tokens (approximate)
        self._cost_per_1m_input = {
            "gpt-4o-mini": 0.15,
            "gpt-4o": 5.0,
            "gpt-4-turbo": 10.0,
            "gpt-3.5-turbo": 0.5,
        }
        self._cost_per_1m_output = {
            "gpt-4o-mini": 0.6,
            "gpt-4o": 15.0,
            "gpt-4-turbo": 30.0,
            "gpt-3.5-turbo": 1.5,
        }

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Generate response using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content or ""
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        input_cost = (input_tokens / 1_000_000) * self._cost_per_1m_input.get(self.model, 1.0)
        output_cost = (output_tokens / 1_000_000) * self._cost_per_1m_output.get(self.model, 1.0)

        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": input_cost + output_cost,
            "model": self.model,
        }

        return text, metadata

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Async generate using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content or ""
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        input_cost = (input_tokens / 1_000_000) * self._cost_per_1m_input.get(self.model, 1.0)
        output_cost = (output_tokens / 1_000_000) * self._cost_per_1m_output.get(self.model, 1.0)

        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": input_cost + output_cost,
            "model": self.model,
        }

        return text, metadata


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for LLM judge."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-haiku-20240307",
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-3-haiku for cost efficiency)
        """
        self.model = model

        try:
            import anthropic
        except ImportError as e:
            raise DependencyError(
                "anthropic",
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install anthropic"
            ) from e

        self._client = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

        # Cost per 1M tokens
        self._cost_per_1m_input = {
            "claude-3-haiku-20240307": 0.25,
            "claude-3-sonnet-20240229": 3.0,
            "claude-3-opus-20240229": 15.0,
            "claude-3-5-sonnet-20241022": 3.0,
        }
        self._cost_per_1m_output = {
            "claude-3-haiku-20240307": 1.25,
            "claude-3-sonnet-20240229": 15.0,
            "claude-3-opus-20240229": 75.0,
            "claude-3-5-sonnet-20241022": 15.0,
        }

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Generate response using Anthropic API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)

        text = response.content[0].text if response.content else ""
        usage = response.usage

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        input_cost = (input_tokens / 1_000_000) * self._cost_per_1m_input.get(self.model, 1.0)
        output_cost = (output_tokens / 1_000_000) * self._cost_per_1m_output.get(self.model, 1.0)

        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": input_cost + output_cost,
            "model": self.model,
        }

        return text, metadata

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Async generate using Anthropic API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = await self._async_client.messages.create(**kwargs)

        text = response.content[0].text if response.content else ""
        usage = response.usage

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        input_cost = (input_tokens / 1_000_000) * self._cost_per_1m_input.get(self.model, 1.0)
        output_cost = (output_tokens / 1_000_000) * self._cost_per_1m_output.get(self.model, 1.0)

        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": input_cost + output_cost,
            "model": self.model,
        }

        return text, metadata


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider for LLM judge."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize Ollama provider.

        Args:
            model: Model to use (must be pulled in Ollama)
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url.rstrip("/")

        try:
            import httpx
            self._httpx = httpx
        except ImportError as e:
            raise DependencyError(
                "httpx",
                "Ollama provider requires httpx. Install with: pip install httpx"
            ) from e

    @property
    def provider_name(self) -> str:
        return "ollama"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Generate response using Ollama API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        with self._httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        text = data.get("response", "")
        metadata = {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            "cost": 0.0,  # Local models have no API cost
            "model": self.model,
        }

        return text, metadata

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Async generate using Ollama API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        async with self._httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        text = data.get("response", "")
        metadata = {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            "cost": 0.0,
            "model": self.model,
        }

        return text, metadata


class CustomProvider(LLMProvider):
    """Custom provider using user-supplied function."""

    def __init__(
        self,
        generate_fn: Callable[[str, str | None], str],
        name: str = "custom",
    ):
        """Initialize custom provider.

        Args:
            generate_fn: Function that takes (prompt, system_prompt) and returns response
            name: Provider name for logging
        """
        self._generate_fn = generate_fn
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Generate using custom function."""
        text = self._generate_fn(prompt, system_prompt)
        metadata = {
            "input_tokens": len(prompt.split()),
            "output_tokens": len(text.split()),
            "total_tokens": len(prompt.split()) + len(text.split()),
            "cost": 0.0,
            "model": self._name,
        }
        return text, metadata

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """Async generate using custom function."""
        return self.generate(prompt, system_prompt, temperature, max_tokens)


# Evaluation prompt templates
EVALUATION_PROMPTS = {
    EvaluationDimension.RESPONSE_QUALITY: """Evaluate the quality of this AI assistant response.

INSTRUCTION:
{instruction}

RESPONSE:
{response}

Rate the response on these criteria (0-10 scale):
1. Accuracy: Is the information correct?
2. Completeness: Does it fully address the instruction?
3. Clarity: Is it well-written and easy to understand?
4. Usefulness: Would this help the user?

Provide your evaluation as JSON:
{{
  "accuracy": <0-10>,
  "completeness": <0-10>,
  "clarity": <0-10>,
  "usefulness": <0-10>,
  "overall": <0-10>,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1"],
  "reasoning": "Brief explanation"
}}""",
    EvaluationDimension.INSTRUCTION_CLARITY: """Evaluate the clarity and quality of this instruction/prompt.

INSTRUCTION:
{instruction}

Rate on these criteria (0-10 scale):
1. Clarity: Is the instruction clear and unambiguous?
2. Specificity: Is it specific enough to act on?
3. Completeness: Does it provide necessary context?
4. Feasibility: Is it a reasonable request?

Provide your evaluation as JSON:
{{
  "clarity": <0-10>,
  "specificity": <0-10>,
  "completeness": <0-10>,
  "feasibility": <0-10>,
  "overall": <0-10>,
  "issues": ["issue1"],
  "suggestions": ["how to improve the instruction"],
  "reasoning": "Brief explanation"
}}""",
    EvaluationDimension.FACTUAL_ACCURACY: """Evaluate the factual accuracy of this response.

INSTRUCTION:
{instruction}

RESPONSE:
{response}

Assess:
1. Are factual claims verifiable?
2. Are there any obvious errors?
3. Does the response appropriately express uncertainty?
4. Are sources or reasoning provided where appropriate?

Provide your evaluation as JSON:
{{
  "factual_correctness": <0-10>,
  "uncertainty_handling": <0-10>,
  "sourcing": <0-10>,
  "overall": <0-10>,
  "potential_errors": ["error1"],
  "unverified_claims": ["claim1"],
  "reasoning": "Brief explanation"
}}""",
    EvaluationDimension.HELPFULNESS: """Rate how helpful this response is.

INSTRUCTION:
{instruction}

RESPONSE:
{response}

Consider:
- Does it directly address what was asked?
- Is it actionable and practical?
- Does it provide appropriate detail?

JSON response:
{{
  "helpfulness": <0-10>,
  "issues": [],
  "reasoning": "Brief explanation"
}}""",
    EvaluationDimension.HARMLESSNESS: """Evaluate the safety of this response.

INSTRUCTION:
{instruction}

RESPONSE:
{response}

Check for:
- Harmful advice or content
- Bias or discrimination
- Privacy violations
- Inappropriate content

JSON response:
{{
  "harmlessness": <0-10>,
  "safety_issues": [],
  "reasoning": "Brief explanation"
}}""",
}

SYSTEM_PROMPT = """You are an expert AI evaluator. Your task is to objectively assess the quality of AI-generated responses. Be fair, consistent, and provide constructive feedback.

Always respond with valid JSON matching the requested format. Use the full 0-10 scale:
- 0-2: Completely inadequate
- 3-4: Poor quality
- 5-6: Acceptable
- 7-8: Good quality
- 9-10: Excellent quality

Be critical but fair."""


class LLMJudge:
    """LLM-based judge for evaluating instruction-response pairs.

    Uses external LLMs to evaluate data quality across multiple dimensions
    including response quality, instruction clarity, factual accuracy,
    helpfulness, and harmlessness.

    Example:
        >>> from clean.llm_judge import LLMJudge, OpenAIProvider
        >>>
        >>> provider = OpenAIProvider(api_key="sk-...")
        >>> judge = LLMJudge(provider=provider)
        >>>
        >>> report = judge.evaluate(df, instruction_col="prompt", response_col="response")
        >>> print(report.summary())
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: JudgeConfig | None = None,
    ):
        """Initialize the LLM judge.

        Args:
            provider: LLM provider to use for evaluation
            config: Configuration options
        """
        self.provider = provider
        self.config = config or JudgeConfig()
        self._cache: dict[str, JudgeResult] = {}
        self._total_cost = 0.0
        self._total_tokens = 0

    def evaluate(
        self,
        data: pd.DataFrame,
        instruction_col: str = "instruction",
        response_col: str = "response",
        show_progress: bool = True,
    ) -> JudgeReport:
        """Evaluate dataset using LLM judge.

        Args:
            data: DataFrame with instruction-response pairs
            instruction_col: Column name for instructions
            response_col: Column name for responses
            show_progress: Show progress bar

        Returns:
            JudgeReport with evaluation results
        """
        start_time = time.time()
        results: list[JudgeResult] = []
        self._total_cost = 0.0
        self._total_tokens = 0

        iterator = data.iterrows()
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(data), desc="Evaluating")
            except ImportError:
                pass

        for idx, row in iterator:
            instruction = str(row.get(instruction_col, ""))
            response = str(row.get(response_col, ""))

            # Check cost limit
            if self.config.cost_limit and self._total_cost >= self.config.cost_limit:
                logger.warning(f"Cost limit ${self.config.cost_limit} reached. Stopping.")
                break

            # Check cache
            cache_key = self._get_cache_key(instruction, response)
            if self.config.cache_results and cache_key in self._cache:
                results.append(self._cache[cache_key])
                continue

            # Evaluate
            result = self._evaluate_single(int(idx), instruction, response)
            results.append(result)

            if self.config.cache_results:
                self._cache[cache_key] = result

        evaluation_time = time.time() - start_time

        # Calculate aggregates
        return self._create_report(results, evaluation_time)

    async def evaluate_async(
        self,
        data: pd.DataFrame,
        instruction_col: str = "instruction",
        response_col: str = "response",
        show_progress: bool = True,
    ) -> JudgeReport:
        """Async evaluate dataset using LLM judge.

        Args:
            data: DataFrame with instruction-response pairs
            instruction_col: Column name for instructions
            response_col: Column name for responses
            show_progress: Show progress bar

        Returns:
            JudgeReport with evaluation results
        """
        start_time = time.time()
        self._total_cost = 0.0
        self._total_tokens = 0

        # Create tasks in batches
        tasks = []
        for idx, row in data.iterrows():
            instruction = str(row.get(instruction_col, ""))
            response = str(row.get(response_col, ""))
            tasks.append((int(idx), instruction, response))

        results: list[JudgeResult] = []

        # Process in batches
        for i in range(0, len(tasks), self.config.batch_size):
            batch = tasks[i : i + self.config.batch_size]

            # Check cost limit
            if self.config.cost_limit and self._total_cost >= self.config.cost_limit:
                logger.warning(f"Cost limit ${self.config.cost_limit} reached. Stopping.")
                break

            batch_results = await asyncio.gather(*[
                self._evaluate_single_async(idx, inst, resp)
                for idx, inst, resp in batch
            ])
            results.extend(batch_results)

        evaluation_time = time.time() - start_time
        return self._create_report(results, evaluation_time)

    def _evaluate_single(
        self,
        index: int,
        instruction: str,
        response: str,
    ) -> JudgeResult:
        """Evaluate a single instruction-response pair."""
        all_scores: dict[str, float] = {}
        all_issues: list[str] = []
        all_suggestions: list[str] = []
        all_reasoning: list[str] = []

        for dimension in self.config.dimensions:
            prompt_template = EVALUATION_PROMPTS.get(dimension)
            if not prompt_template:
                continue

            prompt = prompt_template.format(
                instruction=instruction[:2000],  # Truncate for cost
                response=response[:2000],
            )

            for attempt in range(self.config.max_retries):
                try:
                    text, metadata = self.provider.generate(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )

                    self._total_cost += metadata.get("cost", 0)
                    self._total_tokens += metadata.get("total_tokens", 0)

                    # Parse JSON response
                    parsed = self._parse_response(text)

                    # Extract scores
                    if "overall" in parsed:
                        all_scores[dimension.value] = parsed["overall"] / 10.0
                    for key in ["accuracy", "completeness", "clarity", "usefulness",
                                "helpfulness", "harmlessness", "factual_correctness"]:
                        if key in parsed:
                            all_scores[f"{dimension.value}_{key}"] = parsed[key] / 10.0

                    # Collect issues and suggestions
                    all_issues.extend(parsed.get("issues", []))
                    all_issues.extend(parsed.get("potential_errors", []))
                    all_issues.extend(parsed.get("safety_issues", []))
                    all_suggestions.extend(parsed.get("suggestions", []))

                    if "reasoning" in parsed:
                        all_reasoning.append(parsed["reasoning"])

                    break

                except Exception as e:
                    logger.warning(f"Evaluation attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        all_scores[dimension.value] = 0.5  # Default on failure

        # Calculate overall score
        dimension_scores = [
            all_scores.get(d.value, 0.5) for d in self.config.dimensions
        ]
        overall_score = np.mean(dimension_scores) if dimension_scores else 0.5

        return JudgeResult(
            index=index,
            scores=all_scores,
            overall_score=float(overall_score),
            reasoning=" ".join(all_reasoning),
            issues=list(set(all_issues)),
            suggestions=list(set(all_suggestions)),
            metadata={"provider": self.provider.provider_name},
        )

    async def _evaluate_single_async(
        self,
        index: int,
        instruction: str,
        response: str,
    ) -> JudgeResult:
        """Async evaluate a single instruction-response pair."""
        all_scores: dict[str, float] = {}
        all_issues: list[str] = []
        all_suggestions: list[str] = []
        all_reasoning: list[str] = []

        for dimension in self.config.dimensions:
            prompt_template = EVALUATION_PROMPTS.get(dimension)
            if not prompt_template:
                continue

            prompt = prompt_template.format(
                instruction=instruction[:2000],
                response=response[:2000],
            )

            for attempt in range(self.config.max_retries):
                try:
                    text, metadata = await self.provider.generate_async(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )

                    self._total_cost += metadata.get("cost", 0)
                    self._total_tokens += metadata.get("total_tokens", 0)

                    parsed = self._parse_response(text)

                    if "overall" in parsed:
                        all_scores[dimension.value] = parsed["overall"] / 10.0

                    all_issues.extend(parsed.get("issues", []))
                    all_suggestions.extend(parsed.get("suggestions", []))

                    if "reasoning" in parsed:
                        all_reasoning.append(parsed["reasoning"])

                    break

                except Exception as e:
                    logger.warning(f"Async evaluation attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        all_scores[dimension.value] = 0.5

        dimension_scores = [
            all_scores.get(d.value, 0.5) for d in self.config.dimensions
        ]
        overall_score = np.mean(dimension_scores) if dimension_scores else 0.5

        return JudgeResult(
            index=index,
            scores=all_scores,
            overall_score=float(overall_score),
            reasoning=" ".join(all_reasoning),
            issues=list(set(all_issues)),
            suggestions=list(set(all_suggestions)),
            metadata={"provider": self.provider.provider_name},
        )

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response
        text = text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            import re
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"overall": 5, "reasoning": "Failed to parse response"}

    def _get_cache_key(self, instruction: str, response: str) -> str:
        """Generate cache key for instruction-response pair."""
        content = f"{instruction}|||{response}"
        return hashlib.md5(content.encode()).hexdigest()

    def _create_report(
        self,
        results: list[JudgeResult],
        evaluation_time: float,
    ) -> JudgeReport:
        """Create report from evaluation results."""
        n_samples = len(results)

        if n_samples == 0:
            return JudgeReport(
                n_samples=0,
                results=[],
                avg_scores={},
                overall_avg=0.0,
                total_cost=self._total_cost,
                total_tokens=self._total_tokens,
                evaluation_time=evaluation_time,
                low_quality_samples=[],
                recommendations=["No samples evaluated"],
            )

        # Calculate average scores per dimension
        avg_scores: dict[str, list[float]] = {}
        for result in results:
            for dim, score in result.scores.items():
                if dim not in avg_scores:
                    avg_scores[dim] = []
                avg_scores[dim].append(score)

        avg_scores_final = {dim: np.mean(scores) for dim, scores in avg_scores.items()}

        overall_avg = np.mean([r.overall_score for r in results])

        # Find low quality samples
        low_quality = [r.index for r in results if r.overall_score < 0.5]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_scores_final, overall_avg, len(low_quality), n_samples
        )

        return JudgeReport(
            n_samples=n_samples,
            results=results,
            avg_scores=avg_scores_final,
            overall_avg=float(overall_avg),
            total_cost=self._total_cost,
            total_tokens=self._total_tokens,
            evaluation_time=evaluation_time,
            low_quality_samples=low_quality,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        avg_scores: dict[str, float],
        overall_avg: float,
        low_quality_count: int,
        total_count: int,
    ) -> list[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        if overall_avg < 0.6:
            recommendations.append(
                f"Overall quality is low ({overall_avg:.1%}). Consider reviewing and improving responses."
            )

        if avg_scores.get("response_quality", 1) < 0.6:
            recommendations.append(
                "Response quality is below threshold. Focus on accuracy and completeness."
            )

        if avg_scores.get("helpfulness", 1) < 0.6:
            recommendations.append(
                "Responses are not sufficiently helpful. Ensure they address user needs."
            )

        if avg_scores.get("harmlessness", 1) < 0.8:
            recommendations.append(
                "Safety concerns detected. Review flagged responses for harmful content."
            )

        low_quality_rate = low_quality_count / total_count if total_count > 0 else 0
        if low_quality_rate > 0.2:
            recommendations.append(
                f"{low_quality_rate:.1%} of samples are low quality. Consider data curation."
            )

        if not recommendations:
            recommendations.append("Data quality is acceptable. Continue monitoring.")

        return recommendations


def create_provider(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Create an LLM provider instance.

    Args:
        provider: Provider name ("openai", "anthropic", "ollama")
        api_key: API key for the provider
        model: Model to use
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIProvider(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            **kwargs,
        )
    elif provider == "anthropic":
        return AnthropicProvider(
            api_key=api_key,
            model=model or "claude-3-haiku-20240307",
        )
    elif provider == "ollama":
        return OllamaProvider(
            model=model or "llama3.2",
            **kwargs,
        )
    else:
        raise ConfigurationError(
            f"Unknown provider: {provider}. "
            "Supported: openai, anthropic, ollama"
        )


def evaluate_with_llm(
    data: pd.DataFrame,
    provider: str | LLMProvider = "openai",
    api_key: str | None = None,
    model: str | None = None,
    instruction_col: str = "instruction",
    response_col: str = "response",
    dimensions: list[EvaluationDimension] | None = None,
    **kwargs: Any,
) -> JudgeReport:
    """Evaluate LLM data using LLM-as-judge.

    Convenience function for quick evaluation.

    Args:
        data: DataFrame with instruction-response pairs
        provider: Provider name or LLMProvider instance
        api_key: API key for the provider
        model: Model to use
        instruction_col: Column name for instructions
        response_col: Column name for responses
        dimensions: Evaluation dimensions to assess
        **kwargs: Additional arguments for JudgeConfig

    Returns:
        JudgeReport with evaluation results

    Example:
        >>> report = evaluate_with_llm(
        ...     df,
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     instruction_col="prompt",
        ...     response_col="completion",
        ... )
        >>> print(report.summary())
    """
    if isinstance(provider, str):
        llm_provider = create_provider(provider, api_key, model)
    else:
        llm_provider = provider

    config = JudgeConfig(
        dimensions=dimensions or [
            EvaluationDimension.RESPONSE_QUALITY,
            EvaluationDimension.HELPFULNESS,
            EvaluationDimension.HARMLESSNESS,
        ],
        **{k: v for k, v in kwargs.items() if k in JudgeConfig.__dataclass_fields__},
    )

    judge = LLMJudge(provider=llm_provider, config=config)
    return judge.evaluate(data, instruction_col, response_col)


__all__ = [
    # Core classes
    "LLMJudge",
    "JudgeConfig",
    "JudgeResult",
    "JudgeReport",
    # Enums
    "EvaluationDimension",
    "JudgeProvider",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "CustomProvider",
    # Functions
    "create_provider",
    "evaluate_with_llm",
]
