"""Optimization strategies for AutoML quality tuning.

This module provides strategy classes for different optimization algorithms,
implementing the Strategy pattern to reduce complexity in QualityTuner.

Example:
    >>> strategy = RandomSearchOptimizer(config)
    >>> best_params = strategy.optimize(X, y, model, evaluate_fn)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from clean.automl.tuner import TuningConfig


@dataclass
class OptimizationState:
    """Tracks optimization state across trials."""

    trials: list[dict[str, Any]]
    convergence_history: list[float]
    best_score: float
    best_params: Any  # ThresholdParams

    @classmethod
    def create(cls, initial_params: Any) -> OptimizationState:
        """Create initial optimization state."""
        return cls(
            trials=[],
            convergence_history=[],
            best_score=-np.inf,
            best_params=initial_params,
        )

    def record_trial(
        self,
        trial_num: int,
        params: Any,
        score: float,
    ) -> bool:
        """Record a trial result.

        Returns:
            True if this trial improved the best score
        """
        self.trials.append({
            "trial": trial_num,
            "params": params.to_dict() if hasattr(params, "to_dict") else params,
            "score": score,
        })

        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_params = params

        self.convergence_history.append(self.best_score)
        return improved


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies.

    Each strategy implements a specific search algorithm for finding
    optimal threshold parameters.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        ...

    @abstractmethod
    def optimize(
        self,
        config: TuningConfig,
        evaluate_fn: Callable[[Any], float],
        create_params_fn: Callable[[float, float, float], Any],
        state: OptimizationState,
        rng: np.random.RandomState,
    ) -> Any:
        """Run optimization to find best parameters.

        Args:
            config: Tuning configuration with search space and settings
            evaluate_fn: Function to evaluate a ThresholdParams, returns score
            create_params_fn: Factory to create ThresholdParams(le, oc, dt)
            state: Optimization state for tracking progress
            rng: Random number generator

        Returns:
            Best ThresholdParams found
        """
        ...


class GridSearchOptimizer(OptimizationStrategy):
    """Grid search optimization strategy.

    Exhaustively searches over a grid of parameter values.
    Best for small search spaces or when thorough coverage is needed.
    """

    @property
    def name(self) -> str:
        return "grid"

    def optimize(
        self,
        config: TuningConfig,
        evaluate_fn: Callable[[Any], float],
        create_params_fn: Callable[[float, float, float], Any],
        state: OptimizationState,
        rng: np.random.RandomState,  # noqa: ARG002
    ) -> Any:
        """Run grid search optimization."""
        import time

        # Create grid with 5 points per dimension
        le_values = np.linspace(*config.label_error_threshold_range, 5)
        oc_values = np.linspace(*config.outlier_contamination_range, 5)
        dt_values = np.linspace(*config.duplicate_threshold_range, 5)

        trial = 0
        start_time = time.time()

        for le in le_values:
            for oc in oc_values:
                for dt in dt_values:
                    # Check timeout
                    if (
                        config.timeout_seconds
                        and time.time() - start_time > config.timeout_seconds
                    ):
                        return state.best_params

                    params = create_params_fn(float(le), float(oc), float(dt))
                    score = evaluate_fn(params)
                    state.record_trial(trial, params, score)
                    trial += 1

        return state.best_params


class RandomSearchOptimizer(OptimizationStrategy):
    """Random search optimization strategy.

    Randomly samples parameter combinations from the search space.
    Often more efficient than grid search for high-dimensional spaces.
    """

    @property
    def name(self) -> str:
        return "random"

    def optimize(
        self,
        config: TuningConfig,
        evaluate_fn: Callable[[Any], float],
        create_params_fn: Callable[[float, float, float], Any],
        state: OptimizationState,
        rng: np.random.RandomState,
    ) -> Any:
        """Run random search optimization."""
        import time

        no_improvement_count = 0
        start_time = time.time()

        for i in range(config.n_trials):
            # Check timeout
            if (
                config.timeout_seconds
                and time.time() - start_time > config.timeout_seconds
            ):
                break

            # Sample random parameters
            params = create_params_fn(
                rng.uniform(*config.label_error_threshold_range),
                rng.uniform(*config.outlier_contamination_range),
                rng.uniform(*config.duplicate_threshold_range),
            )

            score = evaluate_fn(params)
            improved = state.record_trial(i, params, score)

            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if (
                config.early_stopping_rounds
                and no_improvement_count >= config.early_stopping_rounds
            ):
                break

            if config.verbose and (i + 1) % 10 == 0:
                print(f"Trial {i+1}/{config.n_trials}: best_score={state.best_score:.4f}")

        return state.best_params


class BayesianOptimizer(OptimizationStrategy):
    """Bayesian optimization strategy.

    Uses a surrogate model to guide the search toward promising regions.
    More sample-efficient than random search for expensive evaluations.
    """

    @property
    def name(self) -> str:
        return "bayesian"

    def optimize(
        self,
        config: TuningConfig,
        evaluate_fn: Callable[[Any], float],
        create_params_fn: Callable[[float, float, float], Any],
        state: OptimizationState,
        rng: np.random.RandomState,
    ) -> Any:
        """Run Bayesian optimization."""
        # Store evaluated points for surrogate model
        x_eval: list[list[float]] = []
        y_eval: list[float] = []

        # Initial random exploration
        n_initial = min(10, config.n_trials // 3)
        for i in range(n_initial):
            x = [
                rng.uniform(*config.label_error_threshold_range),
                rng.uniform(*config.outlier_contamination_range),
                rng.uniform(*config.duplicate_threshold_range),
            ]
            params = create_params_fn(x[0], x[1], x[2])
            score = evaluate_fn(params)

            x_eval.append(x)
            y_eval.append(score)
            state.record_trial(i, params, score)

        # Bayesian optimization iterations using local search around best point
        for i in range(config.n_trials - n_initial):
            # Adaptive exploration: decrease variance over time
            decay = 1 / (i + 1)
            x_best = x_eval[np.argmax(y_eval)]

            # Sample near best point with decreasing variance
            x_new = [
                np.clip(
                    x_best[0] + rng.normal(0, 0.1 * decay),
                    *config.label_error_threshold_range,
                ),
                np.clip(
                    x_best[1] + rng.normal(0, 0.05 * decay),
                    *config.outlier_contamination_range,
                ),
                np.clip(
                    x_best[2] + rng.normal(0, 0.05 * decay),
                    *config.duplicate_threshold_range,
                ),
            ]

            params = create_params_fn(x_new[0], x_new[1], x_new[2])
            score = evaluate_fn(params)

            x_eval.append(x_new)
            y_eval.append(score)
            state.record_trial(n_initial + i, params, score)

        return state.best_params


class EvolutionaryOptimizer(OptimizationStrategy):
    """Evolutionary/genetic algorithm optimization strategy.

    Uses population-based search with selection, crossover, and mutation.
    Good for complex, multi-modal search spaces.
    """

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
    ):
        """Initialize evolutionary optimizer.

        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutating each gene
            crossover_rate: Probability of crossover between parents
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    @property
    def name(self) -> str:
        return "evolutionary"

    def optimize(
        self,
        config: TuningConfig,
        evaluate_fn: Callable[[Any], float],
        create_params_fn: Callable[[float, float, float], Any],
        state: OptimizationState,
        rng: np.random.RandomState,
    ) -> Any:
        """Run evolutionary optimization."""
        ranges = [
            config.label_error_threshold_range,
            config.outlier_contamination_range,
            config.duplicate_threshold_range,
        ]

        def create_individual() -> list[float]:
            return [rng.uniform(*r) for r in ranges]

        def mutate(individual: list[float]) -> list[float]:
            result = individual.copy()
            for i in range(len(result)):
                if rng.random() < self.mutation_rate:
                    result[i] = np.clip(
                        result[i] + rng.normal(0, 0.1),
                        ranges[i][0],
                        ranges[i][1],
                    )
            return result

        def crossover(parent1: list[float], parent2: list[float]) -> list[float]:
            if rng.random() < self.crossover_rate:
                point = rng.randint(1, len(parent1))
                return parent1[:point] + parent2[point:]
            return parent1.copy()

        def evaluate_individual(individual: list[float]) -> float:
            params = create_params_fn(individual[0], individual[1], individual[2])
            return evaluate_fn(params)

        # Initialize population
        population = [create_individual() for _ in range(self.population_size)]
        fitness = [evaluate_individual(ind) for ind in population]

        # Record initial best
        best_idx = int(np.argmax(fitness))
        best_individual = population[best_idx]
        state.best_score = fitness[best_idx]
        state.best_params = create_params_fn(
            best_individual[0], best_individual[1], best_individual[2]
        )

        generations = config.n_trials // self.population_size

        for gen in range(generations):
            # Tournament selection
            new_population = []
            for _ in range(self.population_size):
                tournament = rng.choice(self.population_size, 3, replace=False)
                winner_idx = tournament[np.argmax([fitness[i] for i in tournament])]
                new_population.append(population[winner_idx].copy())

            # Crossover and mutation
            for i in range(0, len(new_population) - 1, 2):
                new_population[i] = crossover(new_population[i], new_population[i + 1])
                new_population[i] = mutate(new_population[i])
                new_population[i + 1] = mutate(new_population[i + 1])

            population = new_population
            fitness = [evaluate_individual(ind) for ind in population]

            # Update best
            gen_best_idx = int(np.argmax(fitness))
            if fitness[gen_best_idx] > state.best_score:
                state.best_score = fitness[gen_best_idx]
                best_individual = population[gen_best_idx]
                state.best_params = create_params_fn(
                    best_individual[0], best_individual[1], best_individual[2]
                )

            state.convergence_history.append(state.best_score)

            if config.verbose:
                print(f"Generation {gen+1}/{generations}: best_score={state.best_score:.4f}")

        return state.best_params


def create_optimizer(method: str) -> OptimizationStrategy:
    """Factory function to create optimization strategy.

    Args:
        method: One of 'grid', 'random', 'bayesian', 'evolutionary'

    Returns:
        Appropriate optimization strategy instance
    """
    strategies: dict[str, type[OptimizationStrategy]] = {
        "grid": GridSearchOptimizer,
        "random": RandomSearchOptimizer,
        "bayesian": BayesianOptimizer,
        "evolutionary": EvolutionaryOptimizer,
    }

    strategy_class = strategies.get(method.lower())
    if strategy_class is None:
        raise ValueError(
            f"Unknown optimization method: {method}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategy_class()
