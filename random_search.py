"""Случайный поиск: базовый метод для сравнения с генетическим алгоритмом.

Кандидаты независимы и равномерны на кубе [lower, upper]^d. Они
генерируются до первого вычисления целевой функции, поэтому значения
функции никак не влияют на новые точки.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from objective import DIMENSION, LOWER, UPPER, EvaluationCounter, levy


@dataclass(frozen=True)
class RandomSearchConfig:
    """Параметры случайного поиска. Значения по умолчанию соответствуют эксперименту C."""

    dimension: int = DIMENSION
    lower: float = LOWER
    upper: float = UPPER
    budget: int = 6000

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension должен быть не меньше 1")
        if not self.lower < self.upper:
            raise ValueError("Требуется lower < upper")
        if self.budget < 1:
            raise ValueError("budget должен быть не меньше 1")


@dataclass(frozen=True)
class RandomSearchHistory:
    """История запуска: запись на каждое вычисление функции.

    evaluations = 1..budget, best_so_far[i] это лучшее значение среди
    первых evaluations[i] кандидатов.
    """

    evaluations: NDArray[np.int64]
    best_so_far: NDArray[np.float64]


@dataclass(frozen=True)
class RandomSearchResult:
    """Итог одного запуска."""

    best_x: NDArray[np.float64]
    best_f: float
    evaluations: int
    history: RandomSearchHistory
    seed: int
    config: RandomSearchConfig


def sample_candidates(
    rng: np.random.Generator, count: int, dimension: int, lower: float, upper: float
) -> NDArray[np.float64]:
    """count независимых точек, каждая координата ~ U(lower, upper)."""
    return rng.uniform(lower, upper, size=(count, dimension))


def run_random_search(
    config: RandomSearchConfig,
    seed: int,
    objective: Callable[[ArrayLike], float] = levy,
) -> RandomSearchResult:
    """Один независимый запуск со своим генератором default_rng(seed).

    Все вычисления целевой функции идут через EvaluationCounter, по нему
    в конце проверяется, что бюджет израсходован ровно.
    """
    rng = np.random.default_rng(seed)
    counter = EvaluationCounter(objective)

    candidates = sample_candidates(
        rng, config.budget, config.dimension, config.lower, config.upper
    )
    values = np.array([counter(candidate) for candidate in candidates], dtype=np.float64)

    if counter.count != config.budget:
        raise RuntimeError(
            f"Израсходовано {counter.count} вычислений вместо {config.budget}"
        )

    # При равных значениях argmin берёт первого кандидата.
    best_index = int(np.argmin(values))
    history = RandomSearchHistory(
        evaluations=np.arange(1, config.budget + 1, dtype=np.int64),
        best_so_far=np.minimum.accumulate(values),
    )
    return RandomSearchResult(
        best_x=candidates[best_index].copy(),
        best_f=float(values[best_index]),
        evaluations=counter.count,
        history=history,
        seed=seed,
        config=config,
    )