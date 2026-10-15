"""Тесты целевой функции Леви и счётчика вычислений."""

import math

import numpy as np
import pytest

from objective import (
    DIMENSION,
    LOWER,
    OPTIMAL_VALUE,
    OPTIMUM,
    UPPER,
    EvaluationCounter,
    levy,
)


def levy_reference(x: list[float]) -> float:
    """Эталон: прямой перевод формулы из задания обычным циклом."""
    d = len(x)
    w = [1.0 + (xi - 1.0) / 4.0 for xi in x]
    total = math.sin(math.pi * w[0]) ** 2
    for i in range(d - 1):
        total += (w[i] - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * w[i] + 1.0) ** 2)
    total += (w[d - 1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[d - 1]) ** 2)
    return total


@pytest.fixture
def random_points() -> np.ndarray:
    rng = np.random.default_rng(12345)
    return rng.uniform(LOWER, UPPER, size=(100, DIMENSION))


# --- константы задачи -------------------------------------------------------

def test_problem_constants() -> None:
    assert DIMENSION == 6
    assert LOWER == -10.0
    assert UPPER == 10.0
    assert OPTIMAL_VALUE == 0.0


def test_optimum_is_read_only_and_inside_bounds() -> None:
    assert OPTIMUM.shape == (DIMENSION,)
    assert np.all(OPTIMUM >= LOWER) and np.all(OPTIMUM <= UPPER)
    with pytest.raises(ValueError):
        OPTIMUM[0] = 5.0


# --- значения функции -------------------------------------------------------

@pytest.mark.parametrize(
    "point",
    [[1, 2, 3, 4, 5, 6], (1, 2, 3, 4, 5, 6), np.arange(1.0, 7.0)],
    ids=["list", "tuple", "ndarray"],
)
def test_accepts_valid_vector_and_returns_finite_float(point) -> None:
    value = levy(point)
    assert isinstance(value, float)
    assert math.isfinite(value)


def test_value_at_optimum_is_zero() -> None:
    assert levy(OPTIMUM) == pytest.approx(OPTIMAL_VALUE, abs=1e-12)


def test_hand_computed_value_checks_sine_parentheses() -> None:
    # x = (-3, 1, 1, 1, 1, 1): w1 = 0, остальные w = 1,
    # поэтому f = 0 + (0 - 1)^2 * (1 + 10 sin^2(1)) = 1 + 10 sin^2(1).
    # Вариант sin^2(pi*w) + 1 дал бы 11, а не это число.
    x = [-3, 1, 1, 1, 1, 1]
    assert levy(x) == pytest.approx(1.0 + 10.0 * math.sin(1.0) ** 2, rel=1e-12)


def test_matches_reference_on_random_points(random_points: np.ndarray) -> None:
    for point in random_points:
        assert levy(point) == pytest.approx(levy_reference(list(point)), rel=1e-12, abs=1e-12)


def test_values_are_nonnegative(random_points: np.ndarray) -> None:
    # Все три слагаемых формулы неотрицательны.
    assert all(levy(point) >= 0.0 for point in random_points)


def test_value_is_positive_near_optimum() -> None:
    for i in range(DIMENSION):
        shifted = OPTIMUM.copy()
        shifted[i] += 0.5
        assert levy(shifted) > 0.0


@pytest.mark.parametrize(
    "corner",
    [[-10.0] * 6, [10.0] * 6, [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0]],
    ids=["all_lower", "all_upper", "mixed"],
)
def test_boundary_points(corner: list[float]) -> None:
    value = levy(corner)
    assert math.isfinite(value)
    assert value >= 0.0
    assert value == pytest.approx(levy_reference(corner), rel=1e-12)


def test_does_not_modify_input(random_points: np.ndarray) -> None:
    point = random_points[0]
    before = point.copy()
    levy(point)
    assert np.array_equal(point, before)


# --- отклонение некорректных входов -----------------------------------------

@pytest.mark.parametrize(
    "bad",
    [np.zeros(5), np.zeros(7), np.zeros((6, 1)), np.zeros((2, 6)), 1.0],
    ids=["len5", "len7", "column", "matrix", "scalar"],
)
def test_wrong_shape_raises(bad) -> None:
    with pytest.raises(ValueError):
        levy(bad)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_raises(bad_value: float) -> None:
    point = OPTIMUM.copy()
    point[2] = bad_value
    with pytest.raises(ValueError):
        levy(point)


# --- счётчик вычислений -----------------------------------------------------

def test_counter_starts_at_zero() -> None:
    assert EvaluationCounter().count == 0


def test_counter_counts_each_call() -> None:
    counter = EvaluationCounter()
    for _ in range(7):
        counter(OPTIMUM)
    assert counter.count == 7


def test_counter_ignores_failed_evaluation() -> None:
    counter = EvaluationCounter()
    counter(OPTIMUM)
    with pytest.raises(ValueError):
        counter(np.zeros(3))
    assert counter.count == 1


def test_counter_returns_function_value() -> None:
    counter = EvaluationCounter()
    x = [-3, 1, 1, 1, 1, 1]
    assert counter(x) == levy(x)
    custom = EvaluationCounter(lambda _x: 42.0)
    assert custom(OPTIMUM) == 42.0