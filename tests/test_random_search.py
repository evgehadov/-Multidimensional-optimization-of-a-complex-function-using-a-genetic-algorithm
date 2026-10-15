"""Тесты случайного поиска."""

import numpy as np
import pytest

import random_search
from objective import DIMENSION, LOWER, UPPER, levy
from random_search import (
    RandomSearchConfig,
    run_random_search,
    sample_candidates,
)


def sphere(x) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


class RecordingObjective:
    """Запоминает все точки и значения, которые получила целевая функция."""

    def __init__(self, function) -> None:
        self.function = function
        self.points: list[np.ndarray] = []
        self.values: list[float] = []

    @property
    def calls(self) -> int:
        return len(self.points)

    def __call__(self, x) -> float:
        value = self.function(x)
        self.points.append(np.array(x, dtype=np.float64))
        self.values.append(value)
        return value


@pytest.fixture(scope="module")
def default_run():
    objective = RecordingObjective(levy)
    result = run_random_search(RandomSearchConfig(), 202601, objective)
    return result, objective


# --- конфигурация -----------------------------------------------------------

def test_default_config_matches_experiment_c() -> None:
    config = RandomSearchConfig()
    assert config.dimension == 6
    assert (config.lower, config.upper) == (-10.0, 10.0)
    assert config.budget == 6000


@pytest.mark.parametrize(
    "kwargs",
    [dict(dimension=0), dict(lower=3.0, upper=3.0), dict(lower=5.0, upper=-5.0), dict(budget=0)],
)
def test_invalid_config_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RandomSearchConfig(**kwargs)


# --- генерация кандидатов ---------------------------------------------------

def test_sample_candidates_shape_and_bounds() -> None:
    points = sample_candidates(np.random.default_rng(1), 1000, DIMENSION, LOWER, UPPER)
    assert points.shape == (1000, 6)
    assert points.min() >= LOWER
    assert points.max() <= UPPER


def test_sample_candidates_is_roughly_uniform() -> None:
    points = sample_candidates(np.random.default_rng(2), 100_000, DIMENSION, LOWER, UPPER)
    assert points.mean() == pytest.approx(0.0, abs=0.1)
    # Стандартное отклонение U(-10, 10) равно 20 / sqrt(12).
    assert points.std() == pytest.approx(20 / np.sqrt(12), rel=0.02)
    # Координаты независимы: корреляции между столбцами близки к нулю.
    correlations = np.corrcoef(points, rowvar=False)
    off_diagonal = correlations[~np.eye(DIMENSION, dtype=bool)]
    assert np.all(np.abs(off_diagonal) < 0.02)


# --- допустимость и независимость кандидатов --------------------------------

def test_all_evaluated_points_are_feasible(default_run) -> None:
    _, objective = default_run
    points = np.array(objective.points)
    assert points.shape == (6000, 6)
    assert points.min() >= LOWER and points.max() <= UPPER


def test_evaluated_points_match_independent_sampling(default_run) -> None:
    _, objective = default_run
    expected = sample_candidates(np.random.default_rng(202601), 6000, DIMENSION, LOWER, UPPER)
    assert np.array_equal(np.array(objective.points), expected)


@pytest.mark.parametrize(
    "other_function",
    [lambda x: -levy(x), lambda x: 0.0],
    ids=["negated", "constant"],
)
def test_candidates_do_not_depend_on_objective_values(other_function) -> None:
    config = RandomSearchConfig(budget=500)
    reference = RecordingObjective(levy)
    other = RecordingObjective(other_function)
    run_random_search(config, 7, reference)
    run_random_search(config, 7, other)
    assert np.array_equal(np.array(reference.points), np.array(other.points))


# --- бюджет -----------------------------------------------------------------

def test_default_budget_is_exactly_6000(default_run) -> None:
    result, objective = default_run
    assert objective.calls == 6000
    assert result.evaluations == 6000


@pytest.mark.parametrize("budget", [1, 2, 37, 600])
def test_budget_is_exact_for_small_budgets(budget: int) -> None:
    objective = RecordingObjective(levy)
    result = run_random_search(RandomSearchConfig(budget=budget), 3, objective)
    assert objective.calls == budget
    assert result.evaluations == budget


# --- лучший результат -------------------------------------------------------

def test_best_is_minimum_over_all_candidates(default_run) -> None:
    result, objective = default_run
    values = np.array(objective.values)
    assert result.best_f == values.min()
    assert result.best_f == values[int(np.argmin(values))]
    assert any(np.array_equal(result.best_x, point) for point in objective.points)
    assert levy(result.best_x) == result.best_f


def test_best_x_is_a_copy_not_a_view(default_run) -> None:
    result, objective = default_run
    original = result.best_x.copy()
    for point in objective.points:
        assert not np.shares_memory(result.best_x, point)
    assert np.array_equal(result.best_x, original)


def test_tie_goes_to_first_candidate() -> None:
    objective = RecordingObjective(lambda _x: 1.0)
    result = run_random_search(RandomSearchConfig(budget=20), 5, objective)
    assert result.best_f == 1.0
    assert np.array_equal(result.best_x, objective.points[0])


def test_works_with_custom_dimension_and_bounds() -> None:
    config = RandomSearchConfig(dimension=2, lower=-1.0, upper=3.0, budget=200)
    result = run_random_search(config, 11, sphere)
    assert result.best_x.shape == (2,)
    assert np.all(result.best_x >= -1.0) and np.all(result.best_x <= 3.0)


# --- история ----------------------------------------------------------------

def test_history_structure(default_run) -> None:
    result, _ = default_run
    h = result.history
    assert np.array_equal(h.evaluations, np.arange(1, 6001))
    assert h.best_so_far.shape == (6000,)
    assert np.all(np.isfinite(h.best_so_far))


def test_history_is_running_minimum(default_run) -> None:
    result, objective = default_run
    h = result.history
    values = np.array(objective.values)
    assert h.best_so_far[0] == values[0]
    assert h.best_so_far[-1] == result.best_f
    assert np.all(np.diff(h.best_so_far) <= 0.0)
    for count in (1, 10, 60, 1234, 6000):
        assert h.best_so_far[count - 1] == values[:count].min()


# --- воспроизводимость ------------------------------------------------------

def test_same_seed_reproduces_result_exactly(default_run) -> None:
    result, _ = default_run
    again = run_random_search(RandomSearchConfig(), 202601)
    assert again.best_f == result.best_f
    assert np.array_equal(again.best_x, result.best_x)
    assert np.array_equal(again.history.best_so_far, result.history.best_so_far)


def test_different_seed_gives_different_scenario(default_run) -> None:
    result, _ = default_run
    other = run_random_search(RandomSearchConfig(), 202602)
    # Совпадение теоретически возможно, но означало бы баг с seed.
    assert not np.array_equal(other.best_x, result.best_x)
    assert not np.array_equal(other.history.best_so_far, result.history.best_so_far)


def test_result_does_not_depend_on_global_numpy_state() -> None:
    config = RandomSearchConfig(budget=100)
    np.random.seed(0)
    first = run_random_search(config, 5)
    np.random.seed(999)
    second = run_random_search(config, 5)
    assert first.best_f == second.best_f
    assert np.array_equal(first.best_x, second.best_x)


def test_result_records_seed_and_config(default_run) -> None:
    result, _ = default_run
    assert result.seed == 202601
    assert result.config == RandomSearchConfig()


# --- изоляция от других методов ---------------------------------------------

def test_module_does_not_use_known_optimum_or_ga_operators() -> None:
    assert not hasattr(random_search, "OPTIMUM")
    assert not hasattr(random_search, "OPTIMAL_VALUE")
    for name in ("tournament_select", "arithmetic_crossover", "gaussian_mutation"):
        assert not hasattr(random_search, name)