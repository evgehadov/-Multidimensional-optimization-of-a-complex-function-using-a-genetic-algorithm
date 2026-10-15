"""Тесты генетического алгоритма."""

import numpy as np
import pytest

import genetic_algorithm
from genetic_algorithm import (
    GAConfig,
    apply_elitism,
    arithmetic_crossover,
    create_offspring,
    evaluate_population,
    gaussian_mutation,
    init_population,
    mutation_sigma,
    run_genetic_algorithm,
    tournament_select,
)
from objective import DIMENSION, LOWER, UPPER, levy


def sphere(x) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


class CountingObjective:
    """Независимый от EvaluationCounter подсчёт вызовов."""

    def __init__(self, function) -> None:
        self.function = function
        self.calls = 0

    def __call__(self, x) -> float:
        self.calls += 1
        return self.function(x)


@pytest.fixture(scope="module")
def default_run():
    objective = CountingObjective(levy)
    result = run_genetic_algorithm(GAConfig(), 202601, objective)
    return result, objective


# --- конфигурация -----------------------------------------------------------

def test_default_config_matches_experiment_a() -> None:
    config = GAConfig()
    assert config.population_size == 60
    assert config.dimension == 6
    assert (config.lower, config.upper) == (-10.0, 10.0)
    assert config.tournament_size == 3
    assert config.crossover_prob == 0.9
    assert config.effective_mutation_prob == pytest.approx(1 / 6)
    assert config.sigma == 2.0
    assert config.sigma_end is None
    assert config.budget == 6000
    assert config.n_generations == 99


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(budget=6001),
        dict(budget=30),
        dict(population_size=1, budget=2),
        dict(dimension=0),
        dict(lower=5.0, upper=5.0),
        dict(tournament_size=0),
        dict(crossover_prob=1.5),
        dict(mutation_prob=-0.1),
        dict(sigma=-1.0),
        dict(sigma_end=-0.1),
    ],
)
def test_invalid_config_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        GAConfig(**kwargs)


# --- начальная популяция ----------------------------------------------------

def test_initial_population_shape_and_bounds() -> None:
    population = init_population(np.random.default_rng(1), 60, DIMENSION, LOWER, UPPER)
    assert population.shape == (60, 6)
    assert population.min() >= LOWER
    assert population.max() <= UPPER


def test_initial_population_is_roughly_uniform() -> None:
    population = init_population(np.random.default_rng(2), 50_000, DIMENSION, LOWER, UPPER)
    assert population.mean() == pytest.approx(0.0, abs=0.1)
    assert population.std() == pytest.approx(20 / np.sqrt(12), rel=0.02)


def test_evaluate_population_returns_value_per_individual() -> None:
    population = init_population(np.random.default_rng(3), 10, DIMENSION, LOWER, UPPER)
    fitness = evaluate_population(population, levy)
    assert fitness.shape == (10,)
    assert np.array_equal(fitness, [levy(p) for p in population])


# --- турнирный отбор --------------------------------------------------------

def test_tournament_winner_is_best_of_contestants() -> None:
    fitness = np.random.default_rng(4).uniform(0.0, 100.0, size=30)
    for seed in range(200):
        winner = tournament_select(fitness, np.random.default_rng(seed), 3)
        # Повторяем ту же жеребьёвку участников вторым генератором.
        contestants = np.random.default_rng(seed).integers(0, 30, size=3)
        assert winner in contestants
        assert fitness[winner] == fitness[contestants].min()


def test_tournament_tie_goes_to_first_drawn_contestant() -> None:
    fitness = np.zeros(10)
    for seed in range(50):
        contestants = np.random.default_rng(seed).integers(0, 10, size=3)
        assert tournament_select(fitness, np.random.default_rng(seed), 3) == contestants[0]


def test_tournament_draws_with_replacement() -> None:
    # В популяции из одной особи турнир из трёх возможен только с повторами.
    assert tournament_select(np.array([5.0]), np.random.default_rng(0), 3) == 0


def test_tournament_does_not_modify_fitness() -> None:
    fitness = np.random.default_rng(5).uniform(size=20)
    before = fitness.copy()
    tournament_select(fitness, np.random.default_rng(0), 3)
    assert np.array_equal(fitness, before)


def test_tournament_favours_better_individuals() -> None:
    # n = 10, k = 3: лучший побеждает с вероятностью 1 - 0.9^3 = 0.271,
    # худший только с вероятностью 0.1^3 = 0.001.
    fitness = np.arange(10, dtype=float)
    rng = np.random.default_rng(6)
    winners = np.array([tournament_select(fitness, rng, 3) for _ in range(20_000)])
    assert 0.25 < np.mean(winners == 0) < 0.29
    assert np.mean(winners == 9) < 0.005


# --- скрещивание ------------------------------------------------------------

P1 = np.array([-8.0, -3.0, 0.0, 2.5, 7.0, 9.0])
P2 = np.array([4.0, 1.0, -6.0, 2.5, -2.0, 10.0])


def test_crossover_matches_formula() -> None:
    c1, c2 = arithmetic_crossover(P1, P2, np.random.default_rng(7), 1.0)
    reference = np.random.default_rng(7)
    reference.random()  # решение «скрещивать или нет»
    alpha = reference.random(6)
    np.testing.assert_allclose(c1, alpha * P1 + (1 - alpha) * P2)
    np.testing.assert_allclose(c2, alpha * P2 + (1 - alpha) * P1)


def test_crossover_properties() -> None:
    c1, c2 = arithmetic_crossover(P1, P2, np.random.default_rng(8), 1.0)
    np.testing.assert_allclose(c1 + c2, P1 + P2)
    low, high = np.minimum(P1, P2), np.maximum(P1, P2)
    for child in (c1, c2):
        assert np.all(child >= low - 1e-12) and np.all(child <= high + 1e-12)


@pytest.mark.parametrize("crossover_prob", [0.0, 1.0])
def test_crossover_does_not_modify_parents(crossover_prob: float) -> None:
    p1, p2 = P1.copy(), P2.copy()
    c1, c2 = arithmetic_crossover(p1, p2, np.random.default_rng(9), crossover_prob)
    c1[:] = 123.0
    c2[:] = 456.0
    assert np.array_equal(p1, P1)
    assert np.array_equal(p2, P2)


def test_no_crossover_returns_copies_of_parents() -> None:
    c1, c2 = arithmetic_crossover(P1, P2, np.random.default_rng(10), 0.0)
    assert np.array_equal(c1, P1) and np.array_equal(c2, P2)
    assert not np.shares_memory(c1, P1) and not np.shares_memory(c2, P2)


def test_crossover_probability_is_respected() -> None:
    rng = np.random.default_rng(11)
    applied = 0
    trials = 10_000
    for _ in range(trials):
        c1, _ = arithmetic_crossover(P1, P2, rng, 0.9)
        applied += not np.array_equal(c1, P1)
    assert applied / trials == pytest.approx(0.9, abs=0.02)


# --- мутация ----------------------------------------------------------------

def test_mutation_stays_in_bounds_and_clips() -> None:
    rng = np.random.default_rng(12)
    population = rng.uniform(LOWER, UPPER, size=(100, DIMENSION))
    mutated = gaussian_mutation(population, rng, 1.0, 1000.0, LOWER, UPPER)
    assert mutated.min() >= LOWER and mutated.max() <= UPPER
    # При sigma = 1000 почти всё упирается в границы: проекция реально работает.
    assert np.mean((mutated == LOWER) | (mutated == UPPER)) > 0.9


def test_mutation_works_for_single_individual_and_population() -> None:
    rng = np.random.default_rng(13)
    single = gaussian_mutation(P1, rng, 1.0, 2.0, LOWER, UPPER)
    batch = gaussian_mutation(np.tile(P1, (60, 1)), rng, 1.0, 2.0, LOWER, UPPER)
    assert single.shape == (6,)
    assert batch.shape == (60, 6)


def test_mutation_does_not_modify_input_and_returns_new_array() -> None:
    x = P1.copy()
    mutated = gaussian_mutation(x, np.random.default_rng(14), 1.0, 2.0, LOWER, UPPER)
    assert np.array_equal(x, P1)
    assert not np.shares_memory(mutated, x)


def test_mutation_with_zero_probability_or_zero_sigma_changes_nothing() -> None:
    rng = np.random.default_rng(15)
    assert np.array_equal(gaussian_mutation(P1, rng, 0.0, 2.0, LOWER, UPPER), P1)
    assert np.array_equal(gaussian_mutation(P1, rng, 1.0, 0.0, LOWER, UPPER), P1)


def test_mutation_noise_has_requested_std_and_probability() -> None:
    zeros = np.zeros(200_000)
    wide = (-1e9, 1e9)  # границы далеко, проекция не вмешивается

    noise = gaussian_mutation(zeros, np.random.default_rng(16), 1.0, 2.0, *wide)
    # sigma это стандартное отклонение: если бы это была дисперсия, получили бы 4.
    assert noise.std() == pytest.approx(2.0, rel=0.02)
    assert noise.mean() == pytest.approx(0.0, abs=0.05)

    partial = gaussian_mutation(zeros, np.random.default_rng(17), 1 / 6, 2.0, *wide)
    assert np.mean(partial != 0.0) == pytest.approx(1 / 6, abs=0.01)


# --- расписание sigma для адаптивной мутации --------------------------------

def test_sigma_is_constant_without_sigma_end() -> None:
    config = GAConfig(sigma=4.0)
    assert all(mutation_sigma(config, step, 99) == 4.0 for step in range(99))


def test_adaptive_sigma_endpoints_and_linearity() -> None:
    config = GAConfig(sigma=2.0, sigma_end=0.1)
    assert mutation_sigma(config, 0, 99) == 2.0
    assert mutation_sigma(config, 98, 99) == pytest.approx(0.1)
    assert mutation_sigma(config, 49, 99) == pytest.approx(1.05)  # середина
    values = [mutation_sigma(config, step, 99) for step in range(99)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_adaptive_sigma_with_single_generation() -> None:
    assert mutation_sigma(GAConfig(sigma=2.0, sigma_end=0.1), 0, 1) == 2.0


# --- элитизм ----------------------------------------------------------------

def test_elitism_replaces_worst_offspring_when_elite_is_better() -> None:
    offspring = np.arange(8, dtype=float).reshape(4, 2)
    fitness = np.array([3.0, 9.0, 1.0, 5.0])
    elite = np.array([-1.0, -1.0])
    offspring_before, fitness_before = offspring.copy(), fitness.copy()

    population, new_fitness = apply_elitism(offspring, fitness, elite, 0.5)

    assert np.array_equal(population[1], elite)
    assert new_fitness[1] == 0.5
    assert np.array_equal(np.delete(population, 1, axis=0), np.delete(offspring, 1, axis=0))
    assert np.array_equal(np.delete(new_fitness, 1), np.delete(fitness, 1))
    assert np.array_equal(offspring, offspring_before)
    assert np.array_equal(fitness, fitness_before)


@pytest.mark.parametrize("elite_fitness", [9.0, 10.0])
def test_elitism_keeps_offspring_if_elite_is_not_strictly_better(elite_fitness: float) -> None:
    offspring = np.arange(8, dtype=float).reshape(4, 2)
    fitness = np.array([3.0, 9.0, 1.0, 5.0])
    population, new_fitness = apply_elitism(offspring, fitness, np.array([-1.0, -1.0]), elite_fitness)
    assert np.array_equal(population, offspring)
    assert np.array_equal(new_fitness, fitness)


# --- создание потомков ------------------------------------------------------

def test_create_offspring_shape_bounds_and_parents_untouched() -> None:
    rng = np.random.default_rng(18)
    config = GAConfig()
    population = init_population(rng, 60, DIMENSION, LOWER, UPPER)
    fitness = evaluate_population(population, levy)
    population_before = population.copy()

    offspring = create_offspring(population, fitness, rng, config, config.sigma)

    assert offspring.shape == (60, 6)
    assert offspring.min() >= LOWER and offspring.max() <= UPPER
    assert np.array_equal(population, population_before)
    assert not np.shares_memory(offspring, population)


def test_create_offspring_supports_odd_population_size() -> None:
    rng = np.random.default_rng(19)
    config = GAConfig(population_size=5, budget=20)
    population = init_population(rng, 5, DIMENSION, LOWER, UPPER)
    fitness = evaluate_population(population, levy)
    assert create_offspring(population, fitness, rng, config, 2.0).shape == (5, 6)


# --- полный запуск: бюджет --------------------------------------------------

def test_default_budget_is_exactly_6000(default_run) -> None:
    result, objective = default_run
    assert objective.calls == 6000
    assert result.evaluations == 6000


@pytest.mark.parametrize(
    ("population_size", "generations"),
    [(4, 0), (5, 3), (10, 7), (30, 4)],
)
def test_budget_is_exact_for_small_configs(population_size: int, generations: int) -> None:
    config = GAConfig(population_size=population_size, budget=population_size * (1 + generations))
    objective = CountingObjective(levy)
    result = run_genetic_algorithm(config, 1, objective)
    assert objective.calls == config.budget
    assert result.evaluations == config.budget


def test_adaptive_config_also_keeps_budget() -> None:
    config = GAConfig(population_size=10, budget=10 * 11, sigma=2.0, sigma_end=0.1)
    objective = CountingObjective(levy)
    run_genetic_algorithm(config, 1, objective)
    assert objective.calls == config.budget


# --- полный запуск: история и результат -------------------------------------

def test_history_structure(default_run) -> None:
    result, _ = default_run
    h = result.history
    assert len(h.generation) == 100
    assert np.array_equal(h.generation, np.arange(100))
    assert np.array_equal(h.evaluations, 60 * (np.arange(100) + 1))
    for series in (h.best, h.mean, h.worst, h.best_so_far):
        assert series.shape == (100,)
        assert np.all(np.isfinite(series))


def test_history_values_are_consistent(default_run) -> None:
    result, _ = default_run
    h = result.history
    assert np.all(h.best <= h.mean) and np.all(h.mean <= h.worst)
    assert np.all(np.diff(h.best_so_far) <= 0.0)
    assert h.best_so_far[-1] == result.best_f


def test_elitism_makes_population_best_equal_to_best_so_far(default_run) -> None:
    # Лучшее в популяции никогда не ухудшается и совпадает с глобальным рекордом.
    result, _ = default_run
    h = result.history
    assert np.all(np.diff(h.best) <= 0.0)
    assert np.array_equal(h.best, h.best_so_far)


def test_result_is_consistent(default_run) -> None:
    result, _ = default_run
    assert result.best_x.shape == (6,)
    assert np.all(result.best_x >= LOWER) and np.all(result.best_x <= UPPER)
    assert levy(result.best_x) == result.best_f
    assert result.seed == 202601
    assert result.config == GAConfig()


# --- воспроизводимость ------------------------------------------------------

def test_same_seed_reproduces_result_exactly(default_run) -> None:
    result, _ = default_run
    again = run_genetic_algorithm(GAConfig(), 202601)
    assert again.best_f == result.best_f
    assert np.array_equal(again.best_x, result.best_x)
    assert np.array_equal(again.history.mean, result.history.mean)
    assert np.array_equal(again.history.worst, result.history.worst)


def test_different_seed_gives_different_scenario(default_run) -> None:
    result, _ = default_run
    other = run_genetic_algorithm(GAConfig(), 202602)
    # Совпадение теоретически возможно, но на практике означало бы баг с seed.
    assert other.history.mean[0] != result.history.mean[0]
    assert not np.array_equal(other.history.mean, result.history.mean)


def test_result_does_not_depend_on_global_numpy_state() -> None:
    config = GAConfig(population_size=10, budget=10 * 6)
    np.random.seed(0)
    first = run_genetic_algorithm(config, 5)
    np.random.seed(999)
    second = run_genetic_algorithm(config, 5)
    assert first.best_f == second.best_f
    assert np.array_equal(first.history.mean, second.history.mean)


# --- поведение алгоритма ----------------------------------------------------

def test_ga_improves_on_sphere_function() -> None:
    config = GAConfig(population_size=20, dimension=2, budget=20 * 31, sigma=1.0)
    result = run_genetic_algorithm(config, 3, sphere)
    assert result.best_f < 0.5 * result.history.best_so_far[0]


def test_ga_module_does_not_use_known_optimum() -> None:
    assert not hasattr(genetic_algorithm, "OPTIMUM")
    assert not hasattr(genetic_algorithm, "OPTIMAL_VALUE")