"""Генетический алгоритм с вещественным кодированием (минимизация).

Один запуск: run_genetic_algorithm(config, seed). Операторы вынесены в
отдельные функции, чтобы их можно было проверять независимо.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from objective import DIMENSION, LOWER, UPPER, EvaluationCounter, levy


@dataclass(frozen=True)
class GAConfig:
    """Параметры ГА. Значения по умолчанию соответствуют эксперименту A.

    mutation_prob=None означает 1/dimension.
    sigma_end=None означает постоянный масштаб мутации sigma.
    Если sigma_end задан, sigma линейно уменьшается от sigma (первое
    поколение потомков) до sigma_end (последнее), см. mutation_sigma.
    """

    population_size: int = 60
    dimension: int = DIMENSION
    lower: float = LOWER
    upper: float = UPPER
    tournament_size: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float | None = None
    sigma: float = 2.0
    sigma_end: float | None = None
    budget: int = 6000

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size должен быть не меньше 2")
        if self.dimension < 1:
            raise ValueError("dimension должен быть не меньше 1")
        if not self.lower < self.upper:
            raise ValueError("Требуется lower < upper")
        if self.tournament_size < 1:
            raise ValueError("tournament_size должен быть не меньше 1")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError("crossover_prob должна лежать в [0, 1]")
        if not 0.0 <= self.effective_mutation_prob <= 1.0:
            raise ValueError("mutation_prob должна лежать в [0, 1]")
        if self.sigma < 0.0 or (self.sigma_end is not None and self.sigma_end < 0.0):
            raise ValueError("sigma и sigma_end не могут быть отрицательными")
        if self.budget < self.population_size:
            raise ValueError("budget меньше размера популяции")
        if (self.budget - self.population_size) % self.population_size != 0:
            raise ValueError(
                "budget должен равняться population_size * (1 + число поколений): "
                "иначе точный учёт бюджета невозможен"
            )

    @property
    def effective_mutation_prob(self) -> float:
        """Вероятность мутации одной координаты."""
        if self.mutation_prob is None:
            return 1.0 / self.dimension
        return self.mutation_prob

    @property
    def n_generations(self) -> int:
        """Число поколений потомков (начальная популяция не считается)."""
        return (self.budget - self.population_size) // self.population_size


@dataclass(frozen=True)
class GAHistory:
    """История запуска: по одной записи на начальную популяцию и на каждое поколение.

    Запись с индексом 0 описывает начальную популяцию (generation = 0).
    best, mean, worst относятся к текущей популяции (после элитизма),
    best_so_far это лучшее значение за весь запуск к этому моменту.
    """

    generation: NDArray[np.int64]
    evaluations: NDArray[np.int64]
    best: NDArray[np.float64]
    mean: NDArray[np.float64]
    worst: NDArray[np.float64]
    best_so_far: NDArray[np.float64]


@dataclass(frozen=True)
class GAResult:
    """Итог одного запуска."""

    best_x: NDArray[np.float64]
    best_f: float
    evaluations: int
    history: GAHistory
    seed: int
    config: GAConfig


def mutation_sigma(config: GAConfig, step: int, n_steps: int) -> float:
    """Масштаб мутации (стандартное отклонение) для поколения потомков.

    step = 0..n_steps-1 (номер поколения потомков минус один), значение
    определяется в начале поколения. Это формула
    sigma(t) = sigma_max - t/T * (sigma_max - sigma_min) при T = n_steps - 1:
    первое поколение получает sigma, последнее sigma_end.
    """
    if config.sigma_end is None or n_steps <= 1:
        return config.sigma
    return config.sigma + (config.sigma_end - config.sigma) * step / (n_steps - 1)


def init_population(
    rng: np.random.Generator, size: int, dimension: int, lower: float, upper: float
) -> NDArray[np.float64]:
    """Начальная популяция: координаты независимы и равномерны на [lower, upper)."""
    return rng.uniform(lower, upper, size=(size, dimension))


def evaluate_population(
    population: NDArray[np.float64], objective: Callable[[ArrayLike], float]
) -> NDArray[np.float64]:
    """Значения целевой функции для каждой особи (по одному вызову на особь)."""
    return np.array([objective(individual) for individual in population], dtype=np.float64)


def tournament_select(
    fitness: NDArray[np.float64], rng: np.random.Generator, tournament_size: int
) -> int:
    """Индекс победителя турнира (минимизация).

    Участники выбираются с возвращением, поэтому одна особь может попасть в
    турнир несколько раз. При равенстве значений побеждает участник, который
    был выбран раньше (np.argmin возвращает первый минимум), результат
    детерминирован при фиксированной последовательности случайных чисел.
    Популяция не изменяется: работаем только с индексами.
    """
    contestants = rng.integers(0, fitness.shape[0], size=tournament_size)
    return int(contestants[np.argmin(fitness[contestants])])


def arithmetic_crossover(
    parent1: NDArray[np.float64],
    parent2: NDArray[np.float64],
    rng: np.random.Generator,
    crossover_prob: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Арифметическое скрещивание с покоординатными alpha_i ~ U(0, 1).

    С вероятностью crossover_prob создаются два потомка по формулам задания,
    иначе потомки это копии родителей. Родители не изменяются. Выпуклая
    комбинация точек из допустимой области остаётся в этой области.
    """
    if rng.random() < crossover_prob:
        alpha = rng.random(parent1.shape[0])
        child1 = alpha * parent1 + (1.0 - alpha) * parent2
        child2 = alpha * parent2 + (1.0 - alpha) * parent1
        return child1, child2
    return parent1.copy(), parent2.copy()


def gaussian_mutation(
    x: NDArray[np.float64],
    rng: np.random.Generator,
    mutation_prob: float,
    sigma: float,
    lower: float,
    upper: float,
) -> NDArray[np.float64]:
    """Гауссовская мутация с проекцией на границы.

    Работает и для одной особи (форма (d,)), и для массива особей (N, d).
    Каждая координата мутирует независимо с вероятностью mutation_prob:
    к ней прибавляется N(0, sigma^2), где sigma это стандартное отклонение.
    Возвращается новый массив, входной не изменяется.
    """
    mask = rng.random(x.shape) < mutation_prob
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape)
    return np.clip(np.where(mask, x + noise, x), lower, upper)


def create_offspring(
    population: NDArray[np.float64],
    fitness: NDArray[np.float64],
    rng: np.random.Generator,
    config: GAConfig,
    sigma: float,
) -> NDArray[np.float64]:
    """Новое поколение из ровно population_size потомков (до оценки).

    Потомки создаются парами. Если размер популяции нечётный, последний
    лишний потомок отбрасывается. Один и тот же родитель может быть выбран
    для обоих мест пары, это допустимо.
    """
    size, dimension = population.shape
    n_pairs = (size + 1) // 2
    children = np.empty((2 * n_pairs, dimension), dtype=np.float64)
    for pair in range(n_pairs):
        i = tournament_select(fitness, rng, config.tournament_size)
        j = tournament_select(fitness, rng, config.tournament_size)
        child1, child2 = arithmetic_crossover(
            population[i], population[j], rng, config.crossover_prob
        )
        children[2 * pair] = child1
        children[2 * pair + 1] = child2
    return gaussian_mutation(
        children[:size],
        rng,
        config.effective_mutation_prob,
        sigma,
        config.lower,
        config.upper,
    )


def apply_elitism(
    offspring: NDArray[np.float64],
    offspring_fitness: NDArray[np.float64],
    elite: NDArray[np.float64],
    elite_fitness: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Заменяет худшего потомка элитной особью, если элита строго лучше его.

    Значение целевой функции элиты уже известно и повторно не вычисляется.
    При равенстве значений худшим считается первый по порядку. Возвращает
    новые массивы, входные не изменяются.
    """
    population = offspring.copy()
    fitness = offspring_fitness.copy()
    worst = int(np.argmax(fitness))
    if elite_fitness < fitness[worst]:
        population[worst] = elite
        fitness[worst] = elite_fitness
    return population, fitness


def run_genetic_algorithm(
    config: GAConfig,
    seed: int,
    objective: Callable[[ArrayLike], float] = levy,
) -> GAResult:
    """Один независимый запуск ГА со своим генератором default_rng(seed).

    Все вычисления целевой функции идут через EvaluationCounter, по нему
    в конце проверяется, что бюджет израсходован ровно.
    """
    rng = np.random.default_rng(seed)
    counter = EvaluationCounter(objective)
    n_generations = config.n_generations

    population = init_population(
        rng, config.population_size, config.dimension, config.lower, config.upper
    )
    fitness = evaluate_population(population, counter)

    best_index = int(np.argmin(fitness))
    best_x = population[best_index].copy()
    best_f = float(fitness[best_index])

    records: dict[str, list[float]] = {
        "generation": [], "evaluations": [], "best": [],
        "mean": [], "worst": [], "best_so_far": [],
    }

    def record(generation: int) -> None:
        records["generation"].append(generation)
        records["evaluations"].append(counter.count)
        records["best"].append(float(fitness.min()))
        records["mean"].append(float(fitness.mean()))
        records["worst"].append(float(fitness.max()))
        records["best_so_far"].append(best_f)

    record(0)

    for step in range(n_generations):
        sigma = mutation_sigma(config, step, n_generations)

        elite_index = int(np.argmin(fitness))
        elite = population[elite_index].copy()
        elite_fitness = float(fitness[elite_index])

        offspring = create_offspring(population, fitness, rng, config, sigma)
        offspring_fitness = evaluate_population(offspring, counter)

        # Глобальный рекорд обновляем по всем оценённым потомкам, независимо
        # от того, что останется в популяции после элитизма.
        child_best = int(np.argmin(offspring_fitness))
        if offspring_fitness[child_best] < best_f:
            best_f = float(offspring_fitness[child_best])
            best_x = offspring[child_best].copy()

        population, fitness = apply_elitism(offspring, offspring_fitness, elite, elite_fitness)
        record(step + 1)

    if counter.count != config.budget:
        raise RuntimeError(
            f"Израсходовано {counter.count} вычислений вместо {config.budget}"
        )

    history = GAHistory(
        generation=np.array(records["generation"], dtype=np.int64),
        evaluations=np.array(records["evaluations"], dtype=np.int64),
        best=np.array(records["best"], dtype=np.float64),
        mean=np.array(records["mean"], dtype=np.float64),
        worst=np.array(records["worst"], dtype=np.float64),
        best_so_far=np.array(records["best_so_far"], dtype=np.float64),
    )
    return GAResult(
        best_x=best_x,
        best_f=best_f,
        evaluations=counter.count,
        history=history,
        seed=seed,
        config=config,
    )