"""Организация независимых запусков и сбор сырых результатов.

Каждый запуск получает seed = base_seed + номер запуска и создаёт собственный
генератор внутри run_genetic_algorithm / run_random_search. Общего
изменяемого генератора между запусками и методами нет.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from experiment_statistics import MethodSummary, summarize_method, write_summary_csv
from genetic_algorithm import GAConfig, GAHistory, run_genetic_algorithm
from objective import LOWER, UPPER, levy
from random_search import RandomSearchConfig, RandomSearchHistory, run_random_search

BASE_SEED = 202600  # seed запуска i равен BASE_SEED + i, i = 1..N_RUNS
N_RUNS = 20

RAW_RUNS_FILE = "raw_runs.csv"
HISTORIES_FILE = "histories.npz"
SUMMARY_FILE = "summary.csv"

Config = GAConfig | RandomSearchConfig
History = GAHistory | RandomSearchHistory

_PARAMETER_COLUMNS = (
    "dimension", "lower", "upper", "budget", "population_size", "tournament_size",
    "crossover_prob", "mutation_prob", "sigma", "sigma_end",
)
_SHARED_HISTORY_FIELDS = ("generation", "evaluations")
_KEY_SEPARATOR = "__"


@dataclass(frozen=True)
class ExperimentSpec:
    """Описание эксперимента: имя, пояснение и параметры метода."""

    name: str
    description: str
    config: Config

    def __post_init__(self) -> None:
        if not self.name or _KEY_SEPARATOR in self.name:
            raise ValueError(f"Недопустимое имя эксперимента: {self.name!r}")

    @property
    def method(self) -> str:
        return "genetic_algorithm" if isinstance(self.config, GAConfig) else "random_search"


@dataclass(frozen=True)
class RunRecord:
    """Сырой результат одного запуска: всё нужное для восстановления запуска."""

    experiment: str
    method: str
    run: int
    seed: int
    best_f: float
    best_x: NDArray[np.float64]
    evaluations: int
    time_seconds: float
    config: Config
    history: History


def build_experiments(
    population_size: int = 60, budget: int = 6000
) -> dict[str, ExperimentSpec]:
    """Эксперименты A, B, C, D.

    B и D отличаются от A одним параметром (sigma и sigma_end соответственно).
    Размер популяции и бюджет можно уменьшить для быстрых проверок.
    """
    span = UPPER - LOWER
    base = GAConfig(population_size=population_size, budget=budget, sigma=0.1 * span)
    return {
        "A": ExperimentSpec("A", "ГА, масштаб мутации 0.1(u-l) = 2", base),
        "B": ExperimentSpec("B", "ГА, масштаб мутации 0.2(u-l) = 4", replace(base, sigma=0.2 * span)),
        "C": ExperimentSpec("C", "Случайный поиск", RandomSearchConfig(budget=budget)),
        "D": ExperimentSpec(
            "D", "ГА, масштаб мутации линейно от 2 до 0.1", replace(base, sigma_end=0.1)
        ),
    }


def run_experiment(
    spec: ExperimentSpec,
    n_runs: int = N_RUNS,
    base_seed: int = BASE_SEED,
    objective: Callable[[ArrayLike], float] = levy,
    on_run_finished: Callable[[RunRecord], None] | None = None,
) -> list[RunRecord]:
    """n_runs независимых запусков одного эксперимента с seed = base_seed + run."""
    if n_runs < 1:
        raise ValueError("n_runs должен быть не меньше 1")

    records: list[RunRecord] = []
    for run in range(1, n_runs + 1):
        seed = base_seed + run
        started = time.perf_counter()
        if isinstance(spec.config, GAConfig):
            result = run_genetic_algorithm(spec.config, seed, objective)
        else:
            result = run_random_search(spec.config, seed, objective)
        elapsed = time.perf_counter() - started

        record = RunRecord(
            experiment=spec.name,
            method=spec.method,
            run=run,
            seed=seed,
            best_f=result.best_f,
            best_x=result.best_x,
            evaluations=result.evaluations,
            time_seconds=elapsed,
            config=spec.config,
            history=result.history,
        )
        records.append(record)
        if on_run_finished is not None:
            on_run_finished(record)
    return records


def run_all(
    specs: Mapping[str, ExperimentSpec],
    n_runs: int = N_RUNS,
    base_seed: int = BASE_SEED,
    objective: Callable[[ArrayLike], float] = levy,
    on_run_finished: Callable[[RunRecord], None] | None = None,
) -> dict[str, list[RunRecord]]:
    """Все эксперименты по очереди. Результат одного не влияет на другой."""
    return {
        name: run_experiment(spec, n_runs, base_seed, objective, on_run_finished)
        for name, spec in specs.items()
    }


def summarize_records(
    records: Sequence[RunRecord], success_threshold: float | None = None
) -> MethodSummary:
    """Сводка по запускам одного эксперимента (расчёт в experiment_statistics)."""
    if not records:
        raise ValueError("Нет запусков для агрегирования")
    return summarize_method(
        method=records[0].experiment,
        final_values=[r.best_f for r in records],
        evaluations=[r.evaluations for r in records],
        times_seconds=[r.time_seconds for r in records],
        success_threshold=success_threshold,
    )


def _parameters(config: Config) -> dict[str, float | int | None]:
    values: dict[str, float | int | None] = {
        "dimension": config.dimension,
        "lower": config.lower,
        "upper": config.upper,
        "budget": config.budget,
    }
    if isinstance(config, GAConfig):
        values.update(
            population_size=config.population_size,
            tournament_size=config.tournament_size,
            crossover_prob=config.crossover_prob,
            mutation_prob=config.effective_mutation_prob,
            sigma=config.sigma,
            sigma_end=config.sigma_end,
        )
    return values


def write_raw_runs_csv(
    records_by_experiment: Mapping[str, Sequence[RunRecord]], path: str | Path
) -> Path:
    """Сырые результаты всех запусков, одна строка на запуск, без округления.

    Пустая ячейка означает, что параметр к методу не относится (или None).
    """
    all_records = [r for records in records_by_experiment.values() for r in records]
    if not all_records:
        raise ValueError("Нечего сохранять: нет запусков")
    dimensions = {r.best_x.shape[0] for r in all_records}
    if len(dimensions) != 1:
        raise ValueError("Размерности найденных векторов различаются")
    coordinates = [f"x{i}" for i in range(1, dimensions.pop() + 1)]
    fieldnames = [
        "experiment", "method", "run", "seed", "best_f", *coordinates,
        "evaluations", "time_seconds", *_PARAMETER_COLUMNS,
    ]

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            row: dict[str, object] = {
                "experiment": r.experiment, "method": r.method, "run": r.run,
                "seed": r.seed, "best_f": float(r.best_f),
                "evaluations": r.evaluations, "time_seconds": float(r.time_seconds),
            }
            row.update({name: float(v) for name, v in zip(coordinates, r.best_x)})
            parameters = _parameters(r.config)
            row.update({name: parameters.get(name) for name in _PARAMETER_COLUMNS})
            writer.writerow(row)
    return target


def histories_to_arrays(records: Sequence[RunRecord]) -> dict[str, NDArray]:
    """Истории запусков одного эксперимента в виде массивов.

    Общие для всех запусков оси (generation, evaluations) хранятся один раз,
    остальные ряды складываются в матрицу (запуск, шаг). Если длины или оси
    различаются, это ошибка: сравнение методов было бы некорректным.
    """
    if not records:
        raise ValueError("Нет запусков для сохранения истории")
    arrays: dict[str, NDArray] = {}
    for field in fields(records[0].history):
        series = [getattr(r.history, field.name) for r in records]
        if field.name in _SHARED_HISTORY_FIELDS:
            if any(not np.array_equal(series[0], other) for other in series[1:]):
                raise ValueError(f"История {field.name} различается между запусками")
            arrays[field.name] = series[0]
        else:
            if len({s.shape for s in series}) != 1:
                raise ValueError(f"Длины истории {field.name} различаются между запусками")
            arrays[field.name] = np.stack(series)
    arrays["seeds"] = np.array([r.seed for r in records], dtype=np.int64)
    return arrays


def save_histories(
    records_by_experiment: Mapping[str, Sequence[RunRecord]], path: str | Path
) -> Path:
    """Сохраняет истории всех экспериментов в один .npz (ключи вида A__best)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"{name}{_KEY_SEPARATOR}{key}": value
        for name, records in records_by_experiment.items()
        for key, value in histories_to_arrays(records).items()
    }
    np.savez_compressed(target, **payload)
    return target


def load_histories(path: str | Path) -> dict[str, dict[str, NDArray]]:
    """Загружает истории: {эксперимент: {ряд: массив}}."""
    result: dict[str, dict[str, NDArray]] = {}
    with np.load(Path(path)) as data:
        for key in data.files:
            name, field = key.split(_KEY_SEPARATOR, 1)
            result.setdefault(name, {})[field] = data[key]
    return result


def save_results(
    records_by_experiment: Mapping[str, Sequence[RunRecord]],
    results_dir: str | Path,
    success_threshold: float | None = None,
) -> dict[str, Path]:
    """Сохраняет сырые результаты, истории и сводную таблицу в results_dir."""
    folder = Path(results_dir)
    summaries = [
        summarize_records(records, success_threshold)
        for records in records_by_experiment.values()
    ]
    return {
        "raw_runs": write_raw_runs_csv(records_by_experiment, folder / RAW_RUNS_FILE),
        "histories": save_histories(records_by_experiment, folder / HISTORIES_FILE),
        "summary": write_summary_csv(summaries, folder / SUMMARY_FILE),
    }