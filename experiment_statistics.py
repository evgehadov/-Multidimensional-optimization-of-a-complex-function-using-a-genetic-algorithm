"""Агрегирование результатов независимых запусков одного метода.

Модуль работает с обычными массивами чисел и не знает, как они получены.
Привязка к запускам (seed, параметры, найденные векторы) хранится в сырых
результатах, а не здесь.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Выборочное стандартное отклонение: делитель n - 1.
STD_DDOF = 1


@dataclass(frozen=True)
class MethodSummary:
    """Сводка по n независимым запускам одного метода.

    std это выборочное стандартное отклонение (делитель n - 1); при одном
    запуске оно не определено и равно NaN. Если порог успеха не задан,
    success_threshold и success_rate равны None.
    """

    method: str
    n_runs: int
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float
    evaluations_per_run: int
    mean_time_seconds: float
    total_time_seconds: float
    success_threshold: float | None
    success_rate: float | None


def _as_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name}: нужен непустой одномерный массив")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: найдены NaN или inf")
    return array


def summarize_method(
    method: str,
    final_values: ArrayLike,
    evaluations: ArrayLike,
    times_seconds: ArrayLike,
    success_threshold: float | None = None,
) -> MethodSummary:
    """Статистика по итоговым значениям целевой функции всех запусков метода.

    Args:
        method: название метода.
        final_values: лучшее найденное значение в каждом запуске.
        evaluations: фактическое число вычислений функции в каждом запуске.
            Во всех запусках оно должно быть одинаковым, иначе сравнение
            методов нечестное.
        times_seconds: время выполнения каждого запуска.
        success_threshold: порог успеха. Запуск успешен, если его итоговое
            значение не больше порога. Значения по умолчанию нет: порог
            нужно обосновать и задать явно.

    Raises:
        ValueError: при пустых данных, NaN/inf, разной длине массивов,
            отрицательном времени или разном бюджете в запусках.
    """
    if not method:
        raise ValueError("method: название не может быть пустым")

    values = _as_vector("final_values", final_values)
    counts = _as_vector("evaluations", evaluations)
    times = _as_vector("times_seconds", times_seconds)

    if not values.size == counts.size == times.size:
        raise ValueError("final_values, evaluations и times_seconds разной длины")
    if np.any(times < 0.0):
        raise ValueError("times_seconds: время не может быть отрицательным")
    if np.any(counts < 1.0) or np.any(counts != np.round(counts)):
        raise ValueError("evaluations: нужны целые числа не меньше 1")
    if np.ptp(counts) != 0.0:
        raise ValueError("evaluations: бюджет должен быть одинаковым во всех запусках")

    n_runs = int(values.size)
    std = float(np.std(values, ddof=STD_DDOF)) if n_runs > STD_DDOF else float("nan")

    threshold: float | None = None
    success_rate: float | None = None
    if success_threshold is not None:
        threshold = float(success_threshold)
        if not np.isfinite(threshold):
            raise ValueError("success_threshold должен быть конечным числом")
        success_rate = float(np.mean(values <= threshold))

    return MethodSummary(
        method=method,
        n_runs=n_runs,
        minimum=float(values.min()),
        maximum=float(values.max()),
        mean=float(values.mean()),
        median=float(np.median(values)),
        std=std,
        evaluations_per_run=int(counts[0]),
        mean_time_seconds=float(times.mean()),
        total_time_seconds=float(times.sum()),
        success_threshold=threshold,
        success_rate=success_rate,
    )


def write_summary_csv(summaries: Sequence[MethodSummary], path: str | Path) -> Path:
    """Сохраняет агрегированную таблицу в CSV без какого-либо округления.

    Создаёт недостающие каталоги. Пустое значение в ячейке означает None.
    """
    if not summaries:
        raise ValueError("Нечего сохранять: список сводок пуст")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(MethodSummary)])
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))
    return target


def format_summary_table(summaries: Sequence[MethodSummary], digits: int = 6) -> str:
    """Текстовая таблица для вывода на экран. Округление только здесь."""
    header = ["method", "runs", "min", "max", "mean", "median", "std", "evals", "time_s"]
    show_success = any(s.success_rate is not None for s in summaries)
    if show_success:
        header.append("success")

    def number(value: float) -> str:
        return f"{value:.{digits}g}"

    rows = []
    for s in summaries:
        row = [
            s.method, str(s.n_runs), number(s.minimum), number(s.maximum),
            number(s.mean), number(s.median), number(s.std),
            str(s.evaluations_per_run), number(s.mean_time_seconds),
        ]
        if show_success:
            row.append("" if s.success_rate is None else number(s.success_rate))
        rows.append(row)

    widths = [max(len(line[i]) for line in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(line, widths))
             for line in [header, *rows]]
    return "\n".join(lines)