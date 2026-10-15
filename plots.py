"""Построение графиков по сохранённым результатам экспериментов.

Данные читаются из файлов в каталоге результатов (histories.npz и
raw_runs.csv), поэтому графики можно перестроить без повторного запуска
экспериментов. Для каждого графика рядом сохраняется CSV с данными, по
которым он построен. Сырые данные не меняются: для логарифмической шкалы
значения ниже LOG_FLOOR поднимаются до LOG_FLOOR только в копии, из которой
рисуется график, и это указывается в подписи оси.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import NDArray

from experiments import HISTORIES_FILE, RAW_RUNS_FILE, load_histories

LOG_FLOOR = 1e-16
DPI = 300

CONVERGENCE_FILE = "convergence_ga.png"
CONVERGENCE_DATA_FILE = "convergence_ga_data.csv"
FINAL_VALUES_FILE = "final_values_boxplot.png"
FINAL_VALUES_DATA_FILE = "final_values_boxplot_data.csv"
COMPARISON_FILE = "ga_vs_random_search.png"
COMPARISON_DATA_FILE = "ga_vs_random_search_data.csv"

# Постоянный цвет эксперимента на всех графиках (палитра Окабе-Ито).
COLORS = {"A": "#0072B2", "B": "#E69F00", "C": "#555555", "D": "#009E73"}
_FALLBACK_COLOR = "#7f7f7f"

X_LABEL = "Число вычислений целевой функции"


@dataclass(frozen=True)
class FinalValues:
    """Итоговые значения запусков одного эксперимента из raw_runs.csv."""

    runs: NDArray[np.int64]
    seeds: NDArray[np.int64]
    values: NDArray[np.float64]


@dataclass(frozen=True)
class PlotData:
    """Всё, что нужно графикам: истории, итоговые значения и подписи методов."""

    histories: dict[str, dict[str, NDArray]]
    final_values: dict[str, FinalValues]
    labels: dict[str, str]

    @property
    def names(self) -> list[str]:
        return sorted(self.histories)

    @property
    def runs_phrase(self) -> str:
        counts = {len(item.values) for item in self.final_values.values()}
        if len(counts) == 1:
            return f"{counts.pop()} независимым запускам"
        return "независимым запускам"


@dataclass(frozen=True)
class PlotFiles:
    """Пути к графику и к CSV с его данными."""

    figure: Path
    data: Path


def _describe(row: dict[str, str]) -> str:
    if row["method"] == "random_search":
        return "случайный поиск"
    sigma = float(row["sigma"])
    if row["sigma_end"] == "":
        return f"ГА, σ = {sigma:g}"
    return f"ГА, σ от {sigma:g} до {float(row['sigma_end']):g}"


def _read_raw_runs(path: Path) -> tuple[dict[str, FinalValues], dict[str, str]]:
    rows_by_experiment: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            rows_by_experiment.setdefault(row["experiment"], []).append(row)
    if not rows_by_experiment:
        raise ValueError(f"{path}: нет данных о запусках")

    final_values: dict[str, FinalValues] = {}
    labels: dict[str, str] = {}
    for name, rows in rows_by_experiment.items():
        final_values[name] = FinalValues(
            runs=np.array([int(r["run"]) for r in rows], dtype=np.int64),
            seeds=np.array([int(r["seed"]) for r in rows], dtype=np.int64),
            values=np.array([float(r["best_f"]) for r in rows], dtype=np.float64),
        )
        labels[name] = _describe(rows[0])
    return final_values, labels


def load_plot_data(results_dir: str | Path) -> PlotData:
    """Загружает данные для графиков из каталога результатов.

    Raises:
        FileNotFoundError: если нет histories.npz или raw_runs.csv.
        ValueError: если файлы относятся к разным экспериментам или запускам.
    """
    folder = Path(results_dir)
    histories = load_histories(folder / HISTORIES_FILE)
    final_values, labels = _read_raw_runs(folder / RAW_RUNS_FILE)

    if set(histories) != set(final_values):
        raise ValueError(
            f"Эксперименты в {HISTORIES_FILE} {sorted(histories)} и в "
            f"{RAW_RUNS_FILE} {sorted(final_values)} не совпадают"
        )
    for name, history in histories.items():
        if not np.array_equal(history["seeds"], final_values[name].seeds):
            raise ValueError(f"Эксперимент {name}: seed в файлах результатов не совпадают")
    return PlotData(histories=histories, final_values=final_values, labels=labels)


def _is_ga(history: dict[str, NDArray]) -> bool:
    return "generation" in history


def _color(name: str) -> str:
    return COLORS.get(name, _FALLBACK_COLOR)


def _band(series: NDArray[np.float64]) -> tuple[NDArray, NDArray, NDArray]:
    """Среднее, минимум и максимум по запускам (оси 0) на каждом шаге."""
    return series.mean(axis=0), series.min(axis=0), series.max(axis=0)


def _floored(values: NDArray) -> NDArray[np.float64]:
    """Копия для рисования на логарифмической шкале (исходный массив не меняется)."""
    return np.maximum(np.asarray(values, dtype=np.float64), LOG_FLOOR)


def _needs_floor(*arrays: NDArray) -> bool:
    return any(bool(np.any(np.asarray(a) < LOG_FLOOR)) for a in arrays)


def _log_label(text: str, clipped: bool) -> str:
    if clipped:
        return (
            f"{text}\n(логарифмическая шкала; значения ниже {LOG_FLOOR:g} "
            f"показаны как {LOG_FLOOR:g})"
        )
    return f"{text}\n(логарифмическая шкала)"


def _save_figure(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    return path


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def plot_convergence(data: PlotData, output_dir: str | Path) -> PlotFiles:
    """График 1: сходимость ГА (A, B, D) по числу вычислений.

    Линия это среднее арифметическое лучшего значения за запуск по всем
    запускам, закрашенная область это диапазон от минимума до максимума.
    """
    names = [n for n in data.names if _is_ga(data.histories[n])]
    if not names:
        raise ValueError("В данных нет экспериментов с генетическим алгоритмом")

    fig = Figure(figsize=(5.2 * len(names), 4.8), layout="constrained")
    axes = fig.subplots(1, len(names), sharey=True, squeeze=False)[0]
    rows: list[list[object]] = []
    clipped_any = False

    for ax, name in zip(axes, names):
        history = data.histories[name]
        evaluations = history["evaluations"]
        n_runs = history["best_so_far"].shape[0]
        mean, low, high = _band(history["best_so_far"])
        clipped_any = clipped_any or _needs_floor(mean, low, high)
        color = _color(name)

        ax.fill_between(
            evaluations, _floored(low), _floored(high), color=color, alpha=0.25,
            label=f"Диапазон min–max по {n_runs} запускам",
        )
        ax.plot(
            evaluations, _floored(mean), color=color, linewidth=1.8,
            label=f"Среднее по {n_runs} запускам",
        )
        ax.set_yscale("log")
        ax.set_title(f"{name}: {data.labels[name]}")
        ax.set_xlabel(X_LABEL)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right")

        rows.extend(
            [name, int(e), n_runs, float(m), float(lo), float(hi)]
            for e, m, lo, hi in zip(evaluations, mean, low, high)
        )

    axes[0].set_ylabel(_log_label("Лучшее найденное значение f(x)", clipped_any))
    fig.suptitle("Сходимость генетического алгоритма")

    target = Path(output_dir)
    return PlotFiles(
        figure=_save_figure(fig, target / CONVERGENCE_FILE),
        data=_write_csv(
            target / CONVERGENCE_DATA_FILE,
            ["experiment", "evaluations", "n_runs", "mean_best_so_far",
             "min_best_so_far", "max_best_so_far"],
            rows,
        ),
    )


def _spread(count: int) -> NDArray[np.float64]:
    """Детерминированные горизонтальные сдвиги точек (без случайности)."""
    return ((np.arange(count) % 5) - 2) * 0.07


def plot_final_values(data: PlotData, output_dir: str | Path) -> PlotFiles:
    """График 2: распределение итоговых значений всех запусков (boxplot и точки)."""
    names = data.names
    samples = [_floored(data.final_values[n].values) for n in names]
    clipped = _needs_floor(*(data.final_values[n].values for n in names))

    fig = Figure(figsize=(1.9 * len(names) + 2.5, 5.2), layout="constrained")
    ax = fig.subplots()
    artists = ax.boxplot(
        samples,
        tick_labels=[f"{n}\n{data.labels[n]}" for n in names],
        showfliers=False,  # все точки рисуются отдельно ниже
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for box, name in zip(artists["boxes"], names):
        box.set_facecolor(_color(name))
        box.set_alpha(0.35)

    for position, (name, sample) in enumerate(zip(names, samples), start=1):
        ax.scatter(
            position + _spread(len(sample)), sample, s=24, color=_color(name),
            edgecolors="black", linewidths=0.4, zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Метод")
    ax.set_ylabel(_log_label("Итоговое лучшее значение f(x)", clipped))
    ax.set_title(f"Итоговые значения по {data.runs_phrase} (каждая точка это один запуск)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor="#bbbbbb", edgecolor="black", alpha=0.6,
                  label="Ящик: от 1-го до 3-го квартиля, линия: медиана, усы: 1.5·IQR"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="#bbbbbb",
                   markeredgecolor="black", label="Итоговое значение одного запуска"),
        ],
        loc="best",
    )

    rows: list[list[object]] = []
    for name in names:
        item = data.final_values[name]
        rows.extend(
            [name, int(run), int(seed), float(value)]
            for run, seed, value in zip(item.runs, item.seeds, item.values)
        )

    target = Path(output_dir)
    return PlotFiles(
        figure=_save_figure(fig, target / FINAL_VALUES_FILE),
        data=_write_csv(
            target / FINAL_VALUES_DATA_FILE, ["experiment", "run", "seed", "best_f"], rows
        ),
    )


def _resample(
    evaluations: NDArray, series: NDArray, grid: NDArray
) -> NDArray[np.float64]:
    """Значения ряда в точках сетки grid (по числу вычислений)."""
    index = np.searchsorted(evaluations, grid)
    if np.any(index >= len(evaluations)) or not np.array_equal(evaluations[index], grid):
        raise ValueError("Сетка вычислений ГА не содержится в истории случайного поиска")
    return series[:, index]


def plot_ga_vs_random_search(data: PlotData, output_dir: str | Path) -> PlotFiles:
    """График 3: среднее лучшее значение ГА и случайного поиска по числу вычислений.

    Случайный поиск записывает историю на каждое вычисление, ГА по одной записи
    на поколение. Сравнение идёт на общей сетке вычислений ГА (60, 120, ...).
    """
    ga_names = [n for n in data.names if _is_ga(data.histories[n])]
    search_names = [n for n in data.names if not _is_ga(data.histories[n])]
    if not ga_names or not search_names:
        raise ValueError("Для сравнения нужны эксперименты и с ГА, и со случайным поиском")

    grid = data.histories[ga_names[0]]["evaluations"]
    for name in ga_names[1:]:
        if not np.array_equal(data.histories[name]["evaluations"], grid):
            raise ValueError("Эксперименты с ГА используют разные сетки вычислений")

    series_by_name: dict[str, NDArray] = {}
    for name in ga_names:
        series_by_name[name] = data.histories[name]["best_so_far"]
    for name in search_names:
        history = data.histories[name]
        series_by_name[name] = _resample(history["evaluations"], history["best_so_far"], grid)

    fig = Figure(figsize=(9.0, 5.6), layout="constrained")
    ax = fig.subplots()
    rows: list[list[object]] = []
    clipped = False
    for name in data.names:
        series = series_by_name[name]
        mean, low, high = _band(series)
        clipped = clipped or _needs_floor(mean)
        ax.plot(
            grid, _floored(mean), color=_color(name), linewidth=1.8,
            linestyle="-" if name in ga_names else "--",
            label=f"{name}: {data.labels[name]}",
        )
        rows.extend(
            [name, int(e), series.shape[0], float(m), float(lo), float(hi)]
            for e, m, lo, hi in zip(grid, mean, low, high)
        )

    ax.set_yscale("log")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(_log_label("Среднее лучшее найденное значение f(x)", clipped))
    ax.set_title(
        f"ГА и случайный поиск при одинаковом числе вычислений "
        f"(среднее арифметическое по {data.runs_phrase})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    target = Path(output_dir)
    return PlotFiles(
        figure=_save_figure(fig, target / COMPARISON_FILE),
        data=_write_csv(
            target / COMPARISON_DATA_FILE,
            ["experiment", "evaluations", "n_runs", "mean_best_so_far",
             "min_best_so_far", "max_best_so_far"],
            rows,
        ),
    )


def make_all_plots(
    results_dir: str | Path, output_dir: str | Path | None = None
) -> dict[str, PlotFiles]:
    """Строит все три графика по данным из results_dir.

    Файлы сохраняются в output_dir (по умолчанию туда же, в results_dir).
    """
    data = load_plot_data(results_dir)
    target = Path(results_dir if output_dir is None else output_dir)
    return {
        "convergence": plot_convergence(data, target),
        "final_values": plot_final_values(data, target),
        "comparison": plot_ga_vs_random_search(data, target),
    }