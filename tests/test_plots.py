"""Тесты построения графиков (plots.py)."""

import csv
import hashlib
import shutil
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

import plots
from experiments import (
    HISTORIES_FILE,
    RAW_RUNS_FILE,
    SUMMARY_FILE,
    ExperimentSpec,
    build_experiments,
    load_histories,
    run_all,
    run_experiment,
    save_results,
)
from genetic_algorithm import GAConfig
from plots import (
    COLORS,
    COMPARISON_DATA_FILE,
    COMPARISON_FILE,
    CONVERGENCE_DATA_FILE,
    CONVERGENCE_FILE,
    FINAL_VALUES_DATA_FILE,
    FINAL_VALUES_FILE,
    LOG_FLOOR,
    X_LABEL,
    PlotData,
    load_plot_data,
    make_all_plots,
    plot_convergence,
    plot_final_values,
    plot_ga_vs_random_search,
)
from random_search import RandomSearchConfig

ORIGINAL_DPI = plots.DPI  # запоминаем до того, как фикстура ниже ускорит рисование

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POPULATION = 10
BUDGET = 60
N_TEST_RUNS = 5
SOURCE_FILES = (RAW_RUNS_FILE, HISTORIES_FILE, SUMMARY_FILE)
OUTPUT_FILES = (
    CONVERGENCE_FILE, CONVERGENCE_DATA_FILE,
    FINAL_VALUES_FILE, FINAL_VALUES_DATA_FILE,
    COMPARISON_FILE, COMPARISON_DATA_FILE,
)


# --- вспомогательные функции и фикстуры -------------------------------------

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def listing(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", "файл не является PNG"
    return struct.unpack(">II", header[16:24])


def subset(data: PlotData, names: list[str]) -> PlotData:
    return PlotData(
        histories={n: data.histories[n] for n in names},
        final_values={n: data.final_values[n] for n in names},
        labels={n: data.labels[n] for n in names},
    )


def zeroed_history(data: PlotData, experiment: str) -> PlotData:
    histories = {n: {k: v.copy() for k, v in h.items()} for n, h in data.histories.items()}
    histories[experiment]["best_so_far"] = np.zeros_like(histories[experiment]["best_so_far"])
    return replace(data, histories=histories)


def zeroed_final_values(data: PlotData, experiment: str) -> PlotData:
    item = data.final_values[experiment]
    final_values = {**data.final_values, experiment: replace(item, values=np.zeros_like(item.values))}
    return replace(data, final_values=final_values)


def line_by_experiment(ax) -> dict[str, object]:
    return {line.get_label().split(":")[0]: line for line in ax.lines}


@pytest.fixture(scope="module", autouse=True)
def fast_dpi():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plots, "DPI", 50)
        yield


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("results")
    specs = build_experiments(population_size=POPULATION, budget=BUDGET)
    save_results(run_all(specs, n_runs=N_TEST_RUNS), folder)
    return folder


@pytest.fixture(scope="module")
def plotted(results_dir, tmp_path_factory):
    return make_all_plots(results_dir, tmp_path_factory.mktemp("plots"))


@pytest.fixture
def workdir(results_dir, tmp_path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(results_dir, target)
    return target


@pytest.fixture
def captured(monkeypatch) -> dict[str, Figure]:
    """Вместо сохранения PNG запоминает объекты рисунков для проверки изнутри."""
    figures: dict[str, Figure] = {}

    def fake_save(fig: Figure, path: Path) -> Path:
        figures[path.name] = fig
        return path

    monkeypatch.setattr(plots, "_save_figure", fake_save)
    return figures


# --- загрузка данных --------------------------------------------------------

def test_load_plot_data(results_dir) -> None:
    data = load_plot_data(results_dir)
    assert data.names == ["A", "B", "C", "D"]
    assert data.labels == {
        "A": "ГА, σ = 2",
        "B": "ГА, σ = 4",
        "C": "случайный поиск",
        "D": "ГА, σ от 2 до 0.1",
    }
    for name in "ABCD":
        item = data.final_values[name]
        assert list(item.runs) == list(range(1, N_TEST_RUNS + 1))
        assert list(item.seeds) == list(range(202601, 202601 + N_TEST_RUNS))
        assert np.array_equal(item.seeds, data.histories[name]["seeds"])


def test_labels_come_from_data_not_from_code(tmp_path) -> None:
    spec = ExperimentSpec("A", "ГА с нестандартным sigma", GAConfig(population_size=10, budget=60, sigma=3.0))
    save_results({"A": run_experiment(spec, n_runs=2)}, tmp_path)
    assert load_plot_data(tmp_path).labels == {"A": "ГА, σ = 3"}


def test_runs_phrase_uses_actual_run_count(results_dir) -> None:
    data = load_plot_data(results_dir)
    assert data.runs_phrase == f"{N_TEST_RUNS} независимым запускам"
    shorter = replace(data.final_values["A"], runs=data.final_values["A"].runs[:2],
                      seeds=data.final_values["A"].seeds[:2], values=data.final_values["A"].values[:2])
    mixed = replace(data, final_values={**data.final_values, "A": shorter})
    assert mixed.runs_phrase == "независимым запускам"


def test_missing_files_raise(results_dir, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_plot_data(tmp_path)
    only_histories = tmp_path / "only_histories"
    only_histories.mkdir()
    shutil.copy(results_dir / HISTORIES_FILE, only_histories)
    with pytest.raises(FileNotFoundError):
        load_plot_data(only_histories)


def test_mismatched_seeds_raise(workdir) -> None:
    histories = load_histories(workdir / HISTORIES_FILE)
    histories["A"]["seeds"] = histories["A"]["seeds"] + 1
    payload = {f"{n}__{k}": v for n, h in histories.items() for k, v in h.items()}
    np.savez_compressed(workdir / HISTORIES_FILE, **payload)
    with pytest.raises(ValueError, match="seed"):
        load_plot_data(workdir)


def test_mismatched_experiment_sets_raise(workdir) -> None:
    rows = read_rows(workdir / RAW_RUNS_FILE)
    fieldnames = list(rows[0].keys())
    with (workdir / RAW_RUNS_FILE).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(r for r in rows if r["experiment"] != "C")
    with pytest.raises(ValueError):
        load_plot_data(workdir)


def test_raw_runs_without_rows_raises(workdir) -> None:
    fieldnames = list(read_rows(workdir / RAW_RUNS_FILE)[0].keys())
    with (workdir / RAW_RUNS_FILE).open("w", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=fieldnames).writeheader()
    with pytest.raises(ValueError):
        load_plot_data(workdir)


# --- файлы и каталоги -------------------------------------------------------

def test_figure_resolution_is_publication_quality() -> None:
    assert ORIGINAL_DPI == 300


def test_dpi_setting_controls_image_size(results_dir, tmp_path, monkeypatch) -> None:
    data = load_plot_data(results_dir)
    sizes = []
    for dpi in (40, 80):
        monkeypatch.setattr(plots, "DPI", dpi)
        sizes.append(png_size(plot_final_values(data, tmp_path / str(dpi)).figure))
    assert abs(sizes[1][0] - 2 * sizes[0][0]) <= 2
    assert abs(sizes[1][1] - 2 * sizes[0][1]) <= 2


def test_make_all_plots_saves_next_to_results_by_default(workdir) -> None:
    before = listing(workdir)
    assert before == sorted(SOURCE_FILES)
    files = make_all_plots(workdir)
    assert set(files) == {"convergence", "final_values", "comparison"}
    for item in files.values():
        assert item.figure.parent == workdir
        assert item.data.parent == workdir
        assert item.figure.stat().st_size > 0
        assert item.data.stat().st_size > 0
        png_size(item.figure)
    assert listing(workdir) == sorted([*SOURCE_FILES, *OUTPUT_FILES])


def test_output_dir_is_used_and_created(workdir, tmp_path) -> None:
    before = listing(workdir)
    target = tmp_path / "new" / "nested"
    files = make_all_plots(str(workdir), str(target))
    assert listing(target) == sorted(OUTPUT_FILES)
    assert all(item.figure.parent == target for item in files.values())
    assert listing(workdir) == before


def test_source_data_is_not_modified(workdir) -> None:
    before = {name: digest(workdir / name) for name in SOURCE_FILES}
    make_all_plots(workdir)
    assert {name: digest(workdir / name) for name in SOURCE_FILES} == before


def test_repeated_plotting_gives_identical_data_files(results_dir, tmp_path) -> None:
    first = make_all_plots(results_dir, tmp_path / "first")
    second = make_all_plots(results_dir, tmp_path / "second")
    for key in first:
        assert first[key].data.read_bytes() == second[key].data.read_bytes()


def test_figures_have_expected_shape(plotted) -> None:
    width, height = png_size(plotted["convergence"].figure)
    assert width > 2 * height  # три панели в ряд
    width, height = png_size(plotted["final_values"].figure)
    assert width > 0 and height > 0
    width, height = png_size(plotted["comparison"].figure)
    assert width > height


# --- данные, по которым построены графики -----------------------------------

def test_convergence_data_matches_histories(plotted, results_dir) -> None:
    rows = read_rows(plotted["convergence"].data)
    histories = load_histories(results_dir / HISTORIES_FILE)
    assert list(rows[0].keys()) == [
        "experiment", "evaluations", "n_runs", "mean_best_so_far",
        "min_best_so_far", "max_best_so_far",
    ]
    assert sorted({r["experiment"] for r in rows}) == ["A", "B", "D"]
    assert len(rows) == 3 * (BUDGET // POPULATION)
    for name in "ABD":
        subset_rows = [r for r in rows if r["experiment"] == name]
        series = histories[name]["best_so_far"]
        assert [int(r["evaluations"]) for r in subset_rows] == list(histories[name]["evaluations"])
        assert all(int(r["n_runs"]) == N_TEST_RUNS for r in subset_rows)
        np.testing.assert_allclose(
            [float(r["mean_best_so_far"]) for r in subset_rows], series.mean(axis=0), rtol=1e-12
        )
        assert [float(r["min_best_so_far"]) for r in subset_rows] == list(series.min(axis=0))
        assert [float(r["max_best_so_far"]) for r in subset_rows] == list(series.max(axis=0))


def test_final_values_data_matches_raw_runs(plotted, results_dir) -> None:
    rows = read_rows(plotted["final_values"].data)
    raw = {(r["experiment"], int(r["run"])): r for r in read_rows(results_dir / RAW_RUNS_FILE)}
    assert list(rows[0].keys()) == ["experiment", "run", "seed", "best_f"]
    assert len(rows) == 4 * N_TEST_RUNS
    for row in rows:
        source = raw[(row["experiment"], int(row["run"]))]
        assert int(row["seed"]) == int(source["seed"])
        assert float(row["best_f"]) == float(source["best_f"])


def test_comparison_data_uses_common_grid(plotted, results_dir) -> None:
    rows = read_rows(plotted["comparison"].data)
    histories = load_histories(results_dir / HISTORIES_FILE)
    grid = histories["A"]["evaluations"]
    assert sorted({r["experiment"] for r in rows}) == ["A", "B", "C", "D"]
    assert len(rows) == 4 * len(grid)
    for name in "ABCD":
        subset_rows = [r for r in rows if r["experiment"] == name]
        assert [int(r["evaluations"]) for r in subset_rows] == list(grid)
        series = histories[name]["best_so_far"]
        if name == "C":
            series = series[:, grid - 1]  # запись после k вычислений лежит в позиции k - 1
        np.testing.assert_allclose(
            [float(r["mean_best_so_far"]) for r in subset_rows], series.mean(axis=0), rtol=1e-12
        )
        assert all(int(r["n_runs"]) == N_TEST_RUNS for r in subset_rows)


# --- содержимое графика 1: сходимость ---------------------------------------

def test_convergence_figure_structure(captured, results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    plot_convergence(data, tmp_path)
    fig = captured[CONVERGENCE_FILE]
    assert fig.get_suptitle() == "Сходимость генетического алгоритма"
    assert len(fig.axes) == 3

    expected_legend = {
        f"Среднее по {N_TEST_RUNS} запускам",
        f"Диапазон min–max по {N_TEST_RUNS} запускам",
    }
    for ax, name in zip(fig.axes, "ABD"):
        assert ax.get_title().startswith(f"{name}: ")
        assert ax.get_xlabel() == X_LABEL
        assert ax.get_yscale() == "log"
        assert {t.get_text() for t in ax.get_legend().get_texts()} == expected_legend
        assert len(ax.lines) == 1
        line = ax.lines[0]
        assert to_hex(line.get_color()) == to_hex(COLORS[name])
        assert np.array_equal(line.get_xdata(), data.histories[name]["evaluations"])
        np.testing.assert_allclose(
            line.get_ydata(), data.histories[name]["best_so_far"].mean(axis=0), rtol=1e-12
        )

    ylabel = fig.axes[0].get_ylabel()
    assert "Лучшее найденное значение" in ylabel
    assert "логарифмическая шкала" in ylabel
    assert f"{LOG_FLOOR:g}" not in ylabel  # нулей нет, оговорка не нужна


def test_convergence_with_single_ga_experiment(captured, results_dir, tmp_path) -> None:
    files = plot_convergence(subset(load_plot_data(results_dir), ["A"]), tmp_path)
    assert len(captured[CONVERGENCE_FILE].axes) == 1
    assert {r["experiment"] for r in read_rows(files.data)} == {"A"}


def test_convergence_without_ga_raises(results_dir, tmp_path) -> None:
    with pytest.raises(ValueError):
        plot_convergence(subset(load_plot_data(results_dir), ["C"]), tmp_path)


# --- содержимое графика 2: итоговые значения --------------------------------

def test_final_values_figure_structure(captured, results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    plot_final_values(data, tmp_path)
    fig = captured[FINAL_VALUES_FILE]
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "Метод"
    assert ax.get_ylabel().startswith("Итоговое лучшее значение f(x)")
    assert "логарифмическая шкала" in ax.get_ylabel()
    assert ax.get_title().startswith(f"Итоговые значения по {N_TEST_RUNS} независимым запускам")
    assert len(ax.get_legend().get_texts()) == 2
    assert list(ax.get_xticks()) == [1.0, 2.0, 3.0, 4.0]


def test_final_values_shows_every_run_as_a_point(captured, results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    plot_final_values(data, tmp_path)
    ax = captured[FINAL_VALUES_FILE].axes[0]
    assert len(ax.collections) == 4
    for position, (collection, name) in enumerate(zip(ax.collections, data.names), start=1):
        offsets = np.asarray(collection.get_offsets())
        assert offsets.shape == (N_TEST_RUNS, 2)
        assert np.array_equal(offsets[:, 1], data.final_values[name].values)
        assert np.all(np.abs(offsets[:, 0] - position) <= 0.14 + 1e-9)
        assert to_hex(collection.get_facecolor()[0]) == to_hex(COLORS[name])


def test_point_spread_is_deterministic_and_inside_box() -> None:
    shifts = plots._spread(7)
    assert np.array_equal(shifts, plots._spread(7))
    assert len(shifts) == 7
    assert np.all(np.abs(shifts) < 0.275)  # половина ширины ящика 0.55
    assert len(set(shifts[:5])) == 5  # соседние точки не накладываются


def test_final_values_plot_is_identical_across_runs(captured, results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    plot_final_values(data, tmp_path / "one")
    first = [np.asarray(c.get_offsets()).copy() for c in captured[FINAL_VALUES_FILE].axes[0].collections]
    plot_final_values(data, tmp_path / "two")
    second = [np.asarray(c.get_offsets()) for c in captured[FINAL_VALUES_FILE].axes[0].collections]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


# --- содержимое графика 3: ГА и случайный поиск -----------------------------

def test_comparison_figure_structure(captured, results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    plot_ga_vs_random_search(data, tmp_path)
    fig = captured[COMPARISON_FILE]
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == X_LABEL
    assert "логарифмическая шкала" in ax.get_ylabel()
    assert f"среднее арифметическое по {N_TEST_RUNS} независимым запускам" in ax.get_title()
    assert len(ax.lines) == 4

    expected_labels = [f"{name}: {data.labels[name]}" for name in data.names]
    assert [line.get_label() for line in ax.lines] == expected_labels
    assert [t.get_text() for t in ax.get_legend().get_texts()] == expected_labels
    for name, line in line_by_experiment(ax).items():
        assert to_hex(line.get_color()) == to_hex(COLORS[name])
        assert line.get_linestyle() == ("--" if name == "C" else "-")


def test_comparison_lines_show_real_means_on_common_grid(captured, results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    plot_ga_vs_random_search(data, tmp_path)
    lines = line_by_experiment(captured[COMPARISON_FILE].axes[0])
    grid = data.histories["A"]["evaluations"]
    for name in "ABCD":
        series = data.histories[name]["best_so_far"]
        if name == "C":
            series = series[:, grid - 1]
        assert np.array_equal(lines[name].get_xdata(), grid)
        np.testing.assert_allclose(lines[name].get_ydata(), series.mean(axis=0), rtol=1e-12)


def test_same_experiment_has_same_color_on_all_plots(captured, results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    plot_convergence(data, tmp_path)
    plot_final_values(data, tmp_path)
    plot_ga_vs_random_search(data, tmp_path)
    comparison = line_by_experiment(captured[COMPARISON_FILE].axes[0])
    for ax, name in zip(captured[CONVERGENCE_FILE].axes, "ABD"):
        assert to_hex(ax.lines[0].get_color()) == to_hex(comparison[name].get_color())
    boxplot_axes = captured[FINAL_VALUES_FILE].axes[0]
    for collection, name in zip(boxplot_axes.collections, "ABCD"):
        assert to_hex(collection.get_facecolor()[0]) == to_hex(comparison[name].get_color())


def test_comparison_requires_both_method_kinds(results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    with pytest.raises(ValueError):
        plot_ga_vs_random_search(subset(data, ["A", "B", "D"]), tmp_path)
    with pytest.raises(ValueError):
        plot_ga_vs_random_search(subset(data, ["C"]), tmp_path)


def test_comparison_rejects_different_ga_grids(results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    histories = {n: dict(h) for n, h in data.histories.items()}
    histories["B"]["evaluations"] = histories["B"]["evaluations"] * 2
    with pytest.raises(ValueError):
        plot_ga_vs_random_search(replace(data, histories=histories), tmp_path)


def test_comparison_rejects_grid_missing_in_random_search(results_dir, tmp_path) -> None:
    data = load_plot_data(results_dir)
    histories = {n: dict(h) for n, h in data.histories.items()}
    histories["C"]["evaluations"] = np.arange(1, BUDGET + 1) * 10 + 5
    with pytest.raises(ValueError):
        plot_ga_vs_random_search(replace(data, histories=histories), tmp_path)


# --- нулевые значения на логарифмической шкале ------------------------------

def test_zero_history_is_floored_only_for_drawing(captured, results_dir, tmp_path) -> None:
    data = zeroed_history(load_plot_data(results_dir), "A")
    files = plot_convergence(data, tmp_path)
    fig = captured[CONVERGENCE_FILE]

    assert np.all(fig.axes[0].lines[0].get_ydata() == LOG_FLOOR)
    assert f"{LOG_FLOOR:g}" in fig.axes[0].get_ylabel()
    assert np.all(data.histories["A"]["best_so_far"] == 0.0)
    rows = [r for r in read_rows(files.data) if r["experiment"] == "A"]
    assert all(float(r["mean_best_so_far"]) == 0.0 for r in rows)
    assert all(float(r["min_best_so_far"]) == 0.0 for r in rows)
    # Другие эксперименты нарисованы по своим настоящим данным.
    assert np.all(fig.axes[1].lines[0].get_ydata() > LOG_FLOOR)


def test_zero_final_values_are_floored_only_for_drawing(captured, results_dir, tmp_path) -> None:
    data = zeroed_final_values(load_plot_data(results_dir), "A")
    files = plot_final_values(data, tmp_path)
    ax = captured[FINAL_VALUES_FILE].axes[0]

    assert np.all(np.asarray(ax.collections[0].get_offsets())[:, 1] == LOG_FLOOR)
    assert ax.get_yscale() == "log"
    assert f"{LOG_FLOOR:g}" in ax.get_ylabel()
    assert np.all(data.final_values["A"].values == 0.0)
    rows = [r for r in read_rows(files.data) if r["experiment"] == "A"]
    assert [float(r["best_f"]) for r in rows] == [0.0] * N_TEST_RUNS


def test_zero_random_search_is_floored_only_for_drawing(captured, results_dir, tmp_path) -> None:
    data = zeroed_history(load_plot_data(results_dir), "C")
    files = plot_ga_vs_random_search(data, tmp_path)
    ax = captured[COMPARISON_FILE].axes[0]

    assert np.all(line_by_experiment(ax)["C"].get_ydata() == LOG_FLOOR)
    assert f"{LOG_FLOOR:g}" in ax.get_ylabel()
    assert np.all(data.histories["C"]["best_so_far"] == 0.0)
    rows = [r for r in read_rows(files.data) if r["experiment"] == "C"]
    assert all(float(r["mean_best_so_far"]) == 0.0 for r in rows)


# --- изоляция модуля --------------------------------------------------------

def test_importing_plots_does_not_use_pyplot_or_write_files() -> None:
    results = PROJECT_ROOT / "results"
    before = listing(results) if results.exists() else None
    completed = subprocess.run(
        [sys.executable, "-c", "import plots, sys; print('matplotlib.pyplot' in sys.modules)"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
    assert (listing(results) if results.exists() else None) == before


def test_module_does_not_use_known_optimum() -> None:
    assert not hasattr(plots, "OPTIMUM")
    assert not hasattr(plots, "OPTIMAL_VALUE")