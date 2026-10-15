"""Тесты организации экспериментов (experiments.py)."""

import csv
import dataclasses
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import experiments
from experiment_statistics import summarize_method
from experiments import (
    BASE_SEED,
    HISTORIES_FILE,
    N_RUNS,
    RAW_RUNS_FILE,
    SUMMARY_FILE,
    ExperimentSpec,
    build_experiments,
    histories_to_arrays,
    load_histories,
    run_all,
    run_experiment,
    save_histories,
    save_results,
    summarize_records,
    write_raw_runs_csv,
)
from genetic_algorithm import GAConfig, run_genetic_algorithm
from objective import DIMENSION, LOWER, UPPER, levy
from random_search import RandomSearchConfig, run_random_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POPULATION = 10
BUDGET = 60  # 10 * (1 + 5 поколений)


def sphere(x) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


class CountingObjective:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x) -> float:
        self.calls += 1
        return levy(x)


def config_diff(first, second) -> set[str]:
    return {
        field.name
        for field in dataclasses.fields(first)
        if getattr(first, field.name) != getattr(second, field.name)
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def assert_same_runs(first, second) -> None:
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.seed == b.seed
        assert a.best_f == b.best_f
        assert np.array_equal(a.best_x, b.best_x)
        for field in dataclasses.fields(a.history):
            assert np.array_equal(getattr(a.history, field.name), getattr(b.history, field.name))


@pytest.fixture(scope="module")
def small_specs() -> dict[str, ExperimentSpec]:
    return build_experiments(population_size=POPULATION, budget=BUDGET)


@pytest.fixture(scope="module")
def small_data(small_specs):
    return run_all(small_specs, n_runs=N_RUNS)


# --- протокол и описание экспериментов --------------------------------------

def test_protocol_constants() -> None:
    assert BASE_SEED == 202600
    assert N_RUNS == 20


def test_default_experiments_follow_assignment() -> None:
    specs = build_experiments()
    assert list(specs) == ["A", "B", "C", "D"]
    assert specs["A"].config == GAConfig()
    assert specs["B"].config.sigma == 4.0
    assert specs["C"].config == RandomSearchConfig()
    assert specs["D"].config.sigma == 2.0
    assert specs["D"].config.sigma_end == 0.1
    assert [specs[name].method for name in "ABCD"] == [
        "genetic_algorithm", "genetic_algorithm", "random_search", "genetic_algorithm",
    ]
    assert all(spec.description for spec in specs.values())
    assert all(spec.name == name for name, spec in specs.items())


def test_b_and_d_differ_from_a_in_exactly_one_parameter() -> None:
    specs = build_experiments()
    assert config_diff(specs["A"].config, specs["B"].config) == {"sigma"}
    assert config_diff(specs["A"].config, specs["D"].config) == {"sigma_end"}


def test_population_size_and_budget_of_default_experiments() -> None:
    specs = build_experiments()
    for name in "ABD":
        assert specs[name].config.population_size == 60
        assert specs[name].config.population_size >= 30
    assert {spec.config.budget for spec in specs.values()} == {6000}


def test_total_planned_budget_is_480000() -> None:
    specs = build_experiments()
    assert sum(spec.config.budget for spec in specs.values()) * N_RUNS == 480_000


def test_small_parameters_are_applied_to_every_experiment(small_specs) -> None:
    for name in "ABD":
        assert small_specs[name].config.population_size == POPULATION
    assert {spec.config.budget for spec in small_specs.values()} == {BUDGET}
    assert small_specs["A"].config.sigma == 2.0
    assert small_specs["B"].config.sigma == 4.0
    assert small_specs["D"].config.sigma_end == 0.1


@pytest.mark.parametrize("name", ["", "A__B"])
def test_invalid_experiment_name_raises(name: str) -> None:
    with pytest.raises(ValueError):
        ExperimentSpec(name, "описание", RandomSearchConfig())


# --- независимые запуски ----------------------------------------------------

def test_run_all_collects_20_runs_for_each_of_4_methods(small_data) -> None:
    assert list(small_data) == ["A", "B", "C", "D"]
    assert all(len(records) == 20 for records in small_data.values())


def test_seeds_are_202601_to_202620_for_every_method(small_data) -> None:
    for records in small_data.values():
        assert [r.seed for r in records] == list(range(202601, 202621))
        assert [r.run for r in records] == list(range(1, 21))


def test_custom_base_seed_and_run_count(small_specs) -> None:
    records = run_experiment(small_specs["A"], n_runs=3, base_seed=100)
    assert [r.seed for r in records] == [101, 102, 103]


@pytest.mark.parametrize("n_runs", [0, -1])
def test_invalid_run_count_raises(small_specs, n_runs: int) -> None:
    with pytest.raises(ValueError):
        run_experiment(small_specs["A"], n_runs=n_runs)


def test_records_store_method_parameters_and_budget(small_data, small_specs) -> None:
    for name, records in small_data.items():
        for r in records:
            assert r.experiment == name
            assert r.method == small_specs[name].method
            assert r.config == small_specs[name].config
            assert r.evaluations == BUDGET


def test_records_hold_feasible_results(small_data) -> None:
    for records in small_data.values():
        for r in records:
            assert r.best_x.shape == (DIMENSION,)
            assert np.all(r.best_x >= LOWER) and np.all(r.best_x <= UPPER)
            assert math.isfinite(r.best_f)
            assert levy(r.best_x) == r.best_f
            assert math.isfinite(r.time_seconds) and r.time_seconds >= 0.0


def test_actual_function_calls_match_budget(small_specs) -> None:
    for name, spec in small_specs.items():
        objective = CountingObjective()
        run_experiment(spec, n_runs=3, objective=objective)
        assert objective.calls == 3 * BUDGET, name

    objective = CountingObjective()
    run_all(small_specs, n_runs=2, objective=objective)
    assert objective.calls == 4 * 2 * BUDGET


def test_each_run_equals_direct_call_with_its_seed(small_specs) -> None:
    for name, spec in small_specs.items():
        for r in run_experiment(spec, n_runs=4):
            if isinstance(spec.config, GAConfig):
                direct = run_genetic_algorithm(spec.config, r.seed)
            else:
                direct = run_random_search(spec.config, r.seed)
            assert r.best_f == direct.best_f, name
            assert np.array_equal(r.best_x, direct.best_x), name
            for field in dataclasses.fields(direct.history):
                assert np.array_equal(
                    getattr(r.history, field.name), getattr(direct.history, field.name)
                )


def test_results_do_not_depend_on_other_methods_or_order(small_specs, small_data) -> None:
    reordered = run_all({name: small_specs[name] for name in "DCBA"}, n_runs=N_RUNS)
    assert list(reordered) == ["D", "C", "B", "A"]
    only_a = run_all({"A": small_specs["A"]}, n_runs=N_RUNS)
    for name in "ABCD":
        assert_same_runs(reordered[name], small_data[name])
    assert_same_runs(only_a["A"], small_data["A"])


def test_results_do_not_depend_on_global_numpy_state(small_specs) -> None:
    np.random.seed(0)
    first = run_experiment(small_specs["B"], n_runs=3)
    np.random.seed(999)
    second = run_experiment(small_specs["B"], n_runs=3)
    assert_same_runs(first, second)


def test_repeated_experiment_is_reproducible(small_specs) -> None:
    for name in "ABCD":
        assert_same_runs(
            run_experiment(small_specs[name], n_runs=3),
            run_experiment(small_specs[name], n_runs=3),
        )


def test_same_seed_does_not_make_methods_identical(small_data) -> None:
    a, b, d = (small_data[name][0] for name in "ABD")
    assert a.seed == b.seed == d.seed
    assert not np.array_equal(a.history.mean, b.history.mean)
    assert not np.array_equal(a.history.mean, d.history.mean)
    assert not np.array_equal(b.history.mean, d.history.mean)


def test_different_runs_of_one_method_differ(small_data) -> None:
    for name in "AC":
        first, second = small_data[name][0], small_data[name][1]
        assert not np.array_equal(first.best_x, second.best_x)


def test_callback_is_called_once_per_run_in_order(small_specs) -> None:
    seen = []
    records = run_experiment(small_specs["A"], n_runs=5, on_run_finished=seen.append)
    assert len(seen) == 5
    assert all(a is b for a, b in zip(seen, records))
    assert [r.run for r in seen] == [1, 2, 3, 4, 5]


def test_time_measures_the_run() -> None:
    def slow(x):
        time.sleep(0.005)
        return levy(x)

    spec = ExperimentSpec("S", "медленная функция", GAConfig(population_size=4, budget=8))
    (record,) = run_experiment(spec, n_runs=1, objective=slow)
    assert 0.035 <= record.time_seconds < 5.0  # 8 вычислений по 5 мс


# --- агрегирование ----------------------------------------------------------

def test_summarize_records_matches_manual_statistics(small_data) -> None:
    records = small_data["A"]
    values = np.array([r.best_f for r in records])
    s = summarize_records(records)
    assert s.method == "A"
    assert s.n_runs == 20
    assert s.minimum == values.min()
    assert s.maximum == values.max()
    assert s.mean == values.mean()
    assert s.median == np.median(values)
    assert s.std == pytest.approx(values.std(ddof=1))
    assert s.evaluations_per_run == BUDGET
    assert s.total_time_seconds == pytest.approx(sum(r.time_seconds for r in records))
    assert s.success_rate is None


def test_summarize_records_equals_direct_summary(small_data) -> None:
    records = small_data["B"]
    direct = summarize_method(
        "B",
        [r.best_f for r in records],
        [r.evaluations for r in records],
        [r.time_seconds for r in records],
    )
    assert summarize_records(records) == direct


def test_summarize_records_passes_success_threshold(small_data) -> None:
    records = small_data["C"]
    values = np.array([r.best_f for r in records])
    threshold = float(np.median(values))
    s = summarize_records(records, success_threshold=threshold)
    assert s.success_threshold == threshold
    assert s.success_rate == float(np.mean(values <= threshold))


def test_summarize_records_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        summarize_records([])


# --- сырые результаты в CSV -------------------------------------------------

def test_raw_csv_has_one_row_per_run(small_data, tmp_path: Path) -> None:
    rows = read_rows(write_raw_runs_csv(small_data, tmp_path / "raw.csv"))
    assert len(rows) == 80
    for name in "ABCD":
        subset = [row for row in rows if row["experiment"] == name]
        assert [int(row["seed"]) for row in subset] == list(range(202601, 202621))
        assert [int(row["run"]) for row in subset] == list(range(1, 21))


def test_raw_csv_columns(small_data, tmp_path: Path) -> None:
    path = write_raw_runs_csv(small_data, tmp_path / "raw.csv")
    with path.open(encoding="utf-8") as file:
        header = next(csv.reader(file))
    assert header == [
        "experiment", "method", "run", "seed", "best_f",
        "x1", "x2", "x3", "x4", "x5", "x6",
        "evaluations", "time_seconds",
        "dimension", "lower", "upper", "budget", "population_size", "tournament_size",
        "crossover_prob", "mutation_prob", "sigma", "sigma_end",
    ]


def test_raw_csv_values_are_not_rounded(small_data, tmp_path: Path) -> None:
    rows = read_rows(write_raw_runs_csv(small_data, tmp_path / "raw.csv"))
    flat = [r for records in small_data.values() for r in records]
    assert len(rows) == len(flat)
    for row, r in zip(rows, flat):
        assert row["experiment"] == r.experiment
        assert row["method"] == r.method
        assert float(row["best_f"]) == r.best_f
        assert [float(row[f"x{i}"]) for i in range(1, 7)] == list(r.best_x)
        assert float(row["time_seconds"]) == r.time_seconds
        assert int(row["evaluations"]) == r.evaluations


def test_raw_csv_stores_parameters(small_data, tmp_path: Path) -> None:
    rows = read_rows(write_raw_runs_csv(small_data, tmp_path / "raw.csv"))
    first = {name: next(row for row in rows if row["experiment"] == name) for name in "ABCD"}

    for name in "ABD":
        row = first[name]
        assert int(row["population_size"]) == POPULATION
        assert int(row["tournament_size"]) == 3
        assert float(row["crossover_prob"]) == 0.9
        assert float(row["mutation_prob"]) == 1 / 6
        assert int(row["dimension"]) == 6
        assert float(row["lower"]) == -10.0 and float(row["upper"]) == 10.0
        assert int(row["budget"]) == BUDGET

    assert float(first["A"]["sigma"]) == 2.0 and first["A"]["sigma_end"] == ""
    assert float(first["B"]["sigma"]) == 4.0 and first["B"]["sigma_end"] == ""
    assert float(first["D"]["sigma"]) == 2.0 and float(first["D"]["sigma_end"]) == 0.1

    c = first["C"]
    for column in ("population_size", "tournament_size", "crossover_prob",
                   "mutation_prob", "sigma", "sigma_end"):
        assert c[column] == ""
    assert int(c["budget"]) == BUDGET and int(c["dimension"]) == 6


def test_raw_csv_column_count_follows_dimension(tmp_path: Path) -> None:
    spec = ExperimentSpec("X", "двумерный тест", GAConfig(population_size=4, dimension=2, budget=8))
    records = run_experiment(spec, n_runs=2, objective=sphere)
    rows = read_rows(write_raw_runs_csv({"X": records}, tmp_path / "raw.csv"))
    assert "x1" in rows[0] and "x2" in rows[0] and "x3" not in rows[0]
    assert int(rows[0]["dimension"]) == 2


def test_raw_csv_rejects_empty_and_mixed_dimensions(small_data, tmp_path: Path) -> None:
    target = tmp_path / "raw.csv"
    with pytest.raises(ValueError):
        write_raw_runs_csv({}, target)
    with pytest.raises(ValueError):
        write_raw_runs_csv({"A": []}, target)

    spec = ExperimentSpec("X", "двумерный тест", GAConfig(population_size=4, dimension=2, budget=8))
    records_2d = run_experiment(spec, n_runs=1, objective=sphere)
    with pytest.raises(ValueError):
        write_raw_runs_csv({"A": small_data["A"][:1], "X": records_2d}, target)
    assert not target.exists()


def test_raw_csv_creates_directories_and_accepts_str_path(small_data, tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "raw.csv"
    assert write_raw_runs_csv(small_data, str(target)) == target
    assert target.is_file()


# --- истории ----------------------------------------------------------------

def test_history_arrays_for_genetic_algorithm(small_data) -> None:
    records = small_data["A"]
    arrays = histories_to_arrays(records)
    assert set(arrays) == {"generation", "evaluations", "best", "mean", "worst", "best_so_far", "seeds"}
    assert np.array_equal(arrays["generation"], np.arange(6))
    assert np.array_equal(arrays["evaluations"], POPULATION * np.arange(1, 7))
    assert np.array_equal(arrays["seeds"], np.arange(202601, 202621))
    for key in ("best", "mean", "worst", "best_so_far"):
        assert arrays[key].shape == (20, 6)
    for i, r in enumerate(records):
        assert np.array_equal(arrays["best"][i], r.history.best)
        assert np.array_equal(arrays["best_so_far"][i], r.history.best_so_far)


def test_history_arrays_for_random_search(small_data) -> None:
    records = small_data["C"]
    arrays = histories_to_arrays(records)
    assert set(arrays) == {"evaluations", "best_so_far", "seeds"}
    assert np.array_equal(arrays["evaluations"], np.arange(1, BUDGET + 1))
    assert arrays["best_so_far"].shape == (20, BUDGET)
    for i, r in enumerate(records):
        assert np.array_equal(arrays["best_so_far"][i], r.history.best_so_far)


def test_history_arrays_reject_inconsistent_runs(small_data) -> None:
    longer = build_experiments(population_size=POPULATION, budget=2 * BUDGET)
    for name in "AC":
        other = run_experiment(longer[name], n_runs=1)
        with pytest.raises(ValueError):
            histories_to_arrays([small_data[name][0], other[0]])
    with pytest.raises(ValueError):
        histories_to_arrays([])


def test_histories_round_trip_through_npz(small_data, tmp_path: Path) -> None:
    path = save_histories(small_data, tmp_path / "sub" / HISTORIES_FILE)
    assert path.is_file()
    loaded = load_histories(path)
    assert sorted(loaded) == ["A", "B", "C", "D"]
    for name, records in small_data.items():
        expected = histories_to_arrays(records)
        assert set(loaded[name]) == set(expected)
        for key, value in expected.items():
            assert np.array_equal(loaded[name][key], value)
            assert loaded[name][key].dtype == value.dtype


# --- сохранение результатов -------------------------------------------------

def test_save_results_writes_three_files_in_target_directory(small_data, tmp_path: Path) -> None:
    target = tmp_path / "results_here"
    paths = save_results(small_data, target)
    assert set(paths) == {"raw_runs", "histories", "summary"}
    assert paths["raw_runs"] == target / RAW_RUNS_FILE
    assert paths["histories"] == target / HISTORIES_FILE
    assert paths["summary"] == target / SUMMARY_FILE
    assert sorted(p.name for p in target.iterdir()) == sorted(
        [RAW_RUNS_FILE, HISTORIES_FILE, SUMMARY_FILE]
    )
    assert [p.name for p in tmp_path.iterdir()] == ["results_here"]
    assert all(path.stat().st_size > 0 for path in paths.values())


def test_summary_is_consistent_with_raw_runs(small_data, tmp_path: Path) -> None:
    paths = save_results(small_data, tmp_path / "out")
    raw = read_rows(paths["raw_runs"])
    summary = read_rows(paths["summary"])
    assert [row["method"] for row in summary] == ["A", "B", "C", "D"]
    for row in summary:
        values = np.array([float(r["best_f"]) for r in raw if r["experiment"] == row["method"]])
        assert int(row["n_runs"]) == 20
        assert int(row["evaluations_per_run"]) == BUDGET
        assert float(row["minimum"]) == values.min()
        assert float(row["maximum"]) == values.max()
        assert float(row["mean"]) == pytest.approx(values.mean(), rel=1e-12)
        assert float(row["median"]) == pytest.approx(np.median(values), rel=1e-12)
        assert float(row["std"]) == pytest.approx(values.std(ddof=1), rel=1e-12)
        assert row["success_rate"] == "" and row["success_threshold"] == ""


def test_save_results_passes_success_threshold(small_data, tmp_path: Path) -> None:
    paths = save_results(small_data, tmp_path / "out", success_threshold=1e9)
    for row in read_rows(paths["summary"]):
        assert float(row["success_threshold"]) == 1e9
        assert float(row["success_rate"]) == 1.0


# --- изоляция модуля --------------------------------------------------------

def test_importing_experiments_runs_nothing_and_loads_no_plotting() -> None:
    results_dir = PROJECT_ROOT / "results"

    def listing() -> list[str] | None:
        return sorted(p.name for p in results_dir.iterdir()) if results_dir.exists() else None

    before = listing()
    completed = subprocess.run(
        [sys.executable, "-c", "import experiments, sys; print('matplotlib' in sys.modules)"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
    assert listing() == before


def test_module_does_not_use_known_optimum() -> None:
    assert not hasattr(experiments, "OPTIMUM")
    assert not hasattr(experiments, "OPTIMAL_VALUE")