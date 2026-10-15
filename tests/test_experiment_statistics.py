"""Тесты агрегирования результатов (experiment_statistics.py)."""

import csv
import dataclasses
import inspect
import math
import statistics
from pathlib import Path

import numpy as np
import pytest

from experiment_statistics import (
    STD_DDOF,
    MethodSummary,
    format_summary_table,
    summarize_method,
    write_summary_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_summary(values=(1.0, 2.0, 3.0, 4.0), method="demo", threshold=None, times=None):
    values = list(values)
    return summarize_method(
        method,
        values,
        [6000] * len(values),
        times if times is not None else [0.1] * len(values),
        success_threshold=threshold,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


# --- имя модуля -------------------------------------------------------------

def test_standard_library_statistics_is_not_shadowed() -> None:
    assert not (PROJECT_ROOT / "statistics.py").exists()
    assert Path(statistics.__file__).resolve().parent != PROJECT_ROOT


# --- базовые показатели -----------------------------------------------------

def test_basic_statistics() -> None:
    s = make_summary([1.0, 2.0, 3.0, 4.0])
    assert s.method == "demo"
    assert s.n_runs == 4
    assert s.minimum == 1.0
    assert s.maximum == 4.0
    assert s.mean == 2.5
    assert s.median == 2.5
    assert s.std == pytest.approx(math.sqrt(5 / 3))


def test_std_uses_n_minus_one_divisor() -> None:
    # Сумма квадратов отклонений от среднего 5 равна 32.
    # Делитель n - 1 = 7 даёт sqrt(32 / 7), делитель n = 8 дал бы ровно 2.
    s = make_summary([2, 4, 4, 4, 5, 5, 7, 9])
    assert STD_DDOF == 1
    assert s.mean == 5.0
    assert s.std == pytest.approx(math.sqrt(32 / 7))
    assert s.std == pytest.approx(statistics.stdev([2, 4, 4, 4, 5, 5, 7, 9]))
    assert s.std != pytest.approx(2.0)


def test_two_runs_std() -> None:
    assert make_summary([1.0, 3.0]).std == pytest.approx(math.sqrt(2.0))


def test_median_with_odd_count_and_unsorted_input() -> None:
    s = make_summary([9.0, 1.0, 5.0])
    assert s.median == 5.0
    assert s.minimum == 1.0
    assert s.maximum == 9.0
    assert s.mean == 5.0


def test_median_is_robust_to_outlier_while_mean_is_not() -> None:
    s = make_summary([1.0, 2.0, 3.0, 4.0, 1000.0])
    assert s.median == 3.0
    assert s.mean == pytest.approx(202.0)


def test_single_run_has_undefined_std() -> None:
    s = summarize_method("one", [3.0], [10], [1.0])
    assert math.isnan(s.std)
    assert s.n_runs == 1
    assert s.minimum == s.maximum == s.mean == s.median == 3.0


def test_evaluations_and_times() -> None:
    s = summarize_method("m", [1.0, 2.0, 3.0], [6000, 6000, 6000], [0.1, 0.2, 0.6])
    assert s.evaluations_per_run == 6000
    assert isinstance(s.evaluations_per_run, int)
    assert s.mean_time_seconds == pytest.approx(0.3)
    assert s.total_time_seconds == pytest.approx(0.9)


@pytest.mark.parametrize(
    "convert",
    [list, tuple, np.array],
    ids=["list", "tuple", "ndarray"],
)
def test_accepts_sequences_and_arrays(convert) -> None:
    s = summarize_method("m", convert([1.0, 2.0, 3.0]), convert([10, 10, 10]), convert([1.0, 1.0, 1.0]))
    assert s.mean == 2.0


def test_inputs_are_not_modified() -> None:
    values = np.array([3.0, 1.0, 2.0])
    before = values.copy()
    summarize_method("m", values, [10, 10, 10], [1.0, 1.0, 1.0])
    assert np.array_equal(values, before)


def test_summary_is_frozen() -> None:
    s = make_summary()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.mean = 0.0  # type: ignore[misc]


# --- порог успеха -----------------------------------------------------------

def test_success_rate_is_not_computed_without_threshold() -> None:
    s = make_summary()
    assert s.success_threshold is None
    assert s.success_rate is None


def test_success_threshold_has_no_default_value() -> None:
    parameter = inspect.signature(summarize_method).parameters["success_threshold"]
    assert parameter.default is None


def test_success_rate_with_threshold() -> None:
    s = make_summary([1.0, 2.0, 3.0, 4.0], threshold=2.5)
    assert s.success_threshold == 2.5
    assert s.success_rate == 0.5


def test_success_threshold_is_inclusive() -> None:
    # Значение ровно на пороге считается успехом: сравнение «не больше».
    s = make_summary([1.0, 2.0, 3.0, 4.0], threshold=2.0)
    assert s.success_rate == 0.5


def test_success_rate_extremes() -> None:
    assert make_summary([1.0, 2.0], threshold=0.0).success_rate == 0.0
    assert make_summary([1.0, 2.0], threshold=10.0).success_rate == 1.0


@pytest.mark.parametrize("bad_threshold", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_threshold_raises(bad_threshold: float) -> None:
    with pytest.raises(ValueError):
        make_summary(threshold=bad_threshold)


# --- проверка входных данных ------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(final_values=[1.0, 2.0], evaluations=[10, 11], times_seconds=[1.0, 1.0]),
        dict(final_values=[1.0, 2.0], evaluations=[10, 10], times_seconds=[1.0]),
        dict(final_values=[1.0, 2.0], evaluations=[10], times_seconds=[1.0, 1.0]),
        dict(final_values=[1.0, float("nan")], evaluations=[10, 10], times_seconds=[1.0, 1.0]),
        dict(final_values=[1.0, float("inf")], evaluations=[10, 10], times_seconds=[1.0, 1.0]),
        dict(final_values=[1.0, 2.0], evaluations=[10, 10], times_seconds=[1.0, float("nan")]),
        dict(final_values=[], evaluations=[], times_seconds=[]),
        dict(final_values=[[1.0, 2.0]], evaluations=[[10, 10]], times_seconds=[[1.0, 1.0]]),
        dict(final_values=[1.0, 2.0], evaluations=[10, 10], times_seconds=[1.0, -0.1]),
        dict(final_values=[1.0, 2.0], evaluations=[10.5, 10.5], times_seconds=[1.0, 1.0]),
        dict(final_values=[1.0, 2.0], evaluations=[0, 0], times_seconds=[1.0, 1.0]),
    ],
    ids=[
        "different_budgets", "times_too_short", "evaluations_too_short", "nan_value",
        "inf_value", "nan_time", "empty", "two_dimensional", "negative_time",
        "fractional_evaluations", "zero_evaluations",
    ],
)
def test_invalid_input_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        summarize_method("bad", **kwargs)


def test_empty_method_name_raises() -> None:
    with pytest.raises(ValueError):
        summarize_method("", [1.0], [10], [1.0])


# --- сохранение в CSV -------------------------------------------------------

def test_csv_columns_match_summary_fields(tmp_path: Path) -> None:
    path = write_summary_csv([make_summary()], tmp_path / "summary.csv")
    with path.open(encoding="utf-8") as file:
        header = next(csv.reader(file))
    assert header == [f.name for f in dataclasses.fields(MethodSummary)]


def test_csv_has_one_row_per_method_in_order(tmp_path: Path) -> None:
    summaries = [make_summary(method="A"), make_summary(method="B"), make_summary(method="C")]
    rows = read_csv(write_summary_csv(summaries, tmp_path / "summary.csv"))
    assert [row["method"] for row in rows] == ["A", "B", "C"]


def test_csv_creates_missing_directories_and_returns_path(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "summary.csv"
    result = write_summary_csv([make_summary()], str(target))
    assert result == target
    assert target.is_file()


def test_csv_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    write_summary_csv([make_summary(method="A"), make_summary(method="B")], path)
    write_summary_csv([make_summary(method="C")], path)
    assert [row["method"] for row in read_csv(path)] == ["C"]


def test_csv_does_not_round_values(tmp_path: Path) -> None:
    values = [0.1 + 0.2, 1 / 3, 2 / 3, 1e-17]
    s = make_summary(values)
    row = read_csv(write_summary_csv([s], tmp_path / "summary.csv"))[0]
    assert float(row["minimum"]) == 1e-17
    assert float(row["maximum"]) == s.maximum
    assert float(row["mean"]) == s.mean
    assert float(row["median"]) == s.median
    assert float(row["std"]) == s.std
    assert int(row["n_runs"]) == 4
    assert int(row["evaluations_per_run"]) == 6000


def test_csv_stores_missing_threshold_as_empty_cell(tmp_path: Path) -> None:
    rows = read_csv(
        write_summary_csv([make_summary(), make_summary(threshold=2.0)], tmp_path / "summary.csv")
    )
    assert rows[0]["success_threshold"] == "" and rows[0]["success_rate"] == ""
    assert float(rows[1]["success_threshold"]) == 2.0
    assert float(rows[1]["success_rate"]) == 0.5


def test_csv_stores_undefined_std_as_nan(tmp_path: Path) -> None:
    s = summarize_method("one", [3.0], [10], [1.0])
    row = read_csv(write_summary_csv([s], tmp_path / "summary.csv"))[0]
    assert math.isnan(float(row["std"]))


def test_csv_rejects_empty_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_summary_csv([], tmp_path / "summary.csv")
    assert not (tmp_path / "summary.csv").exists()


# --- таблица для вывода на экран --------------------------------------------

def test_table_has_header_and_one_line_per_method() -> None:
    table = format_summary_table([make_summary(method="A"), make_summary(method="B")])
    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[0].split()[0] == "method"
    assert lines[1].split()[0] == "A"
    assert lines[2].split()[0] == "B"
    assert len({len(line) for line in lines}) == 1  # выровнено по столбцам


def test_table_shows_success_column_only_when_threshold_is_set() -> None:
    assert "success" not in format_summary_table([make_summary()])
    with_threshold = format_summary_table([make_summary(), make_summary(threshold=2.0)])
    assert "success" in with_threshold.splitlines()[0]


def test_rounding_applies_only_to_table_not_to_data(tmp_path: Path) -> None:
    s = make_summary([1 / 3, 1 / 3])
    table = format_summary_table([s])
    assert "0.333333" in table
    assert "0.3333333333333333" not in table
    assert s.mean == 1 / 3  # сам объект не изменился
    row = read_csv(write_summary_csv([s], tmp_path / "summary.csv"))[0]
    assert float(row["mean"]) == 1 / 3


def test_table_digits_parameter() -> None:
    s = make_summary([1.23456789, 1.23456789])
    assert "1.23" in format_summary_table([s], digits=3)
    assert "1.23456789" not in format_summary_table([s], digits=3)
    assert "1.23456789" in format_summary_table([s], digits=9)


def test_table_prints_nan_for_undefined_std() -> None:
    table = format_summary_table([summarize_method("one", [3.0], [10], [1.0])])
    assert "nan" in table