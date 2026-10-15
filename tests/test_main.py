"""Тесты интерфейса командной строки (main.py)."""

import contextlib
import csv
import hashlib
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
import plots
from experiments import HISTORIES_FILE, RAW_RUNS_FILE, SUMMARY_FILE, load_histories
from main import PROJECT_ROOT, build_parser

SOURCE_FILES = [HISTORIES_FILE, RAW_RUNS_FILE, SUMMARY_FILE]
PLOT_FILES = [
    "convergence_ga.png", "convergence_ga_data.csv",
    "final_values_boxplot.png", "final_values_boxplot_data.csv",
    "ga_vs_random_search.png", "ga_vs_random_search_data.csv",
]
EXPECTED_FILES = sorted([*SOURCE_FILES, *PLOT_FILES])


class ReachedComputation(Exception):
    """Сигнал: управление дошло до запуска экспериментов."""


# --- вспомогательные функции и фикстуры -------------------------------------

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def listing(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_header(output: str) -> list[str]:
    line = next(line for line in output.splitlines() if line.startswith("method"))
    return line.split()


def cli(argv: list[str], capsys) -> tuple[int, str, str]:
    code = main.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180,
    )


def project_results_listing() -> list[str] | None:
    folder = PROJECT_ROOT / "results"
    return listing(folder) if folder.exists() else None


@pytest.fixture(scope="module", autouse=True)
def fast_dpi():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plots, "DPI", 50)
        yield


@pytest.fixture(scope="module")
def quick_run(fast_dpi, tmp_path_factory):
    """Один настоящий быстрый прогон в ещё не существующий каталог."""
    folder = tmp_path_factory.mktemp("quick") / "out"
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main.main(["quick", "--results-dir", str(folder)])
    return SimpleNamespace(code=code, out=out.getvalue(), err=err.getvalue(), folder=folder)


@pytest.fixture
def run_all_spy(monkeypatch):
    """Подменяет run_all: фиксирует вызов и сразу прерывает расчёт."""
    calls: list[tuple] = []

    def fake(specs, **kwargs):
        calls.append((specs, kwargs))
        raise ReachedComputation

    monkeypatch.setattr(main, "run_all", fake)
    return calls


# --- разбор аргументов ------------------------------------------------------

def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args([])
    assert info.value.code == 2


def test_full_defaults_follow_assignment() -> None:
    args = build_parser().parse_args(["full"])
    assert (args.runs, args.base_seed, args.population_size, args.budget) == (20, 202600, 60, 6000)
    assert args.results_dir == "results"
    assert args.success_threshold is None
    assert args.overwrite is False


def test_quick_defaults_are_small() -> None:
    args = build_parser().parse_args(["quick"])
    assert (args.runs, args.population_size, args.budget) == (3, 10, 60)
    assert args.base_seed == 202600
    assert Path(args.results_dir) == Path("results/quick")
    assert args.success_threshold is None
    assert args.overwrite is False


@pytest.mark.parametrize(
    "argv",
    [
        ["quick", "--runs", "0"],
        ["quick", "--runs", "-1"],
        ["quick", "--runs", "abc"],
        ["quick", "--runs", "2.5"],
        ["full", "--budget", "0"],
        ["full", "--population-size", "0"],
        ["quick", "--base-seed", "-1"],
        ["quick", "--success-threshold", "nan"],
        ["quick", "--success-threshold", "inf"],
        ["quick", "--success-threshold", "abc"],
        ["unknown"],
        ["quick", "--unknown-option", "1"],
        ["full", "--sigma", "3"],
        ["full", "--mutation-prob", "0.5"],
    ],
    ids=lambda argv: " ".join(argv),
)
def test_invalid_arguments_exit_with_code_2(argv: list[str], capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main.main(argv)
    assert info.value.code == 2
    assert capsys.readouterr().err != ""


# --- быстрый прогон ---------------------------------------------------------

def test_quick_run_succeeds_and_creates_missing_directory(quick_run) -> None:
    assert quick_run.code == 0
    assert quick_run.err == ""
    assert listing(quick_run.folder) == EXPECTED_FILES


def test_quick_run_prints_plan_progress_and_table(quick_run) -> None:
    out = quick_run.out
    assert "Эксперименты: A, B, C, D | запусков на метод: 3 | seed 202601..202603" in out
    assert "Популяция: 10 | бюджет на запуск: 60 | всего вычислений: 720" in out
    assert len(re.findall(r"^\s+\[[ABCD]\] запуск \d+/3", out, flags=re.MULTILINE)) == 12
    assert "делитель n-1" in out
    assert "Порог успеха не задан" in out
    assert "Это пробный прогон" in out

    lines = out.splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("method"))
    assert lines[header_index].split()[:3] == ["method", "runs", "min"]
    assert [line.split()[0] for line in lines[header_index + 1 : header_index + 5]] == list("ABCD")
    assert "success" not in table_header(out)


def test_quick_run_lists_every_saved_file(quick_run) -> None:
    for name in EXPECTED_FILES:
        assert str(quick_run.folder / name) in quick_run.out


def test_quick_run_saves_consistent_data(quick_run) -> None:
    rows = read_rows(quick_run.folder / RAW_RUNS_FILE)
    assert len(rows) == 12
    for name in "ABCD":
        subset = [r for r in rows if r["experiment"] == name]
        assert [int(r["seed"]) for r in subset] == [202601, 202602, 202603]
        assert {int(r["evaluations"]) for r in subset} == {60}

    summary = read_rows(quick_run.folder / SUMMARY_FILE)
    assert [r["method"] for r in summary] == list("ABCD")
    assert {int(r["n_runs"]) for r in summary} == {3}
    assert {r["success_rate"] for r in summary} == {""}

    histories = load_histories(quick_run.folder / HISTORIES_FILE)
    assert histories["A"]["best_so_far"].shape == (3, 6)
    assert histories["C"]["best_so_far"].shape == (3, 60)


def test_quick_passes_parameters_to_experiments(tmp_path, capsys) -> None:
    folder = tmp_path / "out"
    code, out, err = cli(
        ["quick", "--results-dir", str(folder), "--runs", "2", "--population-size", "6",
         "--budget", "24", "--base-seed", "100"],
        capsys,
    )
    assert code == 0 and err == ""
    rows = read_rows(folder / RAW_RUNS_FILE)
    assert len(rows) == 8
    assert sorted({int(r["seed"]) for r in rows}) == [101, 102]
    assert {int(r["budget"]) for r in rows} == {24}
    assert {int(r["evaluations"]) for r in rows} == {24}
    assert {int(r["population_size"]) for r in rows if r["method"] == "genetic_algorithm"} == {6}
    assert "seed 101..102" in out
    assert "всего вычислений: 192" in out


def test_success_threshold_is_passed_to_summary_and_explained(tmp_path, capsys) -> None:
    folder = tmp_path / "out"
    code, out, _ = cli(
        ["quick", "--results-dir", str(folder), "--runs", "2", "--success-threshold", "1e9"],
        capsys,
    )
    assert code == 0
    for row in read_rows(folder / SUMMARY_FILE):
        assert float(row["success_threshold"]) == 1e9
        assert float(row["success_rate"]) == 1.0
    assert "success" in table_header(out)
    assert "Успехом считается запуск" in out
    assert "Порог успеха не задан" not in out


# --- защита результатов от перезаписи ---------------------------------------

def test_existing_results_are_protected_without_overwrite(tmp_path, capsys) -> None:
    folder = tmp_path / "out"
    argv = ["quick", "--results-dir", str(folder), "--runs", "1"]
    assert cli(argv, capsys)[0] == 0
    before = {name: digest(folder / name) for name in EXPECTED_FILES}

    code, out, err = cli(argv, capsys)
    assert code == 2
    assert out == ""
    assert "Ошибка" in err and "--overwrite" in err
    assert {name: digest(folder / name) for name in EXPECTED_FILES} == before

    code, out, err = cli([*argv, "--overwrite", "--runs", "2"], capsys)
    assert code == 0
    assert "будут перезаписаны" in out
    assert len(read_rows(folder / RAW_RUNS_FILE)) == 8


def test_single_existing_output_file_is_enough_to_block(tmp_path, capsys) -> None:
    (tmp_path / SUMMARY_FILE).write_text("old", encoding="utf-8")
    code, out, err = cli(["quick", "--results-dir", str(tmp_path), "--runs", "1"], capsys)
    assert code == 2
    assert out == "" and SUMMARY_FILE in err
    assert (tmp_path / SUMMARY_FILE).read_text(encoding="utf-8") == "old"


def test_unrelated_files_do_not_block_and_are_kept(tmp_path, capsys) -> None:
    (tmp_path / "notes.txt").write_text("мои заметки", encoding="utf-8")
    code, _, _ = cli(["quick", "--results-dir", str(tmp_path), "--runs", "1"], capsys)
    assert code == 0
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "мои заметки"
    assert listing(tmp_path) == sorted([*EXPECTED_FILES, "notes.txt"])


def test_default_quick_directory_can_be_rerun_freely(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli(["quick", "--runs", "1"], capsys)[0] == 0
    assert cli(["quick", "--runs", "1"], capsys)[0] == 0
    assert cli(["quick", "--runs", "1", "--results-dir", "results/quick/"], capsys)[0] == 0
    assert listing(tmp_path) == ["results"]
    assert listing(tmp_path / "results") == ["quick"]
    assert listing(tmp_path / "results" / "quick") == EXPECTED_FILES


def test_quick_in_any_other_directory_is_protected(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    argv = ["quick", "--runs", "1", "--results-dir", "results"]
    assert cli(argv, capsys)[0] == 0
    code, out, err = cli(argv, capsys)
    assert code == 2 and out == "" and "--overwrite" in err


# --- full: проверки до расчёта ----------------------------------------------

def test_full_with_defaults_reaches_experiments_with_assignment_parameters(
    tmp_path, run_all_spy, capsys
) -> None:
    folder = tmp_path / "out"
    with pytest.raises(ReachedComputation):
        main.main(["full", "--results-dir", str(folder)])
    [(specs, kwargs)] = run_all_spy
    assert list(specs) == list("ABCD")
    assert kwargs["n_runs"] == 20
    assert kwargs["base_seed"] == 202600
    assert {spec.config.budget for spec in specs.values()} == {6000}
    out = capsys.readouterr().out
    assert "всего вычислений: 480000" in out
    assert "Внимание" not in out
    assert not folder.exists()


def test_full_warns_when_parameters_differ_from_assignment(tmp_path, run_all_spy, capsys) -> None:
    with pytest.raises(ReachedComputation):
        main.main(["full", "--results-dir", str(tmp_path / "out"), "--runs", "5",
                   "--base-seed", "7", "--budget", "3000"])
    out = capsys.readouterr().out
    assert "Внимание" in out
    assert "запусков 5 вместо 20" in out
    assert "base-seed 7 вместо 202600" in out
    assert "бюджет 3000 вместо 6000" in out


def test_full_accepts_population_of_exactly_30(tmp_path, run_all_spy) -> None:
    with pytest.raises(ReachedComputation):
        main.main(["full", "--results-dir", str(tmp_path / "out"), "--population-size", "30"])
    assert len(run_all_spy) == 1


def test_quick_accepts_small_population(tmp_path, run_all_spy) -> None:
    with pytest.raises(ReachedComputation):
        main.main(["quick", "--results-dir", str(tmp_path / "out"),
                   "--population-size", "4", "--budget", "8"])


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["full", "--population-size", "20"], "30"),
        (["full", "--population-size", "29"], "30"),
        (["full", "--budget", "6001"], "budget"),
        (["full", "--population-size", "45"], "budget"),
        (["quick", "--budget", "61"], "budget"),
    ],
    ids=["population20", "population29", "budget6001", "population45", "quick_budget61"],
)
def test_invalid_configuration_stops_before_any_computation(
    argv: list[str], fragment: str, tmp_path, run_all_spy, capsys
) -> None:
    folder = tmp_path / "out"
    code, out, err = cli([*argv, "--results-dir", str(folder)], capsys)
    assert code == 2
    assert out == ""
    assert "Ошибка" in err and fragment in err
    assert run_all_spy == []
    assert not folder.exists()


def test_full_refuses_existing_results_without_overwrite(tmp_path, run_all_spy, capsys) -> None:
    (tmp_path / RAW_RUNS_FILE).write_text("old", encoding="utf-8")
    code, out, err = cli(["full", "--results-dir", str(tmp_path)], capsys)
    assert code == 2
    assert out == ""
    assert RAW_RUNS_FILE in err and "--overwrite" in err
    assert run_all_spy == []

    with pytest.raises(ReachedComputation):
        main.main(["full", "--results-dir", str(tmp_path), "--overwrite"])
    # До окончания расчёта старые результаты не трогаются.
    assert (tmp_path / RAW_RUNS_FILE).read_text(encoding="utf-8") == "old"


# --- команда test -----------------------------------------------------------

@pytest.mark.parametrize("returncode", [0, 5])
def test_test_command_runs_pytest_from_project_root(returncode: int, monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(main.subprocess, "run", fake_run)
    assert main.main(["test"]) == returncode
    [(command, kwargs)] = calls
    assert command == [sys.executable, "-m", "pytest", "-q"]
    assert kwargs["cwd"] == PROJECT_ROOT


# --- команда plots ----------------------------------------------------------

def test_plots_command_rebuilds_figures_from_saved_results(quick_run, tmp_path, capsys) -> None:
    folder = tmp_path / "copy"
    shutil.copytree(quick_run.folder, folder)
    for name in PLOT_FILES:
        (folder / name).unlink()
    before = {name: digest(folder / name) for name in SOURCE_FILES}

    code, out, err = cli(["plots", "--results-dir", str(folder)], capsys)

    assert code == 0 and err == ""
    assert listing(folder) == EXPECTED_FILES
    assert {name: digest(folder / name) for name in SOURCE_FILES} == before
    for name in PLOT_FILES:
        assert str(folder / name) in out


def test_plots_command_reports_missing_directory(tmp_path, capsys) -> None:
    missing = tmp_path / "missing"
    code, out, err = cli(["plots", "--results-dir", str(missing)], capsys)
    assert code == 2 and out == ""
    assert "Сначала выполните" in err
    assert not missing.exists()


def test_plots_command_reports_missing_raw_runs(quick_run, tmp_path, capsys) -> None:
    folder = tmp_path / "copy"
    shutil.copytree(quick_run.folder, folder)
    (folder / RAW_RUNS_FILE).unlink()
    code, _, err = cli(["plots", "--results-dir", str(folder)], capsys)
    assert code == 2
    assert RAW_RUNS_FILE in err


def test_plots_command_reports_inconsistent_data(quick_run, tmp_path, capsys) -> None:
    folder = tmp_path / "copy"
    shutil.copytree(quick_run.folder, folder)
    rows = read_rows(folder / RAW_RUNS_FILE)
    with (folder / RAW_RUNS_FILE).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(r for r in rows if r["experiment"] != "C")
    code, _, err = cli(["plots", "--results-dir", str(folder)], capsys)
    assert code == 2
    assert "некорректные данные" in err


# --- запуск как скрипта -----------------------------------------------------

def test_script_help_lists_commands() -> None:
    result = run_script("--help")
    assert result.returncode == 0
    for command in ("quick", "full", "test", "plots"):
        assert command in result.stdout


def test_script_without_command_exits_with_2() -> None:
    assert run_script().returncode == 2


def test_script_returns_exit_code_2_on_error_without_writing_files() -> None:
    before = project_results_listing()
    result = run_script("full", "--population-size", "20")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "30" in result.stderr
    assert project_results_listing() == before


def test_script_quick_run_end_to_end(tmp_path) -> None:
    folder = tmp_path / "out"
    result = run_script("quick", "--results-dir", str(folder), "--runs", "1")
    assert result.returncode == 0, result.stderr
    assert listing(folder) == EXPECTED_FILES
    assert "Это пробный прогон" in result.stdout


# --- изоляция модуля --------------------------------------------------------

def test_importing_main_has_no_side_effects() -> None:
    before = project_results_listing()
    completed = subprocess.run(
        [sys.executable, "-c", "import main, sys; print('matplotlib.pyplot' in sys.modules)"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
    assert project_results_listing() == before


def test_main_contains_no_algorithm_statistics_or_plotting_code() -> None:
    source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import matplotlib", "from matplotlib", "default_rng"):
        assert forbidden not in source
    for name in ("tournament_select", "arithmetic_crossover", "gaussian_mutation",
                 "levy", "OPTIMUM", "OPTIMAL_VALUE"):
        assert not hasattr(main, name)