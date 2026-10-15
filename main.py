"""Интерфейс командной строки: эксперименты, тесты и графики.

Модуль только связывает готовые части проекта: алгоритмы живут в
genetic_algorithm.py и random_search.py, запуски в experiments.py,
статистика в experiment_statistics.py, графики в plots.py.

Команды (из корня проекта, в активированном окружении):
    python main.py quick     быстрый пробный прогон (результаты в results/quick)
    python main.py test      автоматические тесты
    python main.py full      полный эксперимент (4 метода x 20 запусков x 6000)
    python main.py plots     перестроить графики из сохранённых результатов
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from experiment_statistics import format_summary_table
from experiments import (
    BASE_SEED,
    HISTORIES_FILE,
    N_RUNS,
    RAW_RUNS_FILE,
    SUMMARY_FILE,
    RunRecord,
    build_experiments,
    run_all,
    save_results,
    summarize_records,
)
from genetic_algorithm import GAConfig
from plots import (
    COMPARISON_DATA_FILE,
    COMPARISON_FILE,
    CONVERGENCE_DATA_FILE,
    CONVERGENCE_FILE,
    FINAL_VALUES_DATA_FILE,
    FINAL_VALUES_FILE,
    make_all_plots,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = Path("results")
QUICK_RESULTS_DIR = DEFAULT_RESULTS_DIR / "quick"

_ASSIGNMENT = GAConfig()  # параметры из задания: популяция 60, бюджет 6000
MIN_FULL_POPULATION = 30
QUICK_POPULATION = 10
QUICK_BUDGET = 60
QUICK_RUNS = 3

OUTPUT_FILES = (
    RAW_RUNS_FILE, HISTORIES_FILE, SUMMARY_FILE,
    CONVERGENCE_FILE, CONVERGENCE_DATA_FILE,
    FINAL_VALUES_FILE, FINAL_VALUES_DATA_FILE,
    COMPARISON_FILE, COMPARISON_DATA_FILE,
)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"нужно целое число, получено {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("нужно целое число не меньше 1")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"нужно целое число, получено {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("нужно целое число не меньше 0")
    return value


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"нужно число, получено {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("нужно конечное число")
    return value


def _add_experiment_arguments(
    parser: argparse.ArgumentParser, *, results_dir: Path, runs: int,
    population_size: int, budget: int,
) -> None:
    parser.add_argument("--results-dir", default=str(results_dir),
                        help="каталог для результатов (по умолчанию %(default)s)")
    parser.add_argument("--runs", type=_positive_int, default=runs,
                        help="число независимых запусков каждого метода (по умолчанию %(default)s)")
    parser.add_argument("--base-seed", type=_non_negative_int, default=BASE_SEED,
                        help="seed запуска i равен base-seed + i (по умолчанию %(default)s)")
    parser.add_argument("--population-size", type=_positive_int, default=population_size,
                        help="размер популяции ГА (по умолчанию %(default)s)")
    parser.add_argument("--budget", type=_positive_int, default=budget,
                        help="число вычислений целевой функции на запуск (по умолчанию %(default)s)")
    parser.add_argument("--success-threshold", type=_finite_float, default=None,
                        help="порог успеха: запуск успешен, если итоговое значение не больше "
                             "порога. Значения по умолчанию нет, без порога доля успехов не считается")
    parser.add_argument("--overwrite", action="store_true",
                        help="разрешить перезапись существующих результатов")


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Лабораторная работа №1: генетический алгоритм и функция Леви (вариант 7).",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="команда")

    quick = commands.add_parser(
        "quick", help="быстрый пробный прогон на уменьшенных параметрах",
        description="Пробный прогон: числа не для отчёта. Каталог results/quick "
                    "можно перезаписывать без --overwrite, любой другой нельзя.",
    )
    _add_experiment_arguments(
        quick, results_dir=QUICK_RESULTS_DIR, runs=QUICK_RUNS,
        population_size=QUICK_POPULATION, budget=QUICK_BUDGET,
    )

    full = commands.add_parser(
        "full", help="полный эксперимент: 4 метода x 20 запусков x 6000 вычислений",
        description="Полный эксперимент. Масштаб мутации в экспериментах B и D задан "
                    "заданием и через аргументы не меняется.",
    )
    _add_experiment_arguments(
        full, results_dir=DEFAULT_RESULTS_DIR, runs=N_RUNS,
        population_size=_ASSIGNMENT.population_size, budget=_ASSIGNMENT.budget,
    )

    commands.add_parser("test", help="запустить автоматические тесты (pytest)")

    plots = commands.add_parser(
        "plots", help="перестроить графики из сохранённых результатов",
        description="Читает histories.npz и raw_runs.csv и заново строит три графика.",
    )
    plots.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                       help="каталог с результатами (по умолчанию %(default)s)")
    return parser


def _fail(message: str) -> int:
    print(f"Ошибка: {message}", file=sys.stderr)
    return 2


def _existing_outputs(folder: Path) -> list[str]:
    return [name for name in OUTPUT_FILES if (folder / name).exists()]


def _progress(total_runs: int) -> Callable[[RunRecord], None]:
    width = len(str(total_runs))

    def report(record: RunRecord) -> None:
        print(
            f"  [{record.experiment}] запуск {record.run:>{width}}/{total_runs}  "
            f"seed={record.seed}  f={record.best_f:.6e}  {record.time_seconds:.2f} с",
            flush=True,
        )

    return report


def _run_experiments(args: argparse.Namespace) -> int:
    """quick и full: проверки, запуск, сохранение, графики, сводная таблица."""
    results_dir = Path(args.results_dir)
    is_full = args.command == "full"

    if is_full and args.population_size < MIN_FULL_POPULATION:
        return _fail(
            f"для full размер популяции должен быть не меньше {MIN_FULL_POPULATION} "
            f"(требование задания), получено {args.population_size}"
        )

    # Свою папку quick перезаписывает свободно, всё остальное только с --overwrite.
    protected = is_full or results_dir != QUICK_RESULTS_DIR
    existing = _existing_outputs(results_dir)
    if protected and existing and not args.overwrite:
        return _fail(
            f"в {results_dir} уже есть результаты ({', '.join(existing)}). "
            "Укажите другой --results-dir или добавьте --overwrite."
        )

    try:
        specs = build_experiments(population_size=args.population_size, budget=args.budget)
    except ValueError as error:
        return _fail(str(error))

    total_evaluations = len(specs) * args.runs * args.budget
    print(f"Эксперименты: {', '.join(specs)} | запусков на метод: {args.runs} | "
          f"seed {args.base_seed + 1}..{args.base_seed + args.runs}")
    print(f"Популяция: {args.population_size} | бюджет на запуск: {args.budget} | "
          f"всего вычислений: {total_evaluations}")
    for name, spec in specs.items():
        print(f"  {name}: {spec.description}")

    if is_full:
        deviations = []
        if args.runs != N_RUNS:
            deviations.append(f"запусков {args.runs} вместо {N_RUNS}")
        if args.base_seed != BASE_SEED:
            deviations.append(f"base-seed {args.base_seed} вместо {BASE_SEED}")
        if args.population_size != _ASSIGNMENT.population_size:
            deviations.append(f"популяция {args.population_size} вместо {_ASSIGNMENT.population_size}")
        if args.budget != _ASSIGNMENT.budget:
            deviations.append(f"бюджет {args.budget} вместо {_ASSIGNMENT.budget}")
        if deviations:
            print(f"Внимание: параметры отличаются от задания: {'; '.join(deviations)}.")
    if existing and args.overwrite:
        print(f"Существующие файлы в {results_dir} будут перезаписаны.")

    started = time.perf_counter()
    data = run_all(specs, n_runs=args.runs, base_seed=args.base_seed,
                   on_run_finished=_progress(args.runs))
    elapsed = time.perf_counter() - started

    summaries = [summarize_records(records, args.success_threshold) for records in data.values()]
    paths = save_results(data, results_dir, args.success_threshold)
    plot_files = make_all_plots(results_dir)

    print(f"\nГотово за {elapsed:.1f} с. Запусков на метод: {args.runs}. "
          "Сводная таблица (std выборочное, делитель n-1):")
    print(format_summary_table(summaries))
    if args.success_threshold is None:
        print("Порог успеха не задан, доля успехов не считается.")
    else:
        print(f"Успехом считается запуск, итоговое значение которого не больше "
              f"{args.success_threshold:g}.")

    print("\nСохранено:")
    for path in paths.values():
        print(f"  {path}")
    for item in plot_files.values():
        print(f"  {item.figure}")
        print(f"  {item.data}")
    if not is_full:
        print("\nЭто пробный прогон: числа получены на уменьшенных параметрах "
              "и в отчёт не идут.")
    return 0


def _run_tests() -> int:
    """Запускает pytest из корня проекта и возвращает его код завершения."""
    command = [sys.executable, "-m", "pytest", "-q"]
    return subprocess.run(command, cwd=PROJECT_ROOT).returncode


def _run_plots(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    try:
        files = make_all_plots(results_dir)
    except FileNotFoundError as error:
        return _fail(
            f"в {results_dir} не найден файл {error.filename or error}. "
            "Сначала выполните команду quick или full."
        )
    except ValueError as error:
        return _fail(f"некорректные данные в {results_dir}: {error}")
    print("Графики построены:")
    for item in files.values():
        print(f"  {item.figure}")
        print(f"  {item.data}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа. Возвращает код завершения: 0 успех, 2 ошибка аргументов или данных."""
    args = build_parser().parse_args(argv)
    if args.command == "test":
        return _run_tests()
    if args.command == "plots":
        return _run_plots(args)
    return _run_experiments(args)


if __name__ == "__main__":
    sys.exit(main())