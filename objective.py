"""Обобщённая функция Леви (вариант 7) и учёт вычислений целевой функции.

Задача: минимизация на кубе [-10, 10]^6. Модуль не содержит случайности
и не знает про алгоритмы поиска.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

VARIANT = 7
DIMENSION = 5 + VARIANT % 6  # d = 6
LOWER = -10.0
UPPER = 10.0

# Оптимум нужен только для тестов и подсчёта ошибки.
# Алгоритмы поиска его использовать не должны.
OPTIMUM: NDArray[np.float64] = np.ones(DIMENSION)
OPTIMUM.setflags(write=False)
OPTIMAL_VALUE = 0.0


def levy(x: ArrayLike) -> float:
    """Значение обобщённой функции Леви в точке x.

    Скобки сохраняют структуру из задания: в сумме стоит
    sin^2(pi * w_i + 1), то есть квадрат синуса от всего аргумента.

    Raises:
        ValueError: если x не вектор длины DIMENSION или содержит NaN/inf.
    """
    point = np.asarray(x, dtype=np.float64)
    if point.shape != (DIMENSION,):
        raise ValueError(
            f"Ожидался вектор формы ({DIMENSION},), получена форма {point.shape}"
        )
    if not np.all(np.isfinite(point)):
        raise ValueError("Вектор содержит NaN или inf")

    w = 1.0 + (point - 1.0) / 4.0

    first = np.sin(np.pi * w[0]) ** 2
    middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    last = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)

    return float(first + middle + last)


class EvaluationCounter:
    """Обёртка над целевой функцией, считающая фактические вызовы.

    Вызов засчитывается только после успешного вычисления: если функция
    выбросила исключение, счётчик не растёт.
    """

    def __init__(self, function: Callable[[ArrayLike], float] = levy) -> None:
        self._function = function
        self._count = 0

    @property
    def count(self) -> int:
        """Сколько раз функция реально вычислялась."""
        return self._count

    def __call__(self, x: ArrayLike) -> float:
        value = self._function(x)
        self._count += 1
        return value