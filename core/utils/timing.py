"""
Декораторы для замера времени выполнения функций с красивым логированием.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from config import get_logger

logger = get_logger(__name__)


def format_time(seconds: float) -> str:
    """
    Форматирует время в человекочитаемый вид.

    Args:
        seconds: Время в секундах

    Returns:
        Отформатированная строка (например, "1 minute 23.45 seconds" или "1.23s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} min {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_size(size_bytes: float) -> str:
    """
    Форматирует размер в человекочитаемый вид.

    Args:
        size_bytes: Размер в байтах

    Returns:
        Отформатированная строка (например, "15.3 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def timer(
    func: Callable | None = None,
    *,
    name: str | None = None,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable:
    """
    Декоратор для замера времени выполнения функции.

    Args:
        func: Декорируемая функция
        name: Пользовательское имя для логирования (по умолчанию используется __name__)
        log_args: Логировать аргументы функции
        log_result: Логировать результат функции

    Returns:
        Декорированная функция
    """

    def decorator(func_to_decorate: Callable) -> Callable:
        @functools.wraps(func_to_decorate)
        def wrapper(*args, **kwargs) -> Any:
            # Определяем имя для логирования
            func_name = name or func_to_decorate.__name__

            # Логируем начало выполнения
            start_msg = f"🚀 Запуск: {func_name}"
            if log_args and args:
                start_msg += f" | args: {args}"
            if log_args and kwargs:
                start_msg += f" | kwargs: {kwargs}"
            logger.info(start_msg)

            # Засекаем время
            start_time = time.perf_counter()

            try:
                # Выполняем функцию
                result = func_to_decorate(*args, **kwargs)

                # Считаем время
                end_time = time.perf_counter()
                elapsed = end_time - start_time

                # Форматируем сообщение
                time_str = format_time(elapsed)

                # Собираем дополнительную информацию
                extra_info = []
                if log_result and result is not None:
                    if isinstance(result, (str, int, float, bool)):
                        extra_info.append(f"результат: {result}")
                    elif hasattr(result, "__len__"):
                        extra_info.append(f"размер: {len(result)}")

                # Логируем успешное завершение
                success_msg = f"✅ {func_name} завершена за {time_str}"
                if extra_info:
                    success_msg += f" | {', '.join(extra_info)}"
                logger.info(success_msg)

                return result

            except Exception as e:
                # Логируем ошибку с временем
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                time_str = format_time(elapsed)

                logger.error(f"❌ {func_name} упала с ошибкой через {time_str}: {e}")
                raise

        return wrapper

    # Позволяет использовать декоратор как с аргументами, так и без
    if func is None:
        return decorator
    return decorator(func)


class TimerContext:
    """
    Контекстный менеджер для замера времени в блоке кода.

    Пример:
        with TimerContext("Обработка кадра"):
            process_frame(frame)
    """

    def __init__(self, name: str, log_level: str = "INFO") -> None:
        self.name = name
        self.log_level = log_level
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> "TimerContext":
        self.start_time = time.perf_counter()
        logger.log(self.log_level, f"▶️ Начало: {self.name}")  # type: ignore
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        if self.start_time is None:
            return

        self.elapsed = time.perf_counter() - self.start_time
        time_str = format_time(self.elapsed)

        if exc_type is None:
            logger.log(self.log_level, f"⏹️ Конец: {self.name} ({time_str})")  # type: ignore
        else:
            logger.error(f"💥 {self.name} прервана ошибкой через {time_str}: {exc_val}")

    def get_elapsed(self) -> float | None:
        """Возвращает прошедшее время в секундах."""
        return self.elapsed


def async_timer(func: Callable | None = None, *, name: str | None = None) -> Callable:
    """
    Декоратор для замера времени асинхронных функций.

    Пример:
        @async_timer
        async def process_many_videos():
            ...
    """

    def decorator(func_to_decorate: Callable) -> Callable:
        @functools.wraps(func_to_decorate)
        async def wrapper(*args, **kwargs) -> Any:
            func_name = name or func_to_decorate.__name__

            logger.info(f"🚀 Запуск асинхронной функции: {func_name}")
            start_time = time.perf_counter()

            try:
                result = await func_to_decorate(*args, **kwargs)

                elapsed = time.perf_counter() - start_time
                time_str = format_time(elapsed)

                logger.info(f"✅ {func_name} завершена за {time_str}")
                return result

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                time_str = format_time(elapsed)
                logger.error(f"❌ {func_name} упала через {time_str}: {e}")
                raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


class Profiler:
    """
    Профилировщик для сбора статистики по времени выполнения.
    Полезно для оптимизации производительности.
    """

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, float | int]] = {}
        self.current: dict[str, Any] | None = None

    def start(self, name: str) -> None:
        """Начать замер."""
        self.current = {"name": name, "start": time.perf_counter()}

    def stop(self) -> None:
        """Остановить замер и сохранить статистику."""
        if self.current is None:
            return

        elapsed = time.perf_counter() - self.current["start"]
        name = self.current["name"]

        if name not in self.stats:
            self.stats[name] = {
                "count": 0,
                "total": 0.0,
                "min": float("inf"),
                "max": 0.0,
            }

        stats = self.stats[name]
        stats["count"] += 1
        stats["total"] += elapsed
        stats["min"] = min(stats["min"], elapsed)
        stats["max"] = max(stats["max"], elapsed)

        self.current = None

    def report(self) -> None:
        """Вывести отчет по собранной статистике."""
        if not self.stats:
            logger.info("Нет данных для отчета")
            return

        logger.info("=" * 60)
        logger.info("📊 ОТЧЕТ ПРОФИЛИРОВАНИЯ")
        logger.info("=" * 60)

        for name, stats in sorted(
            self.stats.items(), key=lambda x: x[1]["total"], reverse=True
        ):
            avg = stats["total"] / stats["count"]
            logger.info(
                f"{name:30} | "
                f"вызовов: {stats['count']:4d} | "
                f"всего: {format_time(stats['total']):>8} | "
                f"сред: {format_time(avg):>8} | "
                f"мин: {format_time(stats['min']):>8} | "
                f"макс: {format_time(stats['max']):>8}"
            )

        logger.info("=" * 60)


# Создаем глобальный экземпляр профилировщика
profiler = Profiler()
