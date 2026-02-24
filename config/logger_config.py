# Конфигурация логгирования для всего проекта.

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_file_logging: bool = False,
) -> None:
    """
    Настройка логирования для всего приложения.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_dir: Директория для файлов логов
        enable_file_logging: Включить логирование в файлы
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    else:
        log_dir = Path(log_dir)

    # создаём каталог логов, если нужно
    if enable_file_logging:
        log_dir.mkdir(parents=True, exist_ok=True)

    # форматы сообщений
    formatters = {
        "verbose": {
            "format": "{asctime} | {levelname:8} | {name:20} | {filename:20}:{lineno:4} | {message}",
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "{asctime} | {levelname:8} | {message}",
            "style": "{",
            "datefmt": "%H:%M:%S",
        },
        "json": {
            "format": '{{"time": "{asctime}", "level": "{levelname}", "module": "{name}", "message": "{message}"}}',
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }

    # обработчики (хэндлеры)
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "simple",
            "stream": sys.stdout,
        }
    }
    if enable_file_logging:
        handlers.update(
            {
                "file_app": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "verbose",
                    "filename": str(log_dir / "app.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "file_errors": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "verbose",
                    "filename": str(log_dir / "errors.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "file_processing": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "json",
                    "filename": str(log_dir / "processing.log"),
                    "maxBytes": 20971520,  # 20MB
                    "backupCount": 3,
                    "encoding": "utf-8",
                },
            }
        )

    def _select_handlers(*names: str):
        return [n for n in names if enable_file_logging or not n.startswith("file")]

    loggers = {
        "core": {
            "handlers": _select_handlers("console", "file_app"),
            "level": log_level,
            "propagate": False,
        },
        "core.detection": {
            "handlers": _select_handlers("console", "file_app", "file_processing"),
            "level": "DEBUG",
            "propagate": False,
        },
        "core.tracking": {
            "handlers": _select_handlers("console", "file_app", "file_processing"),
            "level": "DEBUG",
            "propagate": False,
        },
        "core.ocr": {
            "handlers": _select_handlers("console", "file_app", "file_processing"),
            "level": "INFO",
            "propagate": False,
        },
        "api": {
            "handlers": _select_handlers("console", "file_app"),
            "level": log_level,
            "propagate": False,
        },
        "uvicorn": {
            "handlers": _select_handlers("console", "file_app"),
            "level": "INFO",
            "propagate": False,
        },
    }

    root_handlers = ["console"]
    if enable_file_logging:
        root_handlers.extend(["file_app", "file_errors"])

    config = {
        "disable_existing_loggers": False,
        "version": 1,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": loggers,
        "root": {"handlers": root_handlers, "level": log_level},
    }

    # применяем конфигурацию
    logging.config.dictConfig(config)

    log = logging.getLogger(__name__)
    log.info(f"Логирование настроено. Уровень: {log_level}, Директория: {log_dir}")


def get_logger(name: str) -> logging.Logger:
    """
    Получение логгера с правильной иерархией.

    Args:
        name: Имя модуля (обычно __name__)

    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)


setup_logging()

logger = get_logger(__name__)
logger.debug("Логгер успешно инициализирован.")
logger.info("Логирование готово к использованию.")
logger.warning("Это предупреждение для тестирования.")
logger.error("Это ошибка для тестирования.")
