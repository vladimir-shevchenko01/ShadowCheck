# Конфигурация логгирования для всего проекта.and

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

    # Создаем директорию для логов если её нет
    if enable_file_logging:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Базовые настройки логирования
    config = {
        "disable_existing_loggers": False,
        "version": 1,
        "formatters": {
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
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": sys.stdout,
            },
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
        },
        "loggers": {
            "core": {
                "handlers": ["console", "file_app"],
                "level": log_level,
                "propagate": False,
            },
            "core.detection": {
                "handlers": ["console", "file_app", "file_processing"],
                "level": "DEBUG",
                "propagate": False,
            },
            "core.tracking": {
                "handlers": ["console", "file_app", "file_processing"],
                "level": "DEBUG",
                "propagate": False,
            },
            "core.ocr": {
                "handlers": ["console", "file_app", "file_processing"],
                "level": "INFO",
                "propagate": False,
            },
            "api": {
                "handlers": ["console", "file_app"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console", "file_app"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console", "file_app", "file_errors"],
            "level": "WARNING",
        },
    }

    # Применяем конфигурацию
    logging.config.dictConfig(config)

    # Логируем начало работы
    logger = logging.getLogger(__name__)
    logger.info(f"Логирование настроено. Уровень: {log_level}, Директория: {log_dir}")


def get_logger(name: str) -> logging.Logger:
    """
    Получение логгера с правильной иерархией.

    Args:
        name: Имя модуля (обычно __name__)

    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)
