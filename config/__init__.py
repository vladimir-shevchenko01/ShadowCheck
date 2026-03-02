# config/__init__.py
"""
Модуль конфигурации проекта.
Предоставляет глобальный объект config с валидированными настройками.
"""
from .loader import config
from .logger_config import get_logger, setup_logging
from .models import AppConfig

__all__ = [
    "config",  # Глобальный объект конфигурации
    "AppConfig",  # Pydantic модель (для type hints)
    "setup_logging",  # Настройка логирования
    "get_logger",  # Получение логгера
]
