"""
Загрузчик конфигурации с поддержкой YAML + .env + переменных окружения.
Реализует иерархию приоритетов:
1. Значения по умолчанию (Pydantic)
2. YAML файл
3. .env файл
4. Переменные окружения (высший приоритет)
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from sympy import N

from config.logger_config import get_logger
from config.models import AppConfig

logger = get_logger(__name__)


class CinfigLoader:
    """Загрузчик конфигурации"""

    def __init__(
        self,
        config_path: Path | None = None,
        env_path: Path | None = None,
    ):
        """
        Инициализация загрузчика.

        Args:
            config_path: Путь к YAML конфигу
            env_path: Путь к .env файлу
        """
        self.config_path = config_path or Path(__file__).parent / "config.yaml"
        self.env_path = env_path or Path(__file__).parent.parent / ".env"

    def load(self) -> AppConfig:
        """
        Загружает и валидирует конфигурацию.

        Returns:
            AppConfig: Валидированная конфигурация
        """
        logger.info("=" * 50)
        logger.info("Загрузка конфигурации")
        logger.info("=" * 50)

        # 1. Начинаем с пустого словаря
        config_dict: dict[str, Any] = {}

        # 2. Загружаем из YAML
        yaml_config = self._load_yaml()
        if yaml_config:
            logger.info(f"✅ Загружен YAML конфиг: {self.config_path}")
            config_dict.update(yaml_config)
        else:
            logger.warning(f"⚠️ YAML конфиг не найден: {self.config_path}")

        # 3. Загружаем .env файл
        self._load_env_files()

        # 4. Загружаем переменные окружения с префиксом SHADOW_
        env_overrides = self._get_env_overrides()
        if env_overrides:
            logger.info(
                f"✅ Найдены переменные окружения: {list(env_overrides.keys())}"
            )
            self._deep_update(config_dict, env_overrides)

    def _load_yaml(self) -> dict | None:
        """Загрузка конфига из YAML файла"""
        ...

    def _load_env_files(self):
        """Загрузка .env файлов"""
        ...

    def _get_env_overrides(self) -> dict:
        """
        Собирает переменные окружения с префиксом SHADOW_
        и преобразует их в иерархический словарь.

        Пример:
            SHADOW_DETECTION_CONF_THRESHOLD=0.6
            -> {'detection': {'conf_threshold': 0.6}}
        """
        ...

    def _deep_update(self, target: dict, source: dict) -> None:
        """Рекурсивное обновление словаря"""
        ...
