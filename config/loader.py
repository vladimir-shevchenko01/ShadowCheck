"""
Загрузчик конфигурации с поддержкой YAML + .env + переменных окружения.
Реализует иерархию приоритетов:
1. Значения по умолчанию (Pydantic)
2. YAML файл
3. .env файл
4. Переменные окружения (высший приоритет)
"""

from __future__ import annotations  # Для современного парсинга аннотаций [web:21]

import os
from pathlib import Path
from typing import Any  # Any остаётся актуальным [web:12]

import yaml
from dotenv import load_dotenv
from sympy import N

from config.logger_config import get_logger
from config.models import AppConfig

logger = get_logger(__name__)


class ConfigLoader:
    """Загрузчик конфигурации"""

    def __init__(
        self,
        config_path: Path | None = None,
        env_path: Path | None = None,
    ) -> None:
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

        # 5. Валидация через Pydantic
        try:
            app_config = AppConfig.model_validate(config_dict)
            self._log_config_summary(app_config)
            return app_config
        except Exception as e:
            logger.error(f"❌ Ошибка валидации конфига: {e}")
            logger.warning("⚠️ Используется конфигурация по умолчанию")
            return AppConfig()

    def _load_yaml(self) -> dict[str, Any] | None:
        """Загрузка конфига из YAML файла"""
        if not self.config_path.exists():
            return None

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки YAML: {e}")
            return None

    def _load_env_files(self) -> None:
        """Загрузка .env файлов"""
        # Пробуем загрузить указанный .env
        if self.env_path.exists():
            load_dotenv(self.env_path)
            logger.debug(f"Загружен .env: {self.env_path}")

        # Также пробуем загрузить .env из корня проекта
        root_env = Path.cwd() / ".env"
        if root_env.exists() and root_env != self.env_path:
            load_dotenv(root_env)
            logger.debug(f"Загружен .env из корня: {root_env}")

    def _get_env_overrides(self) -> dict[str, Any]:
        """
        Собирает переменные окружения с префиксом SHADOW_
        и преобразует их в иерархический словарь.
        """
        overrides: dict[str, Any] = {}
        prefix = "SHADOW_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Убираем префикс и разбиваем по _
            parts = key[len(prefix) :].lower().split("_")

            # Пытаемся преобразовать значение
            typed_value = self._parse_value(value)

            # Строим вложенный словарь
            current: dict[str, Any] = overrides
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = typed_value
            logger.debug(f"  {key} -> {'.'.join(parts)} = {typed_value}")

        return overrides

    def _parse_value(self, value: str) -> int | float | bool | str | list[Any] | None:
        """Парсит строковое значение в соответствующий тип"""
        value = value.strip()

        # Пустые строки
        if not value:
            return None

        # Булевы значения
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Числа
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Списки (значения через запятую)
        if "," in value:
            return [self._parse_value(v.strip()) for v in value.split(",")]

        # Обычные строки
        return value

    def _deep_update(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """Рекурсивное обновление словаря"""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def _log_config_summary(self, config: AppConfig) -> None:
        """Логирует краткую сводку по конфигурации"""
        logger.info("-" * 50)
        logger.info("📋 Сводка конфигурации:")
        logger.info(f"  Приложение: {config.app_name} v{config.version}")
        logger.info(f"  Режим: {'DEBUG' if config.debug else 'PRODUCTION'}")
        logger.info(
            f"  Детекция: {config.detection.model}, порог {config.detection.conf_threshold}"
        )
        logger.info(f"  Устройство: {config.detection.device}")
        logger.info(f"  OCR: {'включен' if config.ocr.enable else 'выключен'}")
        logger.info(f"  ReID: {'включен' if config.reid.enable else 'выключен'}")
        logger.info(f"  Путь к БД: {config.storage.database_path}")
        logger.info(f"  Папка с видео: {config.storage.input_root}")
        logger.info("=" * 50)


# Создаем глобальный экземпляр конфига для использования во всем приложении
_config_loader = ConfigLoader()
config = _config_loader.load()
