"""
Модуль детекции транспортных средств на базе YOLO.
Использует глобальную конфигурацию из config.
"""

from __future__ import annotations

from pathlib import Path  # Если реально нужен, иначе удалить
from typing import Any

import numpy as np
from ultralytics import YOLO

from config import config, get_logger

logger = get_logger(__name__)


class YOLODetector:
    """
    Класс для детекции объектов с помощью YOLO.
    Настройки загружаются из глобального конфига.
    """

    def __init__(self) -> None:
        """Инициализация детектора с параметрами из конфига."""
        self.model_path: Path | str = config.detection.model
        self.conf_threshold: float = config.detection.conf_threshold
        self.device: str = config.detection.device
        self.classes: list[int] | None = config.detection.classes

        logger.info("Инициализация YOLO детектора:")
        logger.info(f"  Модель: {self.model_path}")
        logger.info(f"  Порог: {self.conf_threshold}")
        logger.info(f"  Устройство: {self.device}")
        logger.info(f"  Классы: {self.classes}")

        try:
            self.model = YOLO(str(self.model_path))

            # Прогрев модели, чтобы избежать задержек при первом вызове
            logger.info("Прогрев модели...")
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_input, verbose=False)

            logger.info("✅ Модель YOLO успешно загружена")

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели YOLO: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Any | None:
        """
        Выполняет детекцию на одном кадре.

        Args:
            frame: Кадр в формате numpy array (BGR)

        Returns:
            Результаты детекции или None в случае ошибки
        """
        if frame is None or frame.size == 0:
            logger.warning("Получен пустой кадр")
            return None

        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                device=self.device,
                classes=self.classes,
                verbose=False,
            )
            return results[0] if results else None

        except Exception as e:
            logger.error(f"Ошибка при детекции: {e}")
            return None

    def detect_with_boxes(
        self, frame: np.ndarray
    ) -> tuple[list[list[int]], list[float], list[int]]:
        """
        Удобная обертка, возвращающая готовые списки.

        Returns:
            Кортеж (boxes, confidences, class_ids)
        """
        detections = self.detect(frame)

        if detections is None or len(detections) == 0:
            return [], [], []

        try:
            boxes = detections.boxes.xyxy.cpu().numpy().astype(int).tolist()
            confidences = detections.boxes.conf.cpu().numpy().tolist()
            class_ids = detections.boxes.cls.cpu().numpy().astype(int).tolist()

            if boxes:
                logger.debug(f"Найдено {len(boxes)} объектов")

            return boxes, confidences, class_ids

        except Exception as e:
            logger.error(f"Ошибка при обработке результатов: {e}")
            return [], [], []

    def get_class_name(self, class_id: int) -> str:
        """Возвращает название класса по ID."""
        class_names: dict[int, str] = {2: "car", 5: "bus", 7: "truck"}
        return class_names.get(class_id, f"unknown_{class_id}")
