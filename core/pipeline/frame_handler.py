"""
Модуль для отрисовки результатов детекции на кадре.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path  # можно удалить, если не используется

import cv2
import numpy as np

from config import config, get_logger

logger = get_logger(__name__)

# Цвета для разных классов (BGR формат для OpenCV)
COLORS: dict[int | str, tuple[int, int, int]] = {
    2: (0, 255, 0),  # car - зеленый
    5: (0, 0, 255),  # bus - красный
    7: (255, 0, 0),  # truck - синий
    "default": (255, 255, 0),  # по умолчанию желтый
}


def draw_detections(
    frame: np.ndarray,
    boxes: list[list[int]] | list[tuple[int, int, int, int]],
    confidences: list[float],
    class_ids: list[int],
    track_ids: list[int] | None = None,
) -> np.ndarray:
    """
    Рисует рамки и подписи на кадре.

    Args:
        frame: Исходный кадр
        boxes: Список боксов [[x1, y1, x2, y2], ...]
        confidences: Список уверенностей
        class_ids: Список ID классов
        track_ids: Список ID треков (опционально)

    Returns:
        Кадр с нарисованными рамками
    """
    if not boxes:
        return frame

    # Копируем, чтобы не портить оригинал
    annotated_frame = frame.copy()

    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(cls_id, COLORS["default"])

        # Рисуем прямоугольник
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Формируем подпись
        if track_ids is not None and i < len(track_ids):
            track_text = f"ID:{track_ids[i]}"
        else:
            track_text = "ID:?"

        # Если дальше классами будешь реально пользоваться — можно вынести в отдельную функцию
        class_name = "car" if cls_id in [2, 5, 7] else f"cls_{cls_id}"
        label = f"{track_text} | {conf:.2f}"

        # Подготовка текста
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Рисуем фон для текста
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1,  # Заливка
        )

        # Пишем текст
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Черный текст на цветном фоне
            1,
        )

    return annotated_frame


def draw_timestamp(frame: np.ndarray, timestamp: float) -> np.ndarray:
    """
    Рисует временную метку на кадре.

    Args:
        frame: Кадр
        timestamp: Время в секундах

    Returns:
        Кадр с меткой времени
    """
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    millis = int((timestamp - int(timestamp)) * 1000)

    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    cv2.putText(
        frame,
        time_str,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),  # Белый
        2,
    )

    return frame


def save_screenshot(
    frame: np.ndarray,
    track_id: int,
    license_plate: str | None = None,
) -> Path:
    """
    Сохраняет скриншот автомобиля.

    Args:
        frame: Кадр с автомобилем
        track_id: ID трека
        license_plate: Распознанный номер (опционально)

    Returns:
        Путь к сохраненному файлу
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if license_plate:
        filename = f"car_{track_id}_{license_plate}_{timestamp}.jpg"
    else:
        filename = f"car_{track_id}_unknown_{timestamp}.jpg"

    save_path = config.storage.screenshots_dir / filename

    cv2.imwrite(str(save_path), frame)
    logger.info(f"Скриншот сохранен: {save_path}")

    return save_path
