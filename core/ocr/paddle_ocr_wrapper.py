"""
Враппер над PaddleOCR для распознавания российских номерных знаков.

Ключевые решения:
- lang='en': российские номера используют только буквы схожие с латиницей
- OCR получает кроп, а не полный кадр — точнее и быстрее
- Постобработка: regex-валидация формата номера + нормализация символов
- Ленивая инициализация: модель грузится только при первом вызове recognize()
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

# Таблица замены визуально похожих символов с учётом позиции.
# OCR путает 0/O, 1/I, Z/7 — исправляем в зависимости от того
# буква это или цифра на данной позиции номера.
_DIGIT_FIXES: dict[str, str] = {
    "O": "0",
    "I": "1",
    "Z": "7",
    "S": "5",
    "B": "8",
}
_LETTER_FIXES: dict[str, str] = {
    "0": "O",
    "1": "I",
}

# Российский номер: 1 буква + 3 цифры + 2 буквы + 2-3 цифры региона
# Примеры: M818MM77, X395CH797, A001AA177
_PLATE_PATTERN = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")


@dataclass
class OCRResult:
    """Результат распознавания одного номера."""

    text: str  # нормализованный текст номера
    confidence: float  # уверенность от 0.0 до 1.0
    is_valid: bool  # прошёл ли regex-валидацию формата


class PlateOCR:
    """Распознаватель номерных знаков на базе PaddleOCR.

    Использование:
        ocr = PlateOCR()
        result = ocr.recognize(crop)   # crop — numpy array (BGR)
        if result and result.is_valid:
            print(result.text, result.confidence)
    """

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """
        Args:
            confidence_threshold: минимальная уверенность для возврата результата.
        """
        self.confidence_threshold = confidence_threshold
        self._ocr = None  # ленивая инициализация — грузим только при первом вызове

    def _get_ocr(self):
        """Ленивая инициализация PaddleOCR.

        Почему не в __init__?
        Загрузка модели занимает ~2-3 секунды. Если создать PlateOCR при старте
        приложения, но видео ещё нет — тормозим запуск зря. Грузим при первом
        реальном вызове.
        """
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
            )
        return self._ocr

    def recognize(self, crop: np.ndarray) -> OCRResult | None:
        """Распознаёт номер на вырезанном изображении.

        Args:
            crop: вырезанная область с номерным знаком (BGR numpy array).
                  Получается через: frame[y1:y2, x1:x2]

        Returns:
            OCRResult если распознано выше порога, иначе None.
        """
        if crop is None or crop.size == 0:
            return None

        prepared = self._preprocess(crop)

        try:
            ocr = self._get_ocr()
            result = ocr.ocr(prepared, cls=True)
        except Exception:
            return None

        if not result or not result[0]:
            return None

        # Из всех найденных блоков берём с максимальной уверенностью.
        # На кропе номера обычно один блок, но на всякий случай.
        best_text, best_conf = self._pick_best(result[0])

        if best_conf < self.confidence_threshold:
            return None

        normalized = self._normalize(best_text)
        is_valid = bool(_PLATE_PATTERN.match(normalized))

        return OCRResult(
            text=normalized,
            confidence=best_conf,
            is_valid=is_valid,
        )

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Подготавливает кроп для OCR.

        1. Увеличиваем если слишком маленький — OCR плохо читает мелкие символы
        2. Grayscale → BGR: убираем цветовой шум, сохраняем 3 канала для PaddleOCR
        """
        h, w = crop.shape[:2]

        # Минимальная ширина для уверенного распознавания — эмпирически 200px
        if w < 200:
            scale = 200 / w
            crop = cv2.resize(
                crop,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _pick_best(self, lines: list) -> tuple[str, float]:
        """Из списка распознанных блоков выбирает лучший по confidence."""
        best_text, best_conf = "", 0.0
        for line in lines:
            _, (text, conf) = line
            if conf > best_conf:
                best_conf = conf
                best_text = text
        return best_text, best_conf

    def _normalize(self, text: str) -> str:
        """Нормализует распознанный текст к формату российского номера.

        Шаги:
        1. Убираем всё кроме букв и цифр (точки, тире, пробелы — как в нашем тесте)
        2. Верхний регистр
        3. Позиционная коррекция визуально похожих символов:
           - позиции 0, 4, 5 → буква  (0→O, 1→I)
           - позиции 1, 2, 3, 6, 7, 8 → цифра (O→0, Z→7, S→5)
        """
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()

        if len(cleaned) < 8:
            return cleaned  # слишком короткий — не пытаемся чинить

        chars = list(cleaned)
        letter_positions = {0, 4, 5}
        digit_positions = {1, 2, 3, 6, 7, 8}

        for i, ch in enumerate(chars):
            if i in letter_positions and ch in _LETTER_FIXES:
                chars[i] = _LETTER_FIXES[ch]
            elif i in digit_positions and ch in _DIGIT_FIXES:
                chars[i] = _DIGIT_FIXES[ch]

        return "".join(chars)


def extract_plate_crop(
    frame: np.ndarray,
    bbox: list[int],
    expand_ratio: float = 0.15,
) -> np.ndarray | None:
    """Вырезает область номерного знака из кадра по боксу автомобиля.

    Номер обычно находится в нижней трети бокса. expand_ratio добавляет
    отступ чтобы не обрезать края.

    Args:
        frame:        полный кадр видео
        bbox:         бокс автомобиля [x1, y1, x2, y2] от YOLO
        expand_ratio: на сколько расширить область (0.15 = +15%)

    Returns:
        numpy array с кропом или None если бокс вне кадра
    """
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame.shape[:2]

    box_h = y2 - y1
    box_w = x2 - x1

    # Нижняя треть бокса — там обычно номер
    plate_y1 = y1 + int(box_h * 0.65)
    plate_y2 = y2
    plate_x1 = x1
    plate_x2 = x2

    # Добавляем отступ
    pad_x = int(box_w * expand_ratio)
    pad_y = int((plate_y2 - plate_y1) * expand_ratio)

    plate_x1 = max(0, plate_x1 - pad_x)
    plate_y1 = max(0, plate_y1 - pad_y)
    plate_x2 = min(w_frame, plate_x2 + pad_x)
    plate_y2 = min(h_frame, plate_y2 + pad_y)

    if plate_x2 <= plate_x1 or plate_y2 <= plate_y1:
        return None

    return frame[plate_y1:plate_y2, plate_x1:plate_x2]
