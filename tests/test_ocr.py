"""
Тест PlateOCR на реальных изображениях.
Запуск: PYTHONPATH=. python core/ocr/test_ocr.py
"""

import cv2
import numpy as np

from core.ocr.paddle_ocr_wrapper import PlateOCR

# ---------------------------------------------------------------------------
# Тест 1: нормализация без OCR (быстро, без загрузки модели)
# ---------------------------------------------------------------------------


def test_normalize():
    ocr = PlateOCR.__new__(PlateOCR)  # создаём без __init__ — без загрузки модели

    cases = [
        ("M818MM77", "M818MM77"),  # уже чистый
        ("M818MMZZ", "M818MM77"),  # Z → 7
        ("X395CH797.", "X395CH797"),  # точка в конце
        ("x395ch797", "X395CH797"),  # нижний регистр
        ("M 818 MM77", "M818MM77"),  # пробелы
    ]

    print("Тест нормализации:")
    all_ok = True
    for raw, expected in cases:
        result = ocr._normalize(raw)
        ok = result == expected
        status = "✅" if ok else "❌"
        print(f"  {status} {raw!r:20} → {result!r:15} (ожидалось {expected!r})")
        if not ok:
            all_ok = False
    assert all_ok


# ---------------------------------------------------------------------------
# Тест 2: синтетический кроп (работает без реальных файлов)
# ---------------------------------------------------------------------------


def test_synthetic_crop():
    print("\nТест на синтетическом кропе:")

    img = np.ones((80, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, "M818MM77", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    ocr = PlateOCR()
    result = ocr.recognize(img)

    if result:
        print(f"  Нормализован: {result.text!r}")
        print(f"  Уверенность:  {result.confidence:.2f}")
        print(f"  Валиден:      {result.is_valid}")


# ---------------------------------------------------------------------------
# Тест 3: реальный кроп из файла
# ---------------------------------------------------------------------------


def check_real_image(image_path: str):
    print(f"\nТест на реальном изображении: {image_path}")

    ocr = PlateOCR()
    img = cv2.imread(image_path)

    if img is None:
        print(f"  ❌ Не удалось загрузить: {image_path}")
        return

    result = ocr.recognize(img)
    if result:
        print(f"  Нормализован: {result.text!r}")
        print(f"  Уверенность:  {result.confidence:.2f}")
        print(f"  Валиден:      {result.is_valid}")


if __name__ == "__main__":
    test_normalize()
    test_synthetic_crop()

    check_real_image("./data/test_images/clean_plate.jpg")
    check_real_image("./data/test_images/dirty_plate.jpg")
