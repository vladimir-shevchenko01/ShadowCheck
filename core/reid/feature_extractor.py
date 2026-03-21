"""
ReID — извлечение векторных "слепков" внешнего вида автомобиля.

Зачем нужно:
    OCR не всегда читает номер (грязь, угол, засвет).
    ReID позволяет узнать машину по внешнему виду без номера.
    Схема: кроп авто → MobileNetV3 → вектор 576-d → cosine similarity.

Как использовать:
    extractor = FeatureExtractor()
    embedding_bytes = extractor.extract(crop)   # crop — numpy BGR
    # сохраняем в БД как BLOB
    # при поиске похожих:
    similarity = FeatureExtractor.cosine_similarity(bytes_a, bytes_b)
    if similarity > config.reid.similarity_threshold:
        # та же машина
"""

from __future__ import annotations

import cv2
import numpy as np


class FeatureExtractor:
    """Извлекает вектор признаков из кропа автомобиля.

    Жизненный цикл:
        1. __init__: загружаем MobileNetV3, обрезаем классификатор
        2. extract(crop): препроцессинг → инференс → нормализованный вектор → bytes
        3. cosine_similarity(a, b): сравниваем два вектора из БД
    """

    # Размер входа MobileNetV3 (стандарт ImageNet)
    INPUT_SIZE = (224, 224)

    # Нормализация ImageNet — модель обучена на этих значениях
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self) -> None:
        import torch
        import torchvision.models as models

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MobileNetV3-small: быстрый, лёгкий, достаточно точный для ReID
        # pretrained=True — веса ImageNet, хорошая основа для визуальных признаков
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # Убираем финальный классификатор (1000 классов ImageNet нам не нужны).
        # Оставляем только feature-часть → на выходе тензор [1, 576, 1, 1]
        # AdaptiveAvgPool схлопывает пространственные размеры → [1, 576]
        self._model = torch.nn.Sequential(
            backbone.features,
            backbone.avgpool,
            torch.nn.Flatten(),
        ).to(self._device)

        self._model.eval()  # отключаем dropout и batchnorm в режиме обучения

        import torch

        # Прогрев — первый инференс всегда медленнее из-за JIT-компиляции
        dummy = torch.zeros(1, 3, *self.INPUT_SIZE).to(self._device)
        with torch.no_grad():
            self._model(dummy)

    def extract(self, crop: np.ndarray) -> bytes | None:
        """Извлекает вектор признаков из кропа.

        Args:
            crop: numpy array BGR (как из OpenCV), любого размера.

        Returns:
            Вектор как bytes (float32 little-endian) для хранения в SQLite BLOB.
            None если кроп пустой или слишком маленький.
        """
        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        tensor = self._preprocess(crop)

        import torch

        with torch.no_grad():
            embedding = self._model(tensor)  # [1, 576]

        # L2-нормализация: приводим вектор к единичной длине.
        # После этого cosine_similarity = просто dot product.
        vec = embedding[0].cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.astype(np.float32).tobytes()

    def _preprocess(self, crop: np.ndarray):
        """BGR numpy → нормализованный float32 тензор [1, 3, 224, 224]."""
        import torch

        # OpenCV читает BGR, PyTorch ждёт RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        # [H, W, C] → [C, H, W], нормализация ImageNet
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - self._MEAN) / self._STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

        return tensor.to(self._device)

    # ------------------------------------------------------------------
    # Статические утилиты для работы с эмбеддингами из БД
    # ------------------------------------------------------------------

    @staticmethod
    def bytes_to_vector(data: bytes) -> np.ndarray:
        """Десериализует bytes из БД обратно в numpy вектор."""
        return np.frombuffer(data, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: bytes, b: bytes) -> float:
        """Косинусное сходство двух эмбеддингов из БД.

        Возвращает float от -1 до 1:
            1.0  — одинаковые векторы (та же машина)
            0.0  — не похожи
           -1.0  — противоположные (на практике не бывает)

        Поскольку векторы L2-нормированы в extract(),
        cosine similarity = просто dot product.
        """
        vec_a = FeatureExtractor.bytes_to_vector(a)
        vec_b = FeatureExtractor.bytes_to_vector(b)
        return float(np.dot(vec_a, vec_b))
