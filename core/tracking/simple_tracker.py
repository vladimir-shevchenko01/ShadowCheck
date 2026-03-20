"""
Простейший трекер на основе IoU.
Используется когда полноценный ByteTrack/BOTSORT ещё не настроен.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Структура данных одного завершённого трека
# ---------------------------------------------------------------------------


@dataclass
class TrackRecord:
    """Полная история одного трека — передаётся в БД после обработки видео.

    Почему dataclass, а не dict?
    - IDE видит поля и их типы → меньше опечаток
    - Pydantic/SQLAlchemy умеют конвертировать dataclass напрямую
    - Легко сериализовать в JSON: `asdict(record)`
    """

    track_id: int
    start_frame: int  # кадр первого появления
    end_frame: int  # кадр последнего появления
    bbox_history: List[List[int]]  # список боксов [x1,y1,x2,y2] по кадрам
    frame_indices: List[int]  # номера кадров для каждого бокса в bbox_history
    confidence_history: List[float]  # уверенность детекции на каждом кадре
    best_bbox: Optional[List[int]] = (
        None  # бокс с наибольшей уверенностью (для OCR/скриншота)
    )
    best_frame: Optional[int] = None  # номер кадра best_bbox
    best_confidence: float = 0.0
    license_plate: Optional[str] = None  # лучший распознанный номер
    plate_confidence: float = 0.0  # уверенность OCR для этого номера


# ---------------------------------------------------------------------------
# Вспомогательная функция IoU
# ---------------------------------------------------------------------------


def _iou(boxA: List[int], boxB: List[int]) -> float:
    """Вычисляем IoU для двух прямоугольников [x1, y1, x2, y2].

    IoU (Intersection over Union) — метрика схожести двух прямоугольников:
    от 0 (не пересекаются) до 1 (совпадают полностью).
    Формула: площадь_пересечения / площадь_объединения
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    return interArea / float(union)


# ---------------------------------------------------------------------------
# Внутреннее состояние живого трека (пока видео идёт)
# ---------------------------------------------------------------------------


@dataclass
class _LiveTrack:
    """Состояние трека во время обработки.

    Отделено от TrackRecord намеренно: здесь хранится то, что нужно
    для матчинга (last_box, missed), а не финальная статистика.
    """

    track_id: int
    last_box: List[int]
    start_frame: int
    last_frame: int
    missed: int = 0
    bbox_history: List[List[int]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    best_bbox: Optional[List[int]] = None
    best_frame: Optional[int] = None
    best_confidence: float = 0.0
    license_plate: Optional[str] = None
    plate_confidence: float = 0.0

    def update_plate(self, plate: str, confidence: float) -> None:
        """Обновляем номер если новая уверенность выше текущей.

        Почему не просто перезаписываем?
        OCR запускается несколько раз за трек — на разных кадрах
        качество распознавания разное. Берём лучший результат.
        """
        if confidence > self.plate_confidence:
            self.license_plate = plate
            self.plate_confidence = confidence

    def update(self, box: List[int], frame_idx: int, confidence: float = 1.0) -> None:
        """Обновляем трек новым наблюдением."""
        self.last_box = box
        self.last_frame = frame_idx
        self.missed = 0
        self.bbox_history.append(box)
        self.frame_indices.append(frame_idx)
        self.confidence_history.append(confidence)

        # Обновляем "лучший кадр" — там, где детектор был наиболее уверен.
        # Именно этот кадр потом пойдёт на OCR и сохранится как скриншот.
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_bbox = box
            self.best_frame = frame_idx

    def to_record(self) -> TrackRecord:
        """Конвертирует живой трек в финальную запись для БД."""
        return TrackRecord(
            track_id=self.track_id,
            start_frame=self.start_frame,
            end_frame=self.last_frame,
            bbox_history=self.bbox_history,
            frame_indices=self.frame_indices,
            confidence_history=self.confidence_history,
            best_bbox=self.best_bbox,
            best_frame=self.best_frame,
            best_confidence=self.best_confidence,
            license_plate=self.license_plate,
            plate_confidence=self.plate_confidence,
        )


# ---------------------------------------------------------------------------
# Сам трекер
# ---------------------------------------------------------------------------


class SimpleTracker:
    """Stateful IoU-трекер с историей треков.

    Жизненный цикл трека:
      1. Новый бокс без матча → создаётся _LiveTrack, попадает в self.tracks
      2. На каждом кадре: матчинг по IoU → .update() у совпавших треков
      3. Трек не нашёл пару track_buffer кадров подряд → финализируется
         и перемещается в self.finished_tracks
      4. После process_video(): вызов get_finished_tracks() → список TrackRecord

    Не предназначен для продакшена (лучше ByteTrack), но для MVP достаточен
    и позволяет полностью контролировать логику.
    """

    def __init__(self, match_thresh: float = 0.3, track_buffer: int = 60) -> None:
        # Порог IoU для сопоставления. 0.3 — разумный минимум:
        # при движении камеры боксы смещаются, IoU падает.
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        # Живые треки: id → _LiveTrack
        self._live: Dict[int, _LiveTrack] = {}

        # Завершённые треки (машина пропала или видео кончилось)
        self._finished: List[TrackRecord] = []

        self._next_id = 1

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def update(
        self,
        boxes: List[List[int]],
        frame_idx: int = 0,
        confidences: Optional[List[float]] = None,
    ) -> List[int]:
        """Обновляем состояние трекера на одном кадре.

        Args:
            boxes:       список боксов [[x1,y1,x2,y2], ...] от детектора.
            frame_idx:   номер текущего кадра (нужен для истории).
            confidences: уверенность детекции для каждого бокса.
                         Если None — считаем 1.0 для всех.
        Returns:
            Список track_id в том же порядке, что и boxes.
        """
        if confidences is None:
            confidences = [1.0] * len(boxes)

        # Сначала стареем все живые треки, потом разберёмся с матчингом
        if not boxes:
            self._age_all(frame_idx)
            return []

        box_ids: List[int] = [-1] * len(boxes)

        if self._live:
            live_ids = list(self._live.keys())
            live_boxes = [self._live[tid].last_box for tid in live_ids]

            # Строим матрицу IoU: строки = новые боксы, столбцы = живые треки
            iou_mat = np.zeros((len(boxes), len(live_boxes)), dtype=float)
            for i, b in enumerate(boxes):
                for j, tb in enumerate(live_boxes):
                    iou_mat[i, j] = _iou(b, tb)

            # Жадный матчинг: берём лучшую пару, зануляем строку+столбец, повторяем.
            # Альтернатива — Hungarian algorithm (scipy), но для MVP это излишне.
            while True:
                if iou_mat.size == 0:
                    break
                i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[i, j] < self.match_thresh:
                    break
                matched_id = live_ids[j]
                box_ids[i] = matched_id
                self._live[matched_id].update(boxes[i], frame_idx, confidences[i])
                iou_mat[i, :] = -1.0
                iou_mat[:, j] = -1.0

        # Новые боксы без пары → новые треки
        for idx, bid in enumerate(box_ids):
            if bid == -1:
                new_id = self._next_id
                self._next_id += 1
                track = _LiveTrack(
                    track_id=new_id,
                    last_box=boxes[idx],
                    start_frame=frame_idx,
                    last_frame=frame_idx,
                )
                track.update(boxes[idx], frame_idx, confidences[idx])
                self._live[new_id] = track
                box_ids[idx] = new_id

        # Стареем треки, которые не получили пары на этом кадре
        assigned = set(box_ids)
        for tid in list(self._live.keys()):
            if tid not in assigned:
                self._live[tid].missed += 1
                if self._live[tid].missed > self.track_buffer:
                    self._finalize(tid)

        return box_ids

    def finalize_all(self) -> None:
        """Завершаем все оставшиеся живые треки.

        Вызывать после окончания видео — иначе треки, активные
        до последнего кадра, не попадут в finished_tracks.
        """
        for tid in list(self._live.keys()):
            self._finalize(tid)

    def get_finished_tracks(self) -> List[TrackRecord]:
        """Возвращает все завершённые треки и очищает список.

        Типичный сценарий использования:
            processor.process_video(path)
            records = tracker.get_finished_tracks()
            db.save_tracks(records)
        """
        result = self._finished.copy()
        self._finished.clear()
        return result

    def reset(self) -> None:
        """Полный сброс состояния — вызывать перед обработкой нового видео."""
        self._live.clear()
        self._finished.clear()
        self._next_id = 1

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _finalize(self, tid: int) -> None:
        """Переносит живой трек в список завершённых."""
        if tid in self._live:
            self._finished.append(self._live[tid].to_record())
            del self._live[tid]

    def _age_all(self, frame_idx: int) -> None:
        """Увеличивает missed у всех живых треков (кадр без детекций)."""
        for tid in list(self._live.keys()):
            self._live[tid].missed += 1
            if self._live[tid].missed > self.track_buffer:
                self._finalize(tid)
