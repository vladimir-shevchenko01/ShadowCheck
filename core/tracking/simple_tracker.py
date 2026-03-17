"""
Простейший трекер на основе IoU.
Используется когда полноценный ByteTrack/BOTSORT ещё не настроен.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    bbox_history: list[list[int]]  # список боксов [x1,y1,x2,y2] по кадрам
    frame_indices: list[int]  # номера кадров для каждого бокса в bbox_history
    confidence_history: list[float]  # уверенность детекции на каждом кадре
    best_bbox: list[int] | None = (
        None  # бокс с наибольшей уверенностью (для OCR/скриншота)
    )
    best_frame: int | None = None  # номер кадра best_bbox
    best_confidence: float = 0.0


@dataclass
class _LiveTrack:
    """Состояние трека во время обработки.

    Отделено от TrackRecord намеренно: здесь хранится то, что нужно
    для матчинга (last_box, missed), а не финальная статистика.
    """

    track_id: int
    last_box: list[int]
    start_frame: int
    last_frame: int
    missed: int = 0
    bbox_history: list[list[int]] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)
    best_bbox: list[int] | None = None
    best_frame: int | None = None
    best_confidence: float = 0.0

    def update(self, box: list[int], frame_idx: int, confidence: float = 1.0) -> None:
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
        )


def _iou(boxA: list[int], boxB: list[int]) -> float:
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


class SimpleTracker:
    """
    Stateful IoU-трекер с историей треков.

    Жизненный цикл трека:
      1. Новый бокс без матча → создаётся _LiveTrack, попадает в self.tracks
      2. На каждом кадре: матчинг по IoU → .update() у совпавших треков
      3. Трек не нашёл пару track_buffer кадров подряд → финализируется
         и перемещается в self.finished_tracks
      4. После process_video(): вызов get_finished_tracks() → список TrackRecord

    Не предназначен для продакшена (лучше ByteTrack), но для MVP достаточен
    и позволяет полностью контролировать логику.
    """

    def __init__(self, match_thresh: float = 0.8, track_buffer: int = 60) -> None:
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, boxes: list[list[int]]) -> list[int]:
        """Обновляем состояние трекера на новом кадре.

        Args:
            boxes: список боксов из детектора.
        Returns:
            список идентификаторов треков, в том же порядке, что и boxes.
        """
        if not boxes:
            # никаких новых обнаружений — увеличим счетчики пропусков
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.track_buffer:
                    del self.tracks[tid]
            return []

        box_ids: list[int] = [-1] * len(boxes)

        if self.tracks:
            track_ids = list(self.tracks.keys())
            track_boxes = [self.tracks[tid]["box"] for tid in track_ids]
            # матрица IoU (Intersection over Union) — метрика схожести двух прямоугольников:
            # от 0 (не пересекаются) до 1 (совпадают полностью). Строим матрицу где строки —
            # новые боксы, столбцы — существующие треки. Каждая ячейка показывает насколько похожи два прямоугольника.
            iou_mat = np.zeros((len(boxes), len(track_boxes)), dtype=float)
            for i, b in enumerate(boxes):
                for j, tb in enumerate(track_boxes):
                    iou_mat[i, j] = _iou(b, tb)

            # жадное сопоставление максимального IoU
            while True:
                if iou_mat.size == 0:
                    break
                i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[i, j] < self.match_thresh:
                    break
                matched_track = track_ids[j]
                box_ids[i] = matched_track
                # обновляем трек
                self.tracks[matched_track]["box"] = boxes[i]
                self.tracks[matched_track]["missed"] = 0
                # зануляем строку и столбец, чтобы не брать снова
                iou_mat[i, :] = -1
                iou_mat[:, j] = -1

        # для всех не сопоставленных создаём новые ID
        for idx, bid in enumerate(box_ids):
            if bid == -1:
                box_ids[idx] = self._next_id
                self.tracks[self._next_id] = {"box": boxes[idx], "missed": 0}
                self._next_id += 1

        # увеличиваем счётчики у пропавших треков
        assigned = set(box_ids)
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.track_buffer:
                    del self.tracks[tid]

        return box_ids
