"""
Простейший трекер на основе IoU.
Используется когда полноценный ByteTrack/BOTSORT ещё не настроен.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _iou(boxA: List[int], boxB: List[int]) -> float:
    """Вычисляем IoU для двух прямоугольников [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)


class SimpleTracker:
    """Небольшой stateful-трекер.

        Идея: сохраняем последнюю позицию каждого трека, сопоставляем
    a новые обнаружения по максимальному IoU и переносим идентификатор.
        Не предназначен для продакшена, но уже сильно уменьшает мерцание
        рамок и позволяет подписывать ID.
    """

    def __init__(self, match_thresh: float = 0.8, track_buffer: int = 60) -> None:
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.tracks: Dict[int, Dict] = {}
        self._next_id = 1

    def update(self, boxes: List[List[int]]) -> List[int]:
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

        box_ids: List[int] = [-1] * len(boxes)

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
