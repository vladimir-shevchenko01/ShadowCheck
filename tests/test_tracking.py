import numpy as np

from core.pipeline.video_processor import VideoProcessor
from core.tracking.simple_tracker import SimpleTracker, TrackRecord


def _box(x: int, y: int, w: int = 50, h: int = 50):
    """Хелпер: создаём бокс по верхнему левому углу и размеру."""
    return [x, y, x + w, y + h]


# ---------------------------------------------------------------------------
# Тест 1: базовый матчинг
# ---------------------------------------------------------------------------
def test_same_box_keeps_id():
    """Один и тот же бокс на двух кадрах подряд → один трек с тем же ID."""
    t = SimpleTracker()
    box = _box(100, 100)

    ids_frame0 = t.update([box], frame_idx=0)
    ids_frame1 = t.update([box], frame_idx=1)

    assert ids_frame0[0] == ids_frame1[0], "ID должен сохраниться"


# ---------------------------------------------------------------------------
# Тест 2: два разных объекта → два разных ID
# ---------------------------------------------------------------------------
def test_two_separate_boxes_get_different_ids():
    t = SimpleTracker()
    box_a = _box(0, 0)
    box_b = _box(500, 500)  # далеко от первого → IoU = 0

    ids = t.update([box_a, box_b], frame_idx=0)
    assert ids[0] != ids[1], "Разные объекты должны получить разные ID"


# ---------------------------------------------------------------------------
# Тест 3: трек финализируется после track_buffer пропущенных кадров
# ---------------------------------------------------------------------------
def test_track_finalized_after_buffer():
    t = SimpleTracker(track_buffer=3)
    box = _box(100, 100)

    t.update([box], frame_idx=0)

    # 4 кадра без детекций → трек должен умереть
    for i in range(1, 5):
        t.update([], frame_idx=i)

    records = t.get_finished_tracks()
    assert len(records) == 1, "Должен быть один завершённый трек"
    assert records[0].start_frame == 0
    assert records[0].end_frame == 0


# ---------------------------------------------------------------------------
# Тест 4: finalize_all() подбирает активные треки
# ---------------------------------------------------------------------------
def test_finalize_all_captures_active_tracks():
    t = SimpleTracker()
    box = _box(100, 100)

    t.update([box], frame_idx=0)
    t.update([box], frame_idx=1)
    t.update([box], frame_idx=2)

    # Трек ещё живой — без finalize_all он не попадёт в finished
    assert t.get_finished_tracks() == []

    t.finalize_all()
    records = t.get_finished_tracks()

    assert len(records) == 1
    assert records[0].start_frame == 0
    assert records[0].end_frame == 2
    assert len(records[0].bbox_history) == 3


# ---------------------------------------------------------------------------
# Тест 5: best_frame выбирается по максимальной confidence
# ---------------------------------------------------------------------------
def test_best_frame_selected_by_confidence():
    t = SimpleTracker()
    box = _box(100, 100)

    t.update([box], frame_idx=0, confidences=[0.5])
    t.update([box], frame_idx=1, confidences=[0.95])  # ← лучший кадр
    t.update([box], frame_idx=2, confidences=[0.7])

    t.finalize_all()
    record = t.get_finished_tracks()[0]

    assert record.best_frame == 1, "Лучший кадр — где confidence максимальна"
    assert abs(record.best_confidence - 0.95) < 1e-6


# ---------------------------------------------------------------------------
# Тест 6: reset() очищает всё состояние
# ---------------------------------------------------------------------------
def test_reset_clears_state():
    t = SimpleTracker()
    t.update([_box(0, 0)], frame_idx=0)
    t.finalize_all()

    t.reset()

    assert t.get_finished_tracks() == []
    # После reset следующий трек должен снова начинаться с ID=1
    ids = t.update([_box(0, 0)], frame_idx=0)
    assert ids[0] == 1
