import numpy as np

from core.pipeline.video_processor import VideoProcessor
from core.tracking.simple_tracker import SimpleTracker


def test_simple_tracker_basic():
    tracker = SimpleTracker(match_thresh=0.5, track_buffer=2)
    # два независимых бокса
    boxes1 = [[0, 0, 10, 10], [100, 100, 110, 110]]
    ids1 = tracker.update(boxes1)
    assert ids1 == [1, 2]

    # немного сместились, должны получить те же ID
    boxes2 = [[1, 1, 11, 11], [102, 100, 112, 110]]
    ids2 = tracker.update(boxes2)
    assert ids2 == ids1

    # один объект исчез
    boxes3 = [[2, 2, 12, 12]]
    ids3 = tracker.update(boxes3)
    assert ids3 == [1]

    # после ряда пустых кадров трек 2 будет сброшен
    for _ in range(3):
        tracker.update([])
    boxes4 = [[200, 200, 210, 210]]
    ids4 = tracker.update(boxes4)
    assert ids4[0] != 2


def test_video_processor_skip_logic(monkeypatch):
    # создаем фиктивный детектор, который возвращает рамку каждую третью
    class DummyDetector:
        def __init__(self):
            self.calls = 0

        def detect_with_boxes(self, frame):
            idx = self.calls
            self.calls += 1
            if idx % 3 == 0:
                # одна рамка в левом верхнем углу
                return [[0, 0, 20, 20]], [0.9], [2]
            return [], [], []

    processor = VideoProcessor()
    processor.detector = DummyDetector()
    # сброс tracker чтобы он не держал старые данные
    processor.tracker = SimpleTracker(match_thresh=0.5, track_buffer=5)

    # шесть кадров чёрного цвета
    frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(6)]
    annotated = []
    for i, f in enumerate(frames):
        aframe, used = processor._annotate_frame(f, i, fps=1.0)
        annotated.append(aframe)

    # в изначальном варианте боксы рисовались только на кадрах 0 и 3
    # после исправления они должны рисоваться на всех кадрах начиная с 0
    for i, af in enumerate(annotated):
        # цвет рамки (зелёный) имеет ненулевые значения в канале G
        assert af[:, :, 1].sum() > 0, f"кадр {i} не размечен"
