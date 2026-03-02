"""
Модуль для обработки видео: чтение, детекция, отрисовка, сохранение.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2

from config import config, get_logger
from core.detection.yolo_detector import YOLODetector
from core.pipeline.frame_handler import draw_detections, draw_timestamp

logger = get_logger(__name__)


class VideoProcessor:
    """
    Класс для обработки видеофайлов.
    Использует конфигурацию для настройки параметров обработки.
    """

    def __init__(self) -> None:
        """Инициализация процессора видео."""
        self.detector = YOLODetector()

        # Загружаем настройки обработки
        self.frame_skip: int = config.processing.frame_skip
        self.target_fps: float | None = config.processing.target_fps

        logger.info("Видеопроцессор инициализирован")
        logger.info(f"  Frame skip: {self.frame_skip}")
        logger.info(f"  Target FPS: {self.target_fps}")

    def process_video(
        self,
        input_path: Path,
        output_path: Path | None = None,
    ) -> Path:
        """
        Обрабатывает одно видео.

        Args:
            input_path: Путь к исходному видео.
            output_path: Путь для сохранения результата
                (если None, генерируется автоматически).

        Returns:
            Путь к обработанному видео.
        """
        logger.info(f"🎬 Начинаем обработку видео: {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"Видео не найдено: {input_path}")

        if output_path is None:
            output_path = (
                config.storage.output_root
                / f"{input_path.stem}_annotated{input_path.suffix}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        logger.info("📊 Параметры видео:")
        logger.info(f"  Размер: {width}x{height}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Кадров: {total_frames}")
        logger.info(f"  Длительность: {duration:.2f} сек")

        process_fps = self.target_fps or fps

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, process_fps, (width, height))

        if not out.isOpened():
            raise RuntimeError(f"Не удалось создать выходное видео: {output_path}")

        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (self.frame_skip + 1) == 0:
                boxes, confs, cls_ids = self.detector.detect_with_boxes(frame)
                annotated_frame = draw_detections(frame, boxes, confs, cls_ids)
                annotated_frame = draw_timestamp(annotated_frame, frame_count / fps)
                processed_count += 1
            else:
                annotated_frame = draw_timestamp(frame, frame_count / fps)

            out.write(annotated_frame)
            frame_count += 1

            if fps > 0 and frame_count % (fps * 10) == 0:
                progress = (frame_count / total_frames) * 100 if total_frames else 0.0
                logger.info(
                    f"⏳ Прогресс: {frame_count}/{total_frames} ({progress:.1f}%)"
                )

        cap.release()
        out.release()

        logger.info("✅ Обработка завершена!")
        logger.info(f"  Всего кадров: {frame_count}")
        logger.info(f"  Обработано детекцией: {processed_count}")
        logger.info(f"  Результат: {output_path}")

        return output_path

    def process_folder(self, folder_path: Path) -> list[Path]:
        """
        Обрабатывает все видео в папке.

        Args:
            folder_path: Путь к папке с видео.

        Returns:
            Список путей к обработанным видео.
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")

        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        video_files: list[Path] = []

        for ext in video_extensions:
            video_files.extend(folder_path.glob(f"*{ext}"))
            video_files.extend(folder_path.glob(f"*{ext.upper()}"))

        logger.info(f"Найдено видео в папке {folder_path}: {len(video_files)}")

        results: list[Path] = []
        for video_file in video_files:
            try:
                output = self.process_video(video_file)
                results.append(output)
            except Exception as e:
                logger.error(f"Ошибка обработки {video_file}: {e}")

        return results


# Для вызова из коммандной строки и тестирования
def main() -> None:
    """Точка входа для тестирования."""
    import sys

    logger.info("=" * 60)
    logger.info("🚀 ShadowCheck Video Processor")
    logger.info("=" * 60)

    processor = VideoProcessor()

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_file():
            processor.process_video(path)
        elif path.is_dir():
            processor.process_folder(path)
        else:
            logger.error(f"Неверный путь: {path}")
    else:
        test_video = config.storage.input_root / "test_video.mp4"

        if test_video.exists():
            processor.process_video(test_video)
        else:
            logger.warning(
                "Тестовое видео не найдено. " "Положите видео в data/raw/test_video.mp4"
            )
            logger.info("Создана структура папок:")
            logger.info(f"  📁 {config.storage.input_root}")
            logger.info(f"  📁 {config.storage.output_root}")
            logger.info(f"  📁 {config.storage.screenshots_dir}")
            logger.info(f"  🗄️  {config.storage.database_path}")


if __name__ == "__main__":
    main()
