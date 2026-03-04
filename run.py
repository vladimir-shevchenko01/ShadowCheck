#!/usr/bin/env python
# run.py
"""
Точка входа для запуска обработки видео.
Использование:
    python run.py                     # тестовое видео
    python run.py video.mp4            # конкретный файл
    python run.py /path/to/folder/     # все видео в папке
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from config import config, get_logger, setup_logging
from core.pipeline.video_processor import VideoProcessor
from core.utils import timing


@timing.timer
def main() -> int:
    """Главная функция."""

    # Настраиваем логирование
    setup_logging(
        log_level="DEBUG" if config.debug else "INFO", enable_file_logging=True
    )

    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("🚀 ShadowCheck - Система видеоаналитики")
    logger.info("=" * 60)

    # Создаем процессор
    processor = VideoProcessor()

    # Определяем что обрабатывать
    if len(sys.argv) > 1:
        target: Path = Path(sys.argv[1])
    else:
        target: Path = config.storage.input_root / "test_video.mp4"
        if not target.exists():
            logger.error("❌ Тестовое видео не найдено!")
            logger.info(f"Положите видео в {target}")
            logger.info("\nИли укажите путь к видео:")
            logger.info("  python run.py /путь/к/видео.mp4")
            logger.info("  python run.py /путь/к/папке/")
            return 1

    # Обрабатываем
    try:
        if target.is_file():
            result: Path = processor.process_video(target)
            logger.info(f"✅ Готово: {result}")
        elif target.is_dir():
            results: List[Path] = processor.process_folder(target)
            logger.info(f"✅ Обработано {len(results)} видео")
        else:
            logger.error(f"❌ Неверный путь: {target}")
            return 1
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return 1

    logger.info("=" * 60)
    logger.info("✅ Работа завершена успешно!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
