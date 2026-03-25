"""
Простой in-memory менеджер фоновых задач.

Почему не Celery?
Blueprint упоминает Celery как опцию, но для MVP он избыточен —
нужен Redis, воркеры, мониторинг. Для начала достаточно запускать
обработку в отдельном потоке и хранить статус в словаре.
Заменить на Celery можно позже, не меняя API.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

Status = Literal["pending", "running", "done", "error"]


@dataclass
class Task:
    task_id: str
    file_path: str
    status: Status = "pending"
    message: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


class TaskManager:
    """Хранит задачи и запускает обработку в фоновом потоке."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def submit(self, file_path: str) -> Task:
        """Создаёт задачу и запускает обработку в фоне."""
        task_id = str(uuid.uuid4())
        task = Task(task_id=task_id, file_path=file_path)

        with self._lock:
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._run,
            args=(task_id, file_path),
            daemon=True,
        )
        thread.start()
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def _run(self, task_id: str, file_path: str) -> None:
        """Выполняется в отдельном потоке."""
        # Импортируем здесь чтобы избежать circular imports
        from core.pipeline.video_processor import VideoProcessor

        self._set_status(task_id, "running")
        try:
            processor = VideoProcessor()
            processor.process_video(Path(file_path))
            self._set_status(task_id, "done")
        except Exception as e:
            self._set_status(task_id, "error", message=str(e))

    def _set_status(
        self, task_id: str, status: Status, message: str | None = None
    ) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = status
                self._tasks[task_id].message = message


# Глобальный синглтон — создаётся один раз при старте приложения
task_manager = TaskManager()
