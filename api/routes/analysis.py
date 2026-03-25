"""
Роуты для запуска анализа видео и получения статуса задач.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import AnalyzeIn, CarOut, TaskOut
from api.tasks import task_manager
from database.db_manager import DatabaseManager

router = APIRouter(prefix="/api", tags=["analysis"])


def get_db() -> DatabaseManager:
    from config import config

    db = DatabaseManager(config.storage.database_path)
    db.create_tables()
    return db


@router.post("/analyze", response_model=TaskOut)
def start_analysis(body: AnalyzeIn):
    """Запускает обработку видео в фоновом потоке.

    Возвращает task_id для последующего опроса статуса.
    Видео обрабатывается асинхронно — ответ приходит сразу,
    не ждя окончания обработки.
    """
    task = task_manager.submit(body.file_path)
    return TaskOut(
        task_id=task.task_id,
        status=task.status,
        file_path=task.file_path,
    )


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task_status(task_id: str):
    """Статус задачи обработки видео."""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return TaskOut(
        task_id=task.task_id,
        status=task.status,
        file_path=task.file_path,
        message=task.message,
    )


@router.get("/vehicles", response_model=list[CarOut])
def list_suspicious_vehicles():
    """Список подозрительных автомобилей."""
    db = get_db()
    return db.get_suspicious_cars()
