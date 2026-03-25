"""
Pydantic-схемы для API.

Зачем отдельные схемы, а не возвращать ORM-модели напрямую?
- ORM-объекты "живут" только внутри сессии БД. За её пределами
  обращение к lazy-loaded полям вызовет ошибку.
- Схемы — это чистые данные без зависимости от SQLAlchemy.
- Позволяет явно контролировать что отдаём клиенту (не светим лишнее).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------


class IncidentOut(BaseModel):
    """Инцидент для списка на главном экране."""

    id: int
    track_id: int
    incident_type: str  # 'long_follow', 'repeat_offender', 'both'
    description: str | None
    severity: int  # 1–5
    license_plate_text: str | None
    best_frame_number: int | None
    screenshot_path: str | None
    reviewed: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class IncidentDetailOut(IncidentOut):
    """Детали инцидента — для страницы просмотра."""

    reviewed_by: str | None
    reviewed_at: datetime | None
    notes: str | None
    video_segment_path: str | None


class ReviewIn(BaseModel):
    """Тело запроса при отметке инцидента просмотренным."""

    reviewed_by: str
    notes: str | None = None


# ---------------------------------------------------------------------------
# Vehicles
# ---------------------------------------------------------------------------


class CarOut(BaseModel):
    """Автомобиль из списка подозрительных."""

    id: int
    license_plate: str | None
    plate_confidence: float | None
    first_seen: datetime | None
    last_seen: datetime | None
    total_sightings: int
    is_suspicious: bool
    suspicious_reason: str | None
    notes: str | None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Analysis tasks
# ---------------------------------------------------------------------------


class AnalyzeIn(BaseModel):
    """Тело запроса на запуск обработки видео."""

    file_path: str  # абсолютный путь к видео на сервере


class TaskOut(BaseModel):
    """Статус фоновой задачи обработки."""

    task_id: str
    status: str  # 'pending', 'running', 'done', 'error'
    file_path: str
    message: str | None = None


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


class OkOut(BaseModel):
    ok: bool = True
