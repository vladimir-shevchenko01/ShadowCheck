"""
Роуты для работы с инцидентами.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import IncidentDetailOut, IncidentOut, OkOut, ReviewIn
from database.db_manager import DatabaseManager
from database.models import Incident

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


def get_db() -> DatabaseManager:
    """Dependency Injection для DatabaseManager.

    FastAPI вызывает эту функцию для каждого запроса.
    Позволяет легко подменить БД в тестах через app.dependency_overrides.
    """
    from config import config

    db = DatabaseManager(config.storage.database_path)
    db.create_tables()
    return db


@router.get("", response_model=list[IncidentOut])
def list_incidents(
    limit: int = 50,
    unreviewed_only: bool = False,
    db: DatabaseManager = Depends(get_db),
):
    """Список последних инцидентов для главного экрана."""
    incidents = db.get_recent_incidents(limit=limit)
    if unreviewed_only:
        incidents = [i for i in incidents if not i.reviewed]
    return incidents


@router.get("/{incident_id}", response_model=IncidentDetailOut)
def get_incident(incident_id: int, db: DatabaseManager = Depends(get_db)):
    """Детали одного инцидента."""
    with db.session() as s:
        incident = s.get(Incident, incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="Инцидент не найден")
        # Считываем все поля пока сессия открыта
        return IncidentDetailOut.model_validate(incident)


@router.post("/{incident_id}/review", response_model=OkOut)
def review_incident(
    incident_id: int,
    body: ReviewIn,
    db: DatabaseManager = Depends(get_db),
):
    """Оператор помечает инцидент как просмотренный."""
    with db.session() as s:
        incident = s.get(Incident, incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="Инцидент не найден")
        incident.reviewed = True
        incident.reviewed_by = body.reviewed_by
        incident.reviewed_at = datetime.now()
        if body.notes:
            incident.notes = body.notes
    return OkOut()
