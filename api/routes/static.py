"""
Раздача статических файлов и HTML-страниц.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["frontend"])

# Путь к папке web/ относительно корня проекта
WEB_DIR = Path(__file__).parent.parent.parent / "web"


@router.get("/", response_class=FileResponse)
def index():
    """Главная страница — список инцидентов."""
    return FileResponse(WEB_DIR / "templates" / "index.html")
