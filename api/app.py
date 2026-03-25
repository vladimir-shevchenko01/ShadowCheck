"""
FastAPI приложение — точка сборки всех роутов.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analysis import router as analysis_router
from api.routes.incidents import router as incidents_router

app = FastAPI(
    title="ShadowCheck API",
    description="Система видеоаналитики для выявления автомобилей слежки",
    version="0.1.0",
)

# CORS — разрешаем фронтенду (будет на том же хосте) обращаться к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для MVP; в продакшне указать конкретный origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(incidents_router)
app.include_router(analysis_router)


@app.get("/health")
def health():
    """Проверка что сервер жив — удобно для Docker healthcheck."""
    return {"status": "ok"}
