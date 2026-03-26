"""
FastAPI приложение — точка сборки всех роутов.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.analysis import router as analysis_router
from api.routes.incidents import router as incidents_router
from api.routes.static import router as static_router

app = FastAPI(
    title="ShadowCheck API",
    description="Система видеоаналитики для выявления автомобилей слежки",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Раздаём папку web/static/ по урлу /static/
# Именно отсюда браузер подтянет style.css
WEB_DIR = Path(__file__).parent.parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

app.include_router(incidents_router)
app.include_router(analysis_router)
app.include_router(static_router)  # "/" — последним, иначе перехватит /api/


@app.get("/health")
def health():
    return {"status": "ok"}
