"""
SQLAlchemy ORM-модели для базы данных ShadowCheck.

Каждый класс здесь — это таблица в SQLite.
Связи между таблицами описаны через relationship() — это позволяет
обращаться к связанным объектам как к атрибутам Python:
    track.car.license_plate  ← вместо отдельного JOIN-запроса
    video.tracks             ← список всех треков видео
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declared_attr,
    mapped_column,
    relationship,
)
from sympy import N


class Base(DeclarativeBase):
    """
    Все классы-наследники автоматически регистрируются в метаданных
    и будут созданы при вызове Base.metadata.create_all(engine).
    """

    __abstract__ = True

    @declared_attr.directive
    def __tablename__(cls):
        return cls.__name__.lower() + "s"


class Video(Base):
    """Одна запись = одно обработанное видео."""

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    camera_id: Mapped[str | None] = mapped_column(String(100))
    company_car_id: Mapped[str | None] = mapped_column(String(100))
    recording_date: Mapped[datetime | None] = mapped_column()
    processed_date: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    frame_width: Mapped[int | None] = mapped_column(Integer)
    frame_height: Mapped[int | None] = mapped_column(Integer)
    analysed_video_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="processing")

    tracks: Mapped[list["Track"]] = relationship("Track", back_populates="video")

    __table_args__ = (Index("idx_videos_date", "recording_date"),)


class Car(Base):
    """
    Уникальный автомобиль в системе.
    Один автомобиль может встречаться в РАЗНЫХ видео — поэтому
    cars отдельная таблица, а не часть tracks.
    """

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    license_plate: Mapped[str | None] = mapped_column(String(20))
    plate_confidence: Mapped[float | None] = mapped_column(Float)
    plate_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    first_seen: Mapped[datetime | None] = mapped_column()
    last_seen: Mapped[datetime | None] = mapped_column()
    total_sightings: Mapped[int] = mapped_column(Integer, default=1)
    is_suspicious: Mapped[bool] = mapped_column(Boolean, default=False)
    suspicious_reason: Mapped[str | None] = mapped_column(String(50))
    notes: Mapped[str | None] = mapped_column(Text)

    tracks: Mapped[list["Track"]] = relationship("Track", back_populates="car")
    embeddings: Mapped[list["Embedding"]] = relationship(
        "Embedding", back_populates="car", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_cars_license_plate", "license_plate"),
        Index("idx_cars_suspicious", "is_suspicious"),
        Index("idx_cars_last_seen", "last_seen"),
    )


class Track(Base):
    """
    Один трек = одно непрерывное появление автомобиля в одном видео.

    Критерий А считается через: end_time_seconds - start_time_seconds
    bbox_history хранится как JSON-строка (для MVP быстрее отдельной таблицы).
    """

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    car_id: Mapped[int] = mapped_column(ForeignKey("cars.id", ondelete="CASCADE"))
    track_id_in_video: Mapped[int | None] = mapped_column(Integer)
    start_frame: Mapped[int | None] = mapped_column(Integer)
    end_frame: Mapped[int | None] = mapped_column(Integer)
    start_time_seconds: Mapped[float | None] = mapped_column(Float)
    end_time_seconds: Mapped[float | None] = mapped_column(Float)
    bbox_history: Mapped[str | None] = mapped_column(Text)  # JSON
    best_frame_number: Mapped[int | None] = mapped_column(Integer)
    best_bbox: Mapped[str | None] = mapped_column(Text)  # JSON [x1,y1,x2,y2]
    confidence_avg: Mapped[float | None] = mapped_column(Float)
    suspicious_by_criteria_a: Mapped[bool] = mapped_column(Boolean, default=False)
    suspicious_by_criteria_b: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    video: Mapped["Video"] = relationship("Video", back_populates="tracks")
    car: Mapped["Car"] = relationship("Car", back_populates="tracks")
    incident: Mapped[Incident | None] = relationship(
        "Incident", back_populates="track", uselist=False
    )
    embeddings: Mapped[list["Embedding"]] = relationship(
        "Embedding", back_populates="source_track"
    )

    __table_args__ = (
        Index("idx_tracks_car_id", "car_id"),
        Index("idx_tracks_video_id", "video_id"),
        Index("idx_tracks_suspicious_a", "suspicious_by_criteria_a"),
        Index("idx_tracks_suspicious_b", "suspicious_by_criteria_b"),
    )

    @property
    def bbox_history_list(self) -> list:
        """Десериализует bbox_history из JSON в Python-список."""
        return json.loads(self.bbox_history) if self.bbox_history else []

    @bbox_history_list.setter
    def bbox_history_list(self, value: list) -> None:
        self.bbox_history = json.dumps(value)

    @property
    def duration_seconds(self) -> float | None:
        """Длительность трека — для проверки критерия А."""
        if self.start_time_seconds is not None and self.end_time_seconds is not None:
            return self.end_time_seconds - self.start_time_seconds
        return None


class Embedding(Base):
    """
    512-мерный вектор внешнего вида автомобиля (numpy float32 как BLOB).
    Нужен для критерия Б когда номер не распознан:
    ищем похожие вектора через косинусное расстояние.
    """

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    car_id: Mapped[int] = mapped_column(ForeignKey("cars.id", ondelete="CASCADE"))
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    source_track_id: Mapped[int | None] = mapped_column(
        ForeignKey("tracks.id", ondelete="SET NULL")
    )
    source_frame_number: Mapped[int | None] = mapped_column(Integer)
    quality: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    car: Mapped["Car"] = relationship("Car", back_populates="embeddings")
    source_track: Mapped[Track | None] = relationship(
        "Track", back_populates="embeddings"
    )

    __table_args__ = (Index("idx_embeddings_car_id", "car_id"),)


class Incident(Base):
    """
    Создаётся когда трек нарушает критерий А или Б.
    Один трек → максимум один инцидент (unique=True на track_id).
    """

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), unique=True
    )
    incident_type: Mapped[str] = mapped_column(String(30), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    severity: Mapped[int] = mapped_column(Integer, default=1)  # 1–5
    screenshot_path: Mapped[str | None] = mapped_column(Text)
    video_segment_path: Mapped[str | None] = mapped_column(Text)
