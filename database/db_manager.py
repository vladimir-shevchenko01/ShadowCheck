"""
DatabaseManager — единственная точка входа для работы с БД.

Принцип: весь SQL/ORM-код живёт здесь. Остальные модули (pipeline,
behavioral analysis) вызывают методы этого класса и не знают ничего
про SQLAlchemy или SQLite.

Паттерн: Repository + Unit of Work (сессия как контекстный менеджер).
"""

from __future__ import annotations

import json
import statistics
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Sequence

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base, Car, Incident, Track, Video


class DatabaseManager:
    """Управляет подключением к БД и предоставляет методы для CRUD-операций."""

    def __init__(self, db_path: Path | str) -> None:
        """
        Args:
            db_path: Путь к файлу SQLite.
                     Используй ":memory:" для тестов — БД в RAM, без файлов.
        """
        db_url = f"sqlite:///{db_path}"

        # check_same_thread=False нужен для SQLite + многопоточность.
        # connect_args передаётся напрямую в драйвер (sqlite3).
        self.engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False,  # echo=True → печатает все SQL-запросы (удобно при отладке)
        )

        # sessionmaker создаёт фабрику сессий.
        # Сессия — это "единица работы": все изменения накапливаются в ней,
        # затем фиксируются через commit() или откатываются через rollback().
        self._SessionFactory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        """Создаёт все таблицы если их нет (безопасно — не удаляет существующие)."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Контекстный менеджер для сессии с автоматическим commit/rollback.

        Использование:
            with db.session() as s:
                s.add(car)
            # ← здесь автоматически commit()

        Если внутри блока возникнет исключение — автоматически rollback().
        """
        s = self._SessionFactory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------

    def create_video(
        self,
        file_path: str,
        filename: str,
        fps: float,
        frame_width: int,
        frame_height: int,
        duration_seconds: float,
        camera_id: str | None = None,
        recording_date: datetime | None = None,
    ) -> int:
        """Создаёт запись о видео со статусом 'processing'. Возвращает video.id."""
        with self.session() as s:
            video = Video(
                file_path=file_path,
                filename=filename,
                fps=fps,
                frame_width=frame_width,
                frame_height=frame_height,
                duration_seconds=duration_seconds,
                camera_id=camera_id,
                recording_date=recording_date,
                status="processing",
            )
            s.add(video)
            s.flush()  # flush() присваивает id без полного commit
            return video.id

    def mark_video_done(self, video_id: int, analysed_path: str | None = None) -> None:
        """Обновляет статус видео на 'processed' после успешной обработки."""
        with self.session() as s:
            video = s.get(Video, video_id)
            if video:
                video.status = "processed"
                video.processed_date = datetime.now()
                if analysed_path:
                    video.analysed_video_path = analysed_path

    def mark_video_error(self, video_id: int) -> None:
        with self.session() as s:
            video = s.get(Video, video_id)
            if video:
                video.status = "error"

    # ------------------------------------------------------------------
    # Car — поиск или создание
    # ------------------------------------------------------------------

    def get_or_create_car(
        self,
        session: Session,
        license_plate: str | None,
        plate_confidence: float | None,
        seen_at: datetime,
    ) -> Car:
        """
        Ищет автомобиль по номеру или создаёт новый.

        Почему принимает session, а не создаёт сам?
        Потому что этот метод вызывается внутри транзакции save_tracks_for_video,
        и нам нужно, чтобы Car и Track были в одной транзакции.
        Если бы мы открывали новую сессию — могли бы получить конфликт.

        Логика:
        - Если номер известен → ищем существующую машину по номеру
        - Если нашли → обновляем last_seen и total_sightings
        - Если не нашли (или номера нет) → создаём новую запись Car
        """
        if license_plate:
            stmt = select(Car).where(Car.license_plate == license_plate)
            existing = session.execute(stmt).scalar_one_or_none()
            if existing:
                existing.last_seen = seen_at
                existing.total_sightings += 1
                # Обновляем уверенность если новая выше
                if plate_confidence and (
                    existing.plate_confidence is None
                    or plate_confidence > existing.plate_confidence
                ):
                    existing.plate_confidence = plate_confidence
                return existing

        # Новый автомобиль
        car = Car(
            license_plate=license_plate,
            plate_confidence=plate_confidence,
            first_seen=seen_at,
            last_seen=seen_at,
        )
        session.add(car)
        session.flush()  # нужен id для последующих FK
        return car

    # ------------------------------------------------------------------
    # Tracks — основной метод сохранения результатов обработки видео
    # ------------------------------------------------------------------

    def save_tracks_for_video(
        self,
        video_id: int,
        track_records: list,  # List[TrackRecord] из simple_tracker
        fps: float,
        recording_dt: datetime | None = None,
    ) -> list[int]:
        """
        Сохраняет все треки видео в БД.

        Args:
            video_id:      id записи videos (уже должна существовать)
            track_records: список TrackRecord из tracker.get_finished_tracks()
            fps:           нужен для конвертации кадров в секунды
            recording_dt:  время съёмки (для first_seen/last_seen автомобиля)

        Returns:
            Список созданных track.id
        """
        seen_at = recording_dt or datetime.now()
        created_ids = []

        with self.session() as s:
            for record in track_records:
                # Конвертируем кадры в секунды
                start_sec = record.start_frame / fps if fps > 0 else 0.0
                end_sec = record.end_frame / fps if fps > 0 else 0.0

                # Средняя уверенность по всем кадрам трека
                avg_conf = (
                    statistics.mean(record.confidence_history)
                    if record.confidence_history
                    else None
                )

                # Пока OCR не реализован — номера нет
                # TODO: после реализации OCR передавать plate сюда
                car = self.get_or_create_car(
                    session=s,
                    license_plate=None,
                    plate_confidence=None,
                    seen_at=seen_at,
                )

                track = Track(
                    video_id=video_id,
                    car_id=car.id,
                    track_id_in_video=record.track_id,
                    start_frame=record.start_frame,
                    end_frame=record.end_frame,
                    start_time_seconds=start_sec,
                    end_time_seconds=end_sec,
                    best_frame_number=record.best_frame,
                    best_bbox=(
                        json.dumps(record.best_bbox) if record.best_bbox else None
                    ),
                    confidence_avg=avg_conf,
                )
                track.bbox_history_list = record.bbox_history

                s.add(track)
                s.flush()
                created_ids.append(track.id)

        return created_ids

    # ------------------------------------------------------------------
    # Behavioral analysis — критерии А и Б
    # ------------------------------------------------------------------

    def apply_criteria_a(self, video_id: int, min_duration_seconds: float) -> list[int]:
        """
        Критерий А: помечает треки где длительность >= порога.
        Возвращает список track.id которые стали подозрительными.
        """
        suspicious_ids = []
        with self.session() as s:
            stmt = select(Track).where(Track.video_id == video_id)
            tracks = s.execute(stmt).scalars().all()

            for track in tracks:
                if (
                    track.duration_seconds is not None
                    and track.duration_seconds >= min_duration_seconds
                ):
                    track.suspicious_by_criteria_a = True

                    # Создаём инцидент если его ещё нет
                    if track.incident is None:
                        incident = Incident(
                            track_id=track.id,
                            incident_type="long_follow",
                            description=(
                                f"Автомобиль в кадре {track.duration_seconds:.1f} сек "
                                f"(порог: {min_duration_seconds:.1f} сек)"
                            ),
                            severity=2,
                            best_frame_number=track.best_frame_number,
                        )
                        s.add(incident)
                    suspicious_ids.append(track.id)

        return suspicious_ids

    def apply_criteria_b(
        self,
        video_id: int,
        min_repeat_count: int,
        lookback_days: int = 1,
    ) -> list[int]:
        """
        Критерий Б: помечает треки принадлежащие машинам,
        встретившимся >= min_repeat_count раз за lookback_days дней.

        Работает только по license_plate (ReID-ветка — в следующих шагах).
        Возвращает список track.id которые стали подозрительными.
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        suspicious_ids = []

        with self.session() as s:
            # Берём все треки за нужный период с известным номером
            stmt = (
                select(Track)
                .join(Car)
                .where(
                    Track.created_at >= cutoff,
                    Car.license_plate.isnot(None),
                )
            )
            tracks = s.execute(stmt).scalars().all()

            # Группируем по номеру
            by_plate: dict[str, list[Track]] = {}
            for t in tracks:
                plate = t.car.license_plate
                if plate is not None:
                    by_plate.setdefault(plate, []).append(t)
                else:
                    # обработайте случай, когда plate равно None
                    # например, можно использовать ключ по умолчанию
                    by_plate["None"].append(t)

            for plate, plate_tracks in by_plate.items():
                if len(plate_tracks) >= min_repeat_count:
                    for t in plate_tracks:
                        t.suspicious_by_criteria_b = True

                        # Обновляем или создаём инцидент
                        if t.incident is None:
                            incident = Incident(
                                track_id=t.id,
                                incident_type="repeat_offender",
                                description=(
                                    f"Номер {plate} появился {len(plate_tracks)} раз "
                                    f"за {lookback_days} дн. (порог: {min_repeat_count})"
                                ),
                                severity=3,
                                license_plate_text=plate,
                                best_frame_number=t.best_frame_number,
                            )
                            s.add(incident)
                        elif t.incident.incident_type == "long_follow":
                            # Уже есть инцидент А → апгрейдим до 'both'
                            t.incident.incident_type = "both"
                            t.incident.severity = 4

                        suspicious_ids.append(t.id)

        return suspicious_ids

    # ------------------------------------------------------------------
    # Queries для API / фронтенда
    # ------------------------------------------------------------------

    def get_recent_incidents(self, limit: int = 50) -> Sequence[Incident]:
        """Возвращает последние инциденты — для главного экрана."""
        with self.session() as s:
            stmt = select(Incident).order_by(Incident.created_at.desc()).limit(limit)
            return s.execute(stmt).scalars().all()

    def get_suspicious_cars(self) -> Sequence[Car]:
        """Все машины с флагом is_suspicious=True."""
        with self.session() as s:
            stmt = select(Car).where(Car.is_suspicious == True)
            return s.execute(stmt).scalars().all()
