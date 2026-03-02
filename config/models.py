"""
Pydantic модели для валидации конфигурации.
Обеспечивают типобезопасность и autocomplete в IDE.
"""

from datetime import time
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class StorageConfig(BaseModel):
    """Конфигурация хранилища файлов"""

    input_root: Path = Field(
        default=Path("./data/raw"), description="Корневая папка с исходными видео"
    )
    output_root: Path = Field(
        default=Path("./data/analysed"), description="Корневая папка для результатов"
    )
    database_path: Path = Field(
        default=Path("./data/surveillance.db"),
        description="Путь к файлу SQLite базы данных",
    )
    screenshots_dir: Path = Field(
        default=Path("./data/screenshots"),
        description="Папка для сохранения скриншотов",
    )

    @field_validator(
        "input_root", "output_root", "database_path", "screenshots_dir", mode="after"
    )
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Автоматически создаем необходимые папки"""
        if v.suffix:  # Это файл
            v.parent.mkdir(parents=True, exist_ok=True)
            # Создаем пустой файл БД если его нет
            if v.suffix == ".db" and not v.exists():
                v.touch()
        else:  # Это папка
            v.mkdir(parents=True, exist_ok=True)
        return v


class DetectionConfig(BaseModel):
    """Конфигурация детекции YOLO"""

    model: str = Field(
        default="yolo11n.pt",
        description="Путь к модели YOLO (будет скачана если не найдена)",
    )
    conf_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Порог уверенности детекции"
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Устройство для инференса (cpu/cuda)"
    )
    classes: List[int] = Field(
        default=[2, 5, 7],
        description="ID классов для детекции (2:car, 5:bus, 7:truck из COCO)",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Проверяем что модель существует или будет скачана"""
        # YOLO автоматически скачает предобученные модели
        if not v.startswith("yolov"):
            model_path = Path(v)
            if not model_path.exists():
                raise ValueError(f"Файл модели не найден: {model_path}")
        return v


class TrackingConfig(BaseModel):
    """Конфигурация трекинга"""

    method: Literal["bytetrack", "botsort"] = Field(
        default="bytetrack", description="Алгоритм трекинга"
    )
    track_thresh: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Порог для создания нового трека"
    )
    match_thresh: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Порог сопоставления треков"
    )
    track_buffer: int = Field(
        default=60, ge=1, description="Сколько кадров держать трек при потере объекта"
    )


class OCRConfig(BaseModel):
    """Конфигурация распознавания номеров"""

    enable: bool = Field(default=True, description="Включить распознавание номеров")
    engine: Literal["paddle", "easyocr"] = Field(
        default="paddle", description="Движок OCR"
    )
    languages: List[str] = Field(
        default=["ru", "en"], description="Языки для распознавания"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Минимальная уверенность для сохранения номера",
    )
    run_interval: int = Field(
        default=30, ge=1, description="Запускать OCR каждый N-й кадр"
    )


class ReIDConfig(BaseModel):
    """Конфигурация ReID (цифровые слепки)"""

    enable: bool = Field(default=True, description="Включить создание цифровых слепков")
    model: str = Field(
        default="osnet_x0_25", description="Модель для извлечения признаков"
    )
    similarity_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Порог похожести для повторяемости"
    )
    feature_dim: int = Field(default=512, description="Размерность вектора признаков")


class BehavioralRulesConfig(BaseModel):
    """Конфигурация правил поведенческого анализа"""

    follow_time_minutes: float = Field(
        default=5.0, ge=0.1, description="Критерий А: время следования в минутах"
    )
    repeat_count_per_day: int = Field(
        default=3, ge=1, description="Критерий Б: количество появлений за день"
    )
    lookback_days: int = Field(
        default=1,
        ge=1,
        le=30,
        description="За сколько дней анализировать повторяемость",
    )


class ProcessingConfig(BaseModel):
    """Конфигурация обработки видео"""

    frame_skip: int = Field(
        default=2, ge=0, description="Сколько кадров пропускать (0 - обрабатывать все)"
    )
    target_fps: Optional[float] = Field(
        default=None,
        description="Привести видео к этому FPS (None - оставить исходный)",
    )
    batch_size: int = Field(
        default=1, ge=1, le=4, description="Сколько видео обрабатывать одновременно"
    )


class APIConfig(BaseModel):
    """Конфигурация API сервера"""

    host: str = Field(default="0.0.0.0", description="Хост для привязки сервера")
    port: int = Field(default=8000, ge=1024, le=65535, description="Порт для сервера")
    workers: int = Field(default=1, ge=1, le=8, description="Количество воркеров")
    reload: bool = Field(
        default=True,
        description="Автоматическая перезагрузка при изменении кода (только для dev)",
    )


class AppConfig(BaseModel):
    """
    Главная конфигурация приложения.
    Объединяет все подконфигурации.
    """

    # Основные настройки
    app_name: str = Field(
        default="ShadowCheck",
        validation_alias="app.name",
        description="Название приложения",
    )
    debug: bool = Field(
        default=False, validation_alias="app.debug", description="Режим отладки"
    )
    version: str = Field(
        default="0.1.0", validation_alias="app.version", description="Версия приложения"
    )

    # Подконфигурации
    storage: StorageConfig = Field(default_factory=StorageConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    behavioral_rules: BehavioralRulesConfig = Field(
        default_factory=BehavioralRulesConfig
    )
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    class Config:
        # Разрешаем использовать точки в алиасах
        populate_by_name = True
        # Дополнительная валидация
        validate_assignment = True
