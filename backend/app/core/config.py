import os
from functools import lru_cache
from typing import List

class Settings:
    APP_NAME: str = "Data Analytics Integrator"
    CORS_ORIGINS: List[str] = ["*"]
    KAGGLE_USERNAME: str | None = None
    KAGGLE_KEY: str | None = None
    DATA_DIR: str = "backend/data/uploads"

    def __init__(self):
        # Cargar variables de entorno manualmente (evita dependencia pydantic-settings)
        self.KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
        self.KAGGLE_KEY = os.getenv("KAGGLE_KEY")
        self.DATA_DIR = os.getenv("DATA_DIR", "backend/data/uploads")
        cors_origins = os.getenv("CORS_ORIGINS", "*")
        self.CORS_ORIGINS = [o.strip() for o in cors_origins.split(",")]

@lru_cache
def get_settings() -> Settings:
    return Settings()
