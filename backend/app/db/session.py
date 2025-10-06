from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import get_settings
import os

settings = get_settings()
# SQLite inicial; configurable vía ENV (DB_URL)
DB_URL = os.getenv('DB_URL', 'sqlite:///backend/data/app.db')

# Asegurar carpeta
os.makedirs('backend/data', exist_ok=True)

connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
