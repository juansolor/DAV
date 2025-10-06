from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.db.session import Base

class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    stored_filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    n_rows = Column(Integer)
    n_cols = Column(Integer)
    file_size = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
