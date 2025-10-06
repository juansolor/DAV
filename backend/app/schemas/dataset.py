from pydantic import BaseModel
from datetime import datetime

class DatasetOut(BaseModel):
    id: int
    name: str
    n_rows: int | None = None
    n_cols: int | None = None
    file_size: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True

class DatasetListItem(BaseModel):
    id: int
    name: str
    n_rows: int | None
    n_cols: int | None

    class Config:
        from_attributes = True
