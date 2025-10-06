from fastapi import APIRouter, Query, Depends
from typing import List, Dict, Any
import os, uuid
from sqlalchemy.orm import Session
from app.services.kaggle_loader import download_dataset
from app.db.session import get_db
from app.db import models
from app.schemas.dataset import DatasetOut
from .datasets import IN_MEMORY_DATASETS

router = APIRouter(prefix="/external", tags=["external"])

DATA_DIR = "backend/data/uploads"
os.makedirs(DATA_DIR, exist_ok=True)

@router.post("/kaggle/import", response_model=List[DatasetOut])
async def import_kaggle(dataset: str = Query(..., description="Formato: owner/dataset-name"), db: Session = Depends(get_db)):
    frames = download_dataset(dataset)
    loaded = []
    
    for df in frames:
        kaggle_filename = getattr(df, '_kaggle_filename', 'kaggle.csv')
        name = f"{dataset}:{kaggle_filename}"
        
        # Persistir archivo en disco
        stored_name = f"kaggle_{uuid.uuid4().hex}.csv"
        path = os.path.join(DATA_DIR, stored_name)
        df.to_csv(path, index=False)
        
        # Guardar en memoria para análisis rápido
        IN_MEMORY_DATASETS[name] = df
        
        # Persistir metadata en DB
        dataset_record = models.Dataset(
            name=name,
            stored_filename=stored_name,
            original_filename=kaggle_filename,
            n_rows=len(df),
            n_cols=len(df.columns),
            file_size=os.path.getsize(path)
        )
        db.add(dataset_record)
        db.commit()
        db.refresh(dataset_record)
        loaded.append(dataset_record)
    
    return loaded
