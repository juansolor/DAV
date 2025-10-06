from fastapi import APIRouter, UploadFile, HTTPException, Depends, Query
import pandas as pd
import io, os, uuid
from typing import Dict
from sqlalchemy.orm import Session
from app.db.session import get_db, engine
from app.db import models
from app.schemas.dataset import DatasetListItem, DatasetOut

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Memoria simple en caliente (temporal)
IN_MEMORY_DATASETS: Dict[str, pd.DataFrame] = {}

# Crear tablas si no existen (simple, se puede mover a inicialización central)
models.Base.metadata.create_all(bind=engine)

DATA_DIR = "backend/data/uploads"
os.makedirs(DATA_DIR, exist_ok=True)

@router.post("/upload", response_model=list[DatasetOut])
async def upload_dataset(file: UploadFile, db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo requerido")
    
    allowed_extensions = (".csv", ".txt", ".xlsx", ".xls")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Formato no soportado. Usa: {', '.join(allowed_extensions)}")
    
    raw = await file.read()
    datasets_created = []
    
    try:
        if file.filename.lower().endswith((".xlsx", ".xls")):
            # Manejar archivos Excel (múltiples hojas)
            excel_file = io.BytesIO(raw)
            excel_data = pd.ExcelFile(excel_file)
            sheet_names = excel_data.sheet_names
            
            for sheet_name in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Saltar hojas vacías
                if df.empty:
                    continue
                
                # Persistir archivo por hoja
                stored_name = f"{uuid.uuid4().hex}.csv"
                path = os.path.join(DATA_DIR, stored_name)
                df.to_csv(path, index=False)
                
                # Nombre único por hoja
                dataset_name = f"{file.filename}:{sheet_name}" if len(sheet_names) > 1 else file.filename
                
                # Guardar en memoria
                IN_MEMORY_DATASETS[dataset_name] = df
                
                dataset = models.Dataset(
                    name=dataset_name,
                    stored_filename=stored_name,
                    original_filename=file.filename,
                    n_rows=len(df),
                    n_cols=len(df.columns),
                    file_size=len(raw) // len(sheet_names)  # Aproximado por hoja
                )
                db.add(dataset)
                datasets_created.append(dataset)
                
        else:
            # Manejar CSV/TXT (comportamiento original)
            df = pd.read_csv(io.BytesIO(raw))
            
            # Persistir archivo en disco
            stored_name = f"{uuid.uuid4().hex}.csv"
            path = os.path.join(DATA_DIR, stored_name)
            with open(path, 'wb') as f:
                f.write(raw)
            
            # Guardar en memoria
            IN_MEMORY_DATASETS[file.filename] = df
            
            dataset = models.Dataset(
                name=file.filename,
                stored_filename=stored_name,
                original_filename=file.filename,
                n_rows=len(df),
                n_cols=len(df.columns),
                file_size=len(raw)
            )
            db.add(dataset)
            datasets_created.append(dataset)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando archivo: {e}")
    
    db.commit()
    for dataset in datasets_created:
        db.refresh(dataset)
    
    return datasets_created

@router.get("/", response_model=list[DatasetListItem])
def list_datasets(db: Session = Depends(get_db)):
    rows = db.query(models.Dataset).order_by(models.Dataset.id.desc()).all()
    return rows

@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    return ds

@router.get("/{dataset_id}/preview")
def dataset_preview(dataset_id: int, n: int = Query(20, le=200), db: Session = Depends(get_db)):
    ds = db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    path = os.path.join(DATA_DIR, str(ds.stored_filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Archivo no disponible")
    try:
        df = pd.read_csv(path, nrows=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo archivo: {e}")
    return {"dataset": ds.name, "preview_rows": df.to_dict(orient="records")}
