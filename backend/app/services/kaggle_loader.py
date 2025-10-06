import os
import zipfile
import io
import pandas as pd
from typing import List, Dict, Any
from fastapi import HTTPException
from app.core.config import get_settings

try:
    from kaggle import api  # type: ignore
    _KAGGLE_AVAILABLE = True
except Exception:  # pragma: no cover
    _KAGGLE_AVAILABLE = False


def ensure_kaggle_credentials():
    settings = get_settings()
    if settings.KAGGLE_USERNAME and settings.KAGGLE_KEY:
        os.environ['KAGGLE_USERNAME'] = settings.KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = settings.KAGGLE_KEY
    # Kaggle lib también soporta ~/.kaggle/kaggle.json


def download_dataset(dataset: str) -> List[pd.DataFrame]:
    """Descarga un dataset público de Kaggle.
    dataset: 'owner/dataset-name'
    Retorna lista de DataFrames (uno por archivo CSV)
    """
    if not _KAGGLE_AVAILABLE:
        raise HTTPException(status_code=500, detail="Librería kaggle no instalada. Ejecuta: pip install kaggle")

    ensure_kaggle_credentials()
    try:
        from kaggle import api  # Import here to ensure it's available
        content = api.dataset_download_files(dataset, quiet=True)
        if content is None:
            raise HTTPException(status_code=400, detail="No se pudo descargar el dataset")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Error descargando dataset Kaggle: {e}")

    z = zipfile.ZipFile(io.BytesIO(content))
    frames: List[pd.DataFrame] = []
    for name in z.namelist():
        if name.lower().endswith('.csv'):
            with z.open(name) as f:
                try:
                    df = pd.read_csv(f)
                except Exception:
                    continue
                df._kaggle_filename = name  # type: ignore
                frames.append(df)
    if not frames:
        raise HTTPException(status_code=404, detail="No se encontraron CSVs en el dataset")
    return frames
