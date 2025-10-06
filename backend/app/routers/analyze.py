from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db import models
from .datasets import IN_MEMORY_DATASETS
import os

router = APIRouter(prefix="/analyze", tags=["analysis"])

class BasicSummaryResponse(BaseModel):
    dataset: str
    columns: List[str]
    summary: Dict[str, Dict[str, float]]

class CorrelationResponse(BaseModel):
    dataset: str
    correlation_matrix: Dict[str, Dict[str, float]]
    high_correlations: List[Dict[str, Any]]
    method: str
    data_types_used: List[str]
    columns_analyzed: List[str]

class MissingDataResponse(BaseModel):
    dataset: str
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    total_rows: int

def _load_dataset_by_id(dataset_id: int, db: Session) -> tuple[pd.DataFrame, models.Dataset]:
    """Helper para cargar dataset desde DB por ID"""
    ds = db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    
    # Intentar primero memoria (más rápido)
    dataset_name = str(ds.name)
    if dataset_name in IN_MEMORY_DATASETS:
        return IN_MEMORY_DATASETS[dataset_name], ds
    
    # Cargar desde disco
    stored_filename = str(ds.stored_filename)
    path = os.path.join("backend/data/uploads", stored_filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Archivo no disponible")
    try:
        df = pd.read_csv(path)
        # Cache en memoria para próximas consultas
        IN_MEMORY_DATASETS[dataset_name] = df
        return df, ds
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo archivo: {e}")

@router.get("/basic/{dataset_id}", response_model=BasicSummaryResponse)
async def basic_summary(dataset_id: int, db: Session = Depends(get_db)):
    df, ds = _load_dataset_by_id(dataset_id, db)
    
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise HTTPException(status_code=400, detail="No hay columnas numéricas para analizar")
    
    # Convertir a dict con strings como keys
    summary = {}
    describe_dict = numeric_df.describe().to_dict()
    for col, stats in describe_dict.items():
        summary[str(col)] = {str(k): float(v) for k, v in stats.items()}
    
    return BasicSummaryResponse(dataset=str(ds.name), columns=list(df.columns), summary=summary)

@router.get("/correlation/{dataset_id}", response_model=CorrelationResponse)
async def correlation_analysis(
    dataset_id: int, 
    threshold: float = Query(0.7, ge=0.0, le=1.0),
    method: str = Query("pearson", description="Método de correlación: pearson, spearman, kendall"),
    include_categorical: bool = Query(False, description="Incluir variables categóricas (codificadas)"),
    db: Session = Depends(get_db)
):
    df, ds = _load_dataset_by_id(dataset_id, db)
    
    # Validar método
    valid_methods = ["pearson", "spearman", "kendall"]
    if method not in valid_methods:
        raise HTTPException(status_code=400, detail=f"Método no válido. Usar: {', '.join(valid_methods)}")
    
    # Preparar datos según configuración
    if include_categorical:
        # Incluir categóricas codificando como números
        analysis_df = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Codificar categóricas con LabelEncoder simple
        for col in categorical_cols:
            try:
                # Crear mapeo único para cada valor categórico
                unique_vals = analysis_df[col].dropna().unique()
                val_map = {val: i for i, val in enumerate(unique_vals)}
                analysis_df[col] = analysis_df[col].map(val_map)
            except Exception:
                continue
        
        # Tomar solo columnas numéricas después de codificación
        analysis_df = analysis_df.select_dtypes(include="number")
        data_types_used = ["numeric", "categorical_encoded"]
    else:
        # Solo columnas numéricas originales
        analysis_df = df.select_dtypes(include="number")
        data_types_used = ["numeric"]
    
    if len(analysis_df.columns) < 2:
        raise HTTPException(
            status_code=400, 
            detail=f"Se necesitan al menos 2 columnas para análisis. Datos disponibles: {len(analysis_df.columns)}"
        )
    
    # Calcular correlación con método especificado
    try:
        if method == "pearson":
            corr_matrix = analysis_df.corr(method="pearson")
        elif method == "spearman":
            corr_matrix = analysis_df.corr(method="spearman")
        elif method == "kendall":
            corr_matrix = analysis_df.corr(method="kendall")
        else:
            corr_matrix = analysis_df.corr(method="pearson")  # fallback
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculando correlación {method}: {e}")
    
    # Convertir a dict con claves string
    corr_dict = {}
    for col in corr_matrix.columns:
        corr_dict[str(col)] = {str(k): float(v) if not pd.isna(v) else None for k, v in corr_matrix[col].items()}
    
    # Encontrar correlaciones altas
    high_corrs = []
    try:
        corr_values = corr_matrix.values
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_values[i, j]
                if not np.isnan(val) and abs(val) >= threshold:
                    high_corrs.append({
                        "var1": str(cols[i]),
                        "var2": str(cols[j]),
                        "correlation": round(float(val), 4),
                        "strength": "strong" if abs(val) >= 0.8 else "moderate"
                    })
    except Exception:
        pass
    
    return CorrelationResponse(
        dataset=str(ds.name),
        correlation_matrix=corr_dict,
        high_correlations=high_corrs,
        method=method,
        data_types_used=data_types_used,
        columns_analyzed=list(analysis_df.columns)
    )

@router.get("/missing/{dataset_id}", response_model=MissingDataResponse)
async def missing_data_analysis(dataset_id: int, db: Session = Depends(get_db)):
    df, ds = _load_dataset_by_id(dataset_id, db)
    
    missing_counts = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
    total_rows = len(df)
    missing_percentages = {col: round((count / total_rows) * 100, 2) for col, count in missing_counts.items()}
    
    return MissingDataResponse(
        dataset=str(ds.name),
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        total_rows=total_rows
    )
