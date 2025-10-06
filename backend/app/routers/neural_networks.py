"""
Neural Networks Router
Handles classification, regression, and time series prediction with neural networks
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from ..db.session import get_db
from ..db.models import Dataset
from ..schemas.neural_networks import (
    ClassificationRequest,
    RegressionRequest,
    TimeSeriesRequest,
    ModelResponse,
    TrainingResponse,
    PredictionResponse
)
from ..services.neural_networks import NeuralNetworkService

router = APIRouter(
    prefix="/neural-networks",
    tags=["Neural Networks"]
)

nn_service = NeuralNetworkService()

@router.post("/classification/train", response_model=TrainingResponse)
async def train_classification_model(
    request: ClassificationRequest,
    db: Session = Depends(get_db)
):
    """
    Entrena un modelo de clasificación con redes neuronales
    """
    try:
        # Obtener dataset
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        # Cargar datos - construir path completo
        data_dir = "backend/data/uploads"
        file_path = os.path.join(data_dir, str(dataset.stored_filename))
        df = pd.read_csv(file_path)
        
        # Entrenar modelo
        result = await nn_service.train_classification_model(
            df=df,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            test_size=request.test_size,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            hidden_layers=request.hidden_layers,
            dropout_rate=request.dropout_rate,
            validation_split=request.validation_split
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")

@router.post("/regression/train", response_model=TrainingResponse)
async def train_regression_model(
    request: RegressionRequest,
    db: Session = Depends(get_db)
):
    """
    Entrena un modelo de regresión con redes neuronales
    """
    try:
        # Obtener dataset
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        # Cargar datos - construir path completo
        data_dir = "backend/data/uploads"
        file_path = os.path.join(data_dir, str(dataset.stored_filename))
        df = pd.read_csv(file_path)
        
        # Entrenar modelo
        result = await nn_service.train_regression_model(
            df=df,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            test_size=request.test_size,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            hidden_layers=request.hidden_layers,
            dropout_rate=request.dropout_rate,
            validation_split=request.validation_split
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")

@router.post("/timeseries/train", response_model=TrainingResponse)
async def train_timeseries_model(
    request: TimeSeriesRequest,
    db: Session = Depends(get_db)
):
    """
    Entrena un modelo de series temporales usando Darts
    """
    try:
        # Obtener dataset
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        # Cargar datos - construir path completo
        data_dir = "backend/data/uploads"
        file_path = os.path.join(data_dir, str(dataset.stored_filename))
        df = pd.read_csv(file_path)
        
        # Entrenar modelo
        result = await nn_service.train_timeseries_model(
            df=df,
            target_column=request.target_column,
            date_column=request.date_column,
            feature_columns=request.feature_columns,
            model_type=request.model_type,
            input_chunk_length=request.input_chunk_length,
            output_chunk_length=request.output_chunk_length,
            n_epochs=request.n_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            hidden_size=request.hidden_size,
            num_layers=request.num_layers,
            dropout=request.dropout
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo de series temporales: {str(e)}")

@router.get("/models", response_model=List[ModelResponse])
async def list_trained_models():
    """
    Lista todos los modelos entrenados
    """
    try:
        models = nn_service.list_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando modelos: {str(e)}")

@router.get("/models/{model_id}")
async def get_model_details(model_id: str):
    """
    Obtiene detalles de un modelo específico
    """
    try:
        model_info = nn_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo información del modelo: {str(e)}")

@router.post("/predict/{model_id}", response_model=PredictionResponse)
async def make_prediction(
    model_id: str,
    data: Dict[str, Any]
):
    """
    Realiza predicciones con un modelo entrenado
    """
    try:
        prediction = await nn_service.predict(model_id, data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error realizando predicción: {str(e)}")

@router.post("/predict/{model_id}/batch")
async def make_batch_prediction(
    model_id: str,
    file: UploadFile = File(...)
):
    """
    Realiza predicciones en lote desde un archivo CSV
    """
    temp_path = f"temp_{file.filename}"
    try:
        # Guardar archivo temporalmente
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Cargar datos
        df = pd.read_csv(temp_path)
        
        # Realizar predicciones
        predictions = await nn_service.batch_predict(model_id, df)
        
        # Limpiar archivo temporal
        os.remove(temp_path)
        
        return {"predictions": predictions}
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error en predicción en lote: {str(e)}")

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Elimina un modelo entrenado
    """
    try:
        success = nn_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        return {"message": "Modelo eliminado exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando modelo: {str(e)}")

@router.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """
    Obtiene métricas detalladas de un modelo
    """
    try:
        metrics = nn_service.get_model_metrics(model_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")

@router.get("/models/{model_id}/plot/{plot_type}")
async def get_model_plot(model_id: str, plot_type: str):
    """
    Genera gráficos de evaluación del modelo
    plot_type: 'training_history', 'confusion_matrix', 'roc_curve', 'predictions'
    """
    try:
        plot_data = nn_service.generate_plot(model_id, plot_type)
        if not plot_data:
            raise HTTPException(status_code=404, detail="Gráfico no disponible")
        return plot_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráfico: {str(e)}")