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

@router.get("/test")
async def test_endpoint():
    """Endpoint de prueba para verificar que el router funciona"""
    return {"message": "Neural Networks router funcionando correctamente", "status": "ok"}

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

# Nuevos endpoints para análisis visual

from ..schemas.neural_networks import (
    DataAnalysisRequest,
    DataAnalysisResponse,
    DatasetSummaryResponse,
    ColumnInfoResponse,
    QuickPlotRequest,
    QuickPlotResponse
)

@router.post("/analysis/plots", response_model=DataAnalysisResponse)
async def generate_analysis_plots(
    request: DataAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Genera múltiples gráficos de análisis de datos según configuraciones especificadas
    """
    try:
        # Verificar que el dataset existe
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        # Generar gráficos
        result = await nn_service.generate_data_analysis_plots(
            request.dataset_id, 
            [config.model_dump() for config in request.plot_configs]
        )
        
        return DataAnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráficos: {str(e)}")

@router.get("/analysis/dataset/{dataset_id}/summary", response_model=DatasetSummaryResponse)
async def get_dataset_summary(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Obtiene resumen estadístico completo del dataset
    """
    try:
        # Verificar que el dataset existe
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        summary = await nn_service.get_dataset_summary(dataset_id)
        return DatasetSummaryResponse(**summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo resumen: {str(e)}")

@router.get("/analysis/dataset/{dataset_id}/columns", response_model=ColumnInfoResponse)
async def get_column_info(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Obtiene información detallada de las columnas del dataset
    """
    try:
        # Verificar que el dataset existe
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        column_info = await nn_service.get_column_info(dataset_id)
        return ColumnInfoResponse(**column_info)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo información de columnas: {str(e)}")

@router.post("/analysis/quick-plot", response_model=QuickPlotResponse)
async def generate_quick_plot(
    request: QuickPlotRequest,
    db: Session = Depends(get_db)
):
    """
    Genera un gráfico rápido con configuración simplificada
    """
    try:
        # Verificar que el dataset existe
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        # Crear configuración de gráfico
        plot_config = {
            'type': request.plot_type,
            'title': request.title,
            'x_axis': request.x_column,
            'y_axis': request.y_column,
            'color_by': request.color_column
        }
        
        # Generar gráfico
        result = await nn_service.generate_data_analysis_plots(
            request.dataset_id, 
            [plot_config]
        )
        
        if not result['plots']:
            raise HTTPException(status_code=400, detail="No se pudo generar el gráfico")
        
        plot_data = result['plots'][0]
        
        # Obtener información básica del dataset
        dataset_info = await nn_service.get_column_info(request.dataset_id)
        
        return QuickPlotResponse(
            plot_data=plot_data['plot_data'],
            plot_config=plot_data['plot_config'],
            dataset_info={
                'total_columns': len(dataset_info['columns']),
                'numeric_columns': len(dataset_info['numeric_columns']),
                'categorical_columns': len(dataset_info['categorical_columns'])
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráfico rápido: {str(e)}")

@router.get("/analysis/dataset/{dataset_id}/correlations")
async def get_correlations(
    dataset_id: int,
    method: str = "pearson",
    columns: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Obtiene matriz de correlaciones del dataset
    """
    try:
        # Verificar que el dataset existe
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        # Versión simplificada para debugging
        dataset_info = await nn_service.get_column_info(dataset_id)
        numeric_columns = dataset_info['numeric_columns']
        
        if len(numeric_columns) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 columnas numéricas para correlaciones")
        
        return {
            'message': f'Dataset {dataset_id} tiene {len(numeric_columns)} columnas numéricas',
            'numeric_columns': numeric_columns,
            'method': method,
            'dataset_id': dataset_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando correlaciones: {str(e)}")

@router.get("/analysis/plot-types")
async def get_available_plot_types():
    """
    Obtiene los tipos de gráficos disponibles con sus descripciones
    """
    return {
        'plot_types': [
            {
                'type': 'scatter',
                'name': 'Scatter Plot',
                'description': 'Gráfico de dispersión para mostrar relación entre dos variables numéricas',
                'required_params': ['x_axis', 'y_axis'],
                'optional_params': ['color_by', 'size_by', 'hover_data']
            },
            {
                'type': 'histogram',
                'name': 'Histogram',
                'description': 'Histograma para mostrar distribución de una variable',
                'required_params': ['column'],
                'optional_params': ['bins', 'color']
            },
            {
                'type': 'box',
                'name': 'Box Plot',
                'description': 'Diagrama de caja para mostrar distribución y outliers',
                'required_params': ['y_axis'],
                'optional_params': ['x_axis']
            },
            {
                'type': 'correlation',
                'name': 'Correlation Matrix',
                'description': 'Matriz de correlación entre variables numéricas',
                'required_params': [],
                'optional_params': ['columns', 'method']
            },
            {
                'type': 'line',
                'name': 'Line Plot',
                'description': 'Gráfico de líneas para series temporales o tendencias',
                'required_params': ['x_axis', 'y_axis'],
                'optional_params': ['group_by']
            },
            {
                'type': 'bar',
                'name': 'Bar Plot',
                'description': 'Gráfico de barras para variables categóricas',
                'required_params': ['x_axis'],
                'optional_params': ['y_axis']
            },
            {
                'type': 'heatmap',
                'name': 'Heatmap',
                'description': 'Mapa de calor para visualizar patrones en datos',
                'required_params': [],
                'optional_params': ['columns']
            },
            {
                'type': 'violin',
                'name': 'Violin Plot',
                'description': 'Gráfico de violín que combina box plot y densidad',
                'required_params': ['y_axis'],
                'optional_params': ['x_axis']
            },
            {
                'type': 'distribution',
                'name': 'Distribution Analysis',
                'description': 'Análisis completo de distribución con estadísticas',
                'required_params': ['column'],
                'optional_params': []
            },
            {
                'type': 'pair',
                'name': 'Pair Plot',
                'description': 'Matriz de gráficos de pares entre variables',
                'required_params': [],
                'optional_params': ['columns']
            }
        ]
    }