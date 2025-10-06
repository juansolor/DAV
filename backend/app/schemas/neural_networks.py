"""
Pydantic schemas for Neural Networks module
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union, Annotated
from datetime import datetime
from enum import Enum
import re

class ModelType(str, Enum):
    """Tipos de modelos disponibles"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIMESERIES = "timeseries"

class TimeSeriesModelType(str, Enum):
    """Tipos de modelos de series temporales en Darts"""
    NBEATS = "nbeats"
    NHITS = "nhits"
    TCN = "tcn"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"

class ActivationFunction(str, Enum):
    """Funciones de activación disponibles"""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LINEAR = "linear"

class Optimizer(str, Enum):
    """Optimizadores disponibles"""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"

class BaseNeuralNetworkRequest(BaseModel):
    """Base para todas las requests de redes neuronales"""
    dataset_id: int
    target_column: str
    feature_columns: Optional[List[str]] = None
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)

class ClassificationRequest(BaseNeuralNetworkRequest):
    """Request para entrenamiento de clasificación"""
    epochs: int = Field(default=100, ge=10, le=1000)
    batch_size: int = Field(default=32, ge=8, le=512)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    hidden_layers: List[int] = Field(default=[128, 64, 32])
    activation: ActivationFunction = Field(default=ActivationFunction.RELU)
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.8)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.3)
    optimizer: Optimizer = Field(default=Optimizer.ADAM)
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=10, ge=5, le=50)

class RegressionRequest(BaseNeuralNetworkRequest):
    """Request para entrenamiento de regresión"""
    epochs: int = Field(default=100, ge=10, le=1000)
    batch_size: int = Field(default=32, ge=8, le=512)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    hidden_layers: List[int] = Field(default=[128, 64, 32])
    activation: ActivationFunction = Field(default=ActivationFunction.RELU)
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.8)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.3)
    optimizer: Optimizer = Field(default=Optimizer.ADAM)
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=10, ge=5, le=50)

class TimeSeriesRequest(BaseNeuralNetworkRequest):
    """Request para entrenamiento de series temporales"""
    date_column: str
    model_type: TimeSeriesModelType = Field(default=TimeSeriesModelType.LSTM)
    input_chunk_length: int = Field(default=12, ge=1, le=100)
    output_chunk_length: int = Field(default=1, ge=1, le=50)
    n_epochs: int = Field(default=100, ge=10, le=1000)
    batch_size: int = Field(default=32, ge=8, le=512)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    hidden_size: int = Field(default=64, ge=16, le=512)
    num_layers: int = Field(default=2, ge=1, le=10)
    dropout: float = Field(default=0.1, ge=0.0, le=0.8)
    optimizer: Optimizer = Field(default=Optimizer.ADAM)
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=10, ge=5, le=50)

class ModelResponse(BaseModel):
    """Response con información del modelo"""
    model_id: str
    model_type: ModelType
    name: str
    created_at: datetime
    dataset_id: int
    target_column: str
    feature_columns: List[str]
    metrics: Dict[str, Any]
    status: str
    file_path: str

class TrainingResponse(BaseModel):
    """Response del entrenamiento"""
    model_id: str
    status: str
    message: str
    training_time: float
    metrics: Dict[str, Any]
    model_info: Dict[str, Any]
    plots: Optional[Dict[str, str]] = None

class PredictionRequest(BaseModel):
    """Request para predicción individual"""
    features: Dict[str, Union[float, int, str]]

class PredictionResponse(BaseModel):
    """Response de predicción"""
    prediction: Union[float, int, str, List[Union[float, int, str]]]
    probability: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    model_id: str
    timestamp: datetime

class BatchPredictionResponse(BaseModel):
    """Response para predicciones en lote"""
    predictions: List[Union[float, int, str]]
    probabilities: Optional[List[Dict[str, float]]] = None
    confidences: Optional[List[float]] = None
    model_id: str
    total_predictions: int
    timestamp: datetime

class ModelMetrics(BaseModel):
    """Métricas detalladas del modelo"""
    model_id: str
    model_type: ModelType
    
    # Métricas generales
    training_time: float
    total_parameters: int
    
    # Métricas de clasificación
    accuracy: Optional[float] = None
    precision: Optional[Dict[str, float]] = None
    recall: Optional[Dict[str, float]] = None
    f1_score: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    
    # Métricas de regresión
    mae: Optional[float] = None  # Mean Absolute Error
    mse: Optional[float] = None  # Mean Squared Error
    rmse: Optional[float] = None  # Root Mean Squared Error
    r2_score: Optional[float] = None  # R² Score
    
    # Métricas de series temporales
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    smape: Optional[float] = None  # Symmetric Mean Absolute Percentage Error
    msis: Optional[float] = None  # Mean Scaled Interval Score
    
    # Histórico de entrenamiento
    training_history: Optional[Dict[str, List[float]]] = None
    validation_history: Optional[Dict[str, List[float]]] = None

class ModelPlotResponse(BaseModel):
    """Response para gráficos del modelo"""
    plot_type: str
    plot_data: Dict[str, Any]
    plot_config: Dict[str, Any]
    model_id: str

class HyperparameterTuningRequest(BaseModel):
    """Request para ajuste de hiperparámetros"""
    dataset_id: int
    target_column: str
    feature_columns: Optional[List[str]] = None
    model_type: ModelType
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=5, ge=3, le=10)
    n_trials: int = Field(default=100, ge=10, le=1000)
    timeout: int = Field(default=3600, ge=300, le=10800)  # segundos
    
    # Rangos de hiperparámetros
    epochs_range: List[int] = Field(default=[50, 200])
    batch_size_options: List[int] = Field(default=[16, 32, 64, 128])
    learning_rate_range: List[float] = Field(default=[0.0001, 0.1])
    hidden_layers_options: List[List[int]] = Field(default=[[64], [128, 64], [256, 128, 64]])
    dropout_range: List[float] = Field(default=[0.0, 0.5])

class HyperparameterTuningResponse(BaseModel):
    """Response del ajuste de hiperparámetros"""
    study_id: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    study_time: float
    trials_data: List[Dict[str, Any]]
    best_model_id: str

class ModelComparisonRequest(BaseModel):
    """Request para comparar modelos"""
    model_ids: Annotated[List[str], Field(min_length=2, max_length=10)]
    metrics: List[str] = Field(default=["accuracy", "precision", "recall", "f1_score"])

class ModelComparisonResponse(BaseModel):
    """Response de comparación de modelos"""
    comparison_data: Dict[str, Dict[str, Any]]
    best_model: str
    ranking: List[str]
    summary: Dict[str, Any]

class ModelExportRequest(BaseModel):
    """Request para exportar modelo"""
    model_id: str
    export_format: str = Field(default="tensorflow")
    include_preprocessing: bool = Field(default=True)
    
    @field_validator('export_format')
    @classmethod
    def validate_export_format(cls, v):
        if v not in ["tensorflow", "onnx", "pickle"]:
            raise ValueError('export_format must be one of: tensorflow, onnx, pickle')
        return v

class ModelExportResponse(BaseModel):
    """Response de exportación de modelo"""
    download_url: str
    file_size: int
    export_format: str
    model_id: str
    exported_at: datetime