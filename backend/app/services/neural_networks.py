"""
Neural Networks Service
Implementa el entrenamiento y predicción con redes neuronales usando TensorFlow/Keras y Darts
"""
import os
import json
import pickle
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import HTTPException

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

# TensorFlow/Keras imports with availability check
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    
    # Clases mock para evitar errores de atributos
    class MockKeras:
        def __getattr__(self, name):
            return None
        
        class Sequential:
            def add(self, layer): pass
            def compile(self, **kwargs): pass
            def fit(self, *args, **kwargs):
                # Retorna objeto mock con atributo history
                class MockHistory:
                    def __init__(self):
                        self.history = {'loss': [0.1], 'accuracy': [0.9]}
                return MockHistory()
            def save(self, path): pass
            def predict(self, *args, **kwargs): return [[0.5], [0.7]]
            def count_params(self): return 1000
        
        class optimizers:
            @staticmethod
            def Adam(**kwargs): return None
        
        class models:
            @staticmethod
            def load_model(path): return None
    
    class MockLayers:
        @staticmethod
        def Dense(*args, **kwargs): return None
        
        @staticmethod
        def Dropout(*args, **kwargs): return None
    
    class MockCallbacks:
        @staticmethod
        def EarlyStopping(**kwargs): return None
    
    tf = None
    keras = MockKeras()
    layers = MockLayers()
    callbacks = MockCallbacks()


# Darts para series temporales
try:
    from darts import TimeSeries
    from darts.models import (
        NBEATSModel, NHiTSModel, TCNModel, TransformerModel,
        RNNModel, BlockRNNModel
    )
    from darts.metrics import mape, smape
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    
    # Clases mock para Darts
    class MockTimeSeries:
        @staticmethod
        def from_dataframe(*args, **kwargs):
            return None
    
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass
        
        def fit(self, *args, **kwargs):
            pass
        
        def predict(self, *args, **kwargs):
            return None
        
        def save(self, path):
            pass
    
    def mock_metric(*args, **kwargs):
        return 0.0
    
    TimeSeries = MockTimeSeries()
    NBEATSModel = MockModel
    NHiTSModel = MockModel
    TCNModel = MockModel
    TransformerModel = MockModel
    RNNModel = MockModel
    BlockRNNModel = MockModel
    mape = mock_metric
    smape = mock_metric

# Visualización y análisis
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend no interactivo para servidor
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    import base64
    import io
    try:
        from scipy import stats
        STATS_AVAILABLE = True
    except ImportError:
        stats = None
        STATS_AVAILABLE = False
    VISUALIZATION_AVAILABLE = True
    PLOTLY_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    PLOTLY_AVAILABLE = False
    STATS_AVAILABLE = False
    
    # Clases mock para visualización
    class MockFigure:
        def __init__(self, *args, **kwargs):
            pass
        
        def update_layout(self, *args, **kwargs):
            pass
        
        def to_json(self):
            return '{"data": [], "layout": {}}'
        
        def to_dict(self):
            return {"data": [], "layout": {}}
        
        def add_trace(self, *args, **kwargs):
            pass
        
        def update_traces(self, *args, **kwargs):
            pass
    
    class MockGO:
        @staticmethod
        def Figure(*args, **kwargs):
            return MockFigure()
        
        @staticmethod
        def Scatter(*args, **kwargs):
            return None
        
        @staticmethod
        def Heatmap(*args, **kwargs):
            return None
        
        @staticmethod
        def Bar(*args, **kwargs):
            return None
        
        @staticmethod
        def Histogram(*args, **kwargs):
            return None
        
        @staticmethod
        def Box(*args, **kwargs):
            return None
        
        @staticmethod
        def Table(*args, **kwargs):
            return None
    
    class MockPX:
        @staticmethod
        def scatter(*args, **kwargs):
            return MockFigure()
        
        @staticmethod
        def histogram(*args, **kwargs):
            return MockFigure()
        
        @staticmethod
        def box(*args, **kwargs):
            return MockFigure()
        
        @staticmethod
        def violin(*args, **kwargs):
            return MockFigure()
        
        @staticmethod
        def line(*args, **kwargs):
            return MockFigure()
        
        @staticmethod
        def imshow(*args, **kwargs):
            return MockFigure()
        
        @staticmethod
        def bar(*args, **kwargs):
            return MockFigure()
    
    def mock_make_subplots(*args, **kwargs):
        return MockFigure()
    
    plt = None
    sns = None
    go = MockGO()
    px = MockPX()
    ff = None
    make_subplots = mock_make_subplots
    base64 = None
    io = None
    stats = None

from ..schemas.neural_networks import ModelType, TimeSeriesModelType

class NeuralNetworkService:
    """Servicio para entrenar y manejar redes neuronales"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.models_metadata = {}
        self._load_models_metadata()
    
    def _load_models_metadata(self):
        """Carga metadatos de modelos existentes"""
        metadata_file = self.models_dir / "models_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.models_metadata = json.load(f)
    
    def _save_models_metadata(self):
        """Guarda metadatos de modelos"""
        metadata_file = self.models_dir / "models_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.models_metadata, f, indent=2, default=str)
    
    async def _load_dataset(self, dataset_id: int) -> Optional[pd.DataFrame]:
        """Carga un dataset por su ID"""
        try:
            # Importamos aquí para evitar dependencias circulares
            from ..db.session import SessionLocal
            from ..db.models import Dataset as DatasetModel
            
            # Crear sesión de base de datos
            db = SessionLocal()
            try:
                # Buscar el dataset en la base de datos
                dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
                if not dataset:
                    return None
                
                # Construir path completo al archivo
                # Usar el mismo path que en datasets.py
                data_dir = "backend/data/uploads"
                file_path = Path(data_dir) / str(dataset.stored_filename)
                
                if not file_path.exists():
                    return None
                
                # Cargar el archivo según su extensión
                if file_path.suffix.lower() == '.csv':
                    return pd.read_csv(file_path)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    return pd.read_excel(file_path)
                else:
                    # Intentar como CSV por defecto
                    return pd.read_csv(file_path)
                    
            finally:
                db.close()
                
        except Exception as e:
            print(f"Error loading dataset {dataset_id}: {str(e)}")
            return None
    
    def _preprocess_data(self, df: pd.DataFrame, target_column: str, 
                        feature_columns: Optional[List[str]] = None) -> Tuple:
        """Preprocesa los datos para entrenamiento"""
        # Seleccionar columnas
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Manejar valores faltantes
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        
        # Codificar variables categóricas en X
        categorical_columns = X.select_dtypes(include=['object']).columns
        encoders = {}
        
        for col in categorical_columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            encoders[col] = encoder
        
        return X, y, encoders, feature_columns
    
    async def train_classification_model(self, df: pd.DataFrame, target_column: str,
                                       feature_columns: Optional[List[str]] = None,
                                       test_size: float = 0.2, epochs: int = 100,
                                       batch_size: int = 32, learning_rate: float = 0.001,
                                       hidden_layers: List[int] = [128, 64, 32],
                                       dropout_rate: float = 0.2,
                                       validation_split: float = 0.2) -> Dict[str, Any]:
        """Entrena un modelo de clasificación"""
        start_time = datetime.now()
        model_id = str(uuid.uuid4())
        
        try:
            # Verificar que TensorFlow esté disponible
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow no está disponible. Instale tensorflow para usar esta funcionalidad.")
            
            # Preprocesar datos
            X, y, encoders, feature_columns = self._preprocess_data(df, target_column, feature_columns)
            
            # Codificar variable objetivo
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            n_classes = len(label_encoder.classes_)
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            # Escalar características
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Crear modelo
            model = keras.Sequential()
            model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(X_train_scaled.shape[1],)))
            model.add(layers.Dropout(dropout_rate))
            
            for units in hidden_layers[1:]:
                model.add(layers.Dense(units, activation='relu'))
                model.add(layers.Dropout(dropout_rate))
            
            # Capa de salida
            if n_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(layers.Dense(n_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
            
            # Compilar modelo
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss,
                metrics=metrics
            )
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            # Entrenar modelo
            history = model.fit(
                X_train_scaled, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluación
            y_pred = model.predict(X_test_scaled)
            # Convertir a numpy array si es necesario
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            
            if n_classes == 2:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred_classes)
            precision = precision_score(y_test, y_pred_classes, average='weighted')
            recall = recall_score(y_test, y_pred_classes, average='weighted')
            f1 = f1_score(y_test, y_pred_classes, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred_classes)
            
            # Guardar modelo y metadatos
            model_path = self.models_dir / f"{model_id}.h5"
            model.save(model_path)
            
            # Guardar preprocessors
            preprocessors = {
                'scaler': scaler,
                'label_encoder': label_encoder,
                'encoders': encoders,
                'feature_columns': feature_columns,
                'n_classes': n_classes
            }
            
            preprocessors_path = self.models_dir / f"{model_id}_preprocessors.pkl"
            with open(preprocessors_path, 'wb') as f:
                pickle.dump(preprocessors, f)
            
            # Metadatos del modelo
            training_time = (datetime.now() - start_time).total_seconds()
            
            model_metadata = {
                'model_id': model_id,
                'model_type': ModelType.CLASSIFICATION,
                'created_at': start_time.isoformat(),
                'target_column': target_column,
                'feature_columns': feature_columns,
                'n_classes': n_classes,
                'classes': label_encoder.classes_.tolist(),
                'model_path': str(model_path),
                'preprocessors_path': str(preprocessors_path),
                'training_time': training_time,
                'parameters': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'hidden_layers': hidden_layers,
                    'dropout_rate': dropout_rate
                },
                'metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'confusion_matrix': conf_matrix.tolist()
                },
                'training_history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'accuracy': [float(x) for x in history.history.get('accuracy', [])],
                    'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                    'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])]
                }
            }
            
            self.models_metadata[model_id] = model_metadata
            self._save_models_metadata()
            
            return {
                'model_id': model_id,
                'status': 'success',
                'message': 'Modelo de clasificación entrenado exitosamente',
                'training_time': training_time,
                'metrics': model_metadata['metrics'],
                'model_info': {
                    'model_type': 'classification',
                    'n_classes': n_classes,
                    'classes': label_encoder.classes_.tolist(),
                    'feature_columns': feature_columns,
                    'total_parameters': model.count_params()
                }
            }
            
        except Exception as e:
            return {
                'model_id': model_id,
                'status': 'error',
                'message': f'Error entrenando modelo: {str(e)}',
                'training_time': 0,
                'metrics': {},
                'model_info': {}
            }
    
    async def train_regression_model(self, df: pd.DataFrame, target_column: str,
                                   feature_columns: Optional[List[str]] = None,
                                   test_size: float = 0.2, epochs: int = 100,
                                   batch_size: int = 32, learning_rate: float = 0.001,
                                   hidden_layers: List[int] = [128, 64, 32],
                                   dropout_rate: float = 0.2,
                                   validation_split: float = 0.2) -> Dict[str, Any]:
        """Entrena un modelo de regresión"""
        start_time = datetime.now()
        model_id = str(uuid.uuid4())
        
        try:
            # Verificar que TensorFlow esté disponible
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow no está disponible. Instale tensorflow para usar esta funcionalidad.")
            
            # Preprocesar datos
            X, y, encoders, feature_columns = self._preprocess_data(df, target_column, feature_columns)
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Escalar datos
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            
            # Crear modelo
            model = keras.Sequential()
            model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(X_train_scaled.shape[1],)))
            model.add(layers.Dropout(dropout_rate))
            
            for units in hidden_layers[1:]:
                model.add(layers.Dense(units, activation='relu'))
                model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.Dense(1))  # Salida lineal para regresión
            
            # Compilar modelo
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            # Entrenar modelo
            history = model.fit(
                X_train_scaled, y_train_scaled,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Predicciones
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            
            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Guardar modelo y metadatos
            model_path = self.models_dir / f"{model_id}.h5"
            model.save(model_path)
            
            # Guardar preprocessors
            preprocessors = {
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'encoders': encoders,
                'feature_columns': feature_columns
            }
            
            preprocessors_path = self.models_dir / f"{model_id}_preprocessors.pkl"
            with open(preprocessors_path, 'wb') as f:
                pickle.dump(preprocessors, f)
            
            # Metadatos del modelo
            training_time = (datetime.now() - start_time).total_seconds()
            
            model_metadata = {
                'model_id': model_id,
                'model_type': ModelType.REGRESSION,
                'created_at': start_time.isoformat(),
                'target_column': target_column,
                'feature_columns': feature_columns,
                'model_path': str(model_path),
                'preprocessors_path': str(preprocessors_path),
                'training_time': training_time,
                'parameters': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'hidden_layers': hidden_layers,
                    'dropout_rate': dropout_rate
                },
                'metrics': {
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2_score': float(r2)
                },
                'training_history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'mae': [float(x) for x in history.history.get('mae', [])],
                    'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                    'val_mae': [float(x) for x in history.history.get('val_mae', [])]
                }
            }
            
            self.models_metadata[model_id] = model_metadata
            self._save_models_metadata()
            
            return {
                'model_id': model_id,
                'status': 'success',
                'message': 'Modelo de regresión entrenado exitosamente',
                'training_time': training_time,
                'metrics': model_metadata['metrics'],
                'model_info': {
                    'model_type': 'regression',
                    'feature_columns': feature_columns,
                    'total_parameters': model.count_params()
                }
            }
            
        except Exception as e:
            return {
                'model_id': model_id,
                'status': 'error',
                'message': f'Error entrenando modelo: {str(e)}',
                'training_time': 0,
                'metrics': {},
                'model_info': {}
            }
    
    async def train_timeseries_model(self, df: pd.DataFrame, target_column: str,
                                   date_column: str, feature_columns: Optional[List[str]] = None,
                                   model_type: TimeSeriesModelType = TimeSeriesModelType.LSTM,
                                   input_chunk_length: int = 12, output_chunk_length: int = 1,
                                   n_epochs: int = 100, batch_size: int = 32,
                                   learning_rate: float = 0.001, hidden_size: int = 64,
                                   num_layers: int = 2, dropout: float = 0.1) -> Dict[str, Any]:
        """Entrena un modelo de series temporales usando Darts"""
        if not DARTS_AVAILABLE:
            return {
                'model_id': '',
                'status': 'error',
                'message': 'Darts no está disponible. Instalar con: pip install darts',
                'training_time': 0,
                'metrics': {},
                'model_info': {}
            }
        
        start_time = datetime.now()
        model_id = str(uuid.uuid4())
        
        try:
            # Preparar datos para serie temporal
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            
            # Crear serie temporal principal
            ts = TimeSeries.from_dataframe(
                df, time_col=date_column, value_cols=target_column
            )
            
            # Series temporales adicionales (covariables)
            covariates = None
            if feature_columns:
                cov_cols = [col for col in feature_columns if col in df.columns and col != target_column]
                if cov_cols:
                    covariates = TimeSeries.from_dataframe(
                        df, time_col=date_column, value_cols=cov_cols
                    )
            
            # Dividir datos (80% entrenamiento, 20% prueba)
            if ts is None:
                raise ValueError("No se pudo crear la serie temporal")
            
            train_size = int(0.8 * len(ts)) if hasattr(ts, '__len__') else 80
            if hasattr(ts, '__getitem__'):
                train_ts = ts[:train_size]
                test_ts = ts[train_size:]
            else:
                # Fallback en caso de que ts sea None o no tenga métodos de slicing
                train_ts = None
                test_ts = None
            
            train_covariates = None
            test_covariates = None
            if covariates:
                train_covariates = covariates[:train_size]
                test_covariates = covariates[train_size:]
            
            # Seleccionar y configurar modelo
            model_params = {
                'input_chunk_length': input_chunk_length,
                'output_chunk_length': output_chunk_length,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'optimizer_kwargs': {'lr': learning_rate},
                'model_name': f'timeseries_{model_id}',
                'save_checkpoints': True,
                'force_reset': True
            }
            
            if model_type == TimeSeriesModelType.NBEATS:
                model = NBEATSModel(**model_params)
            elif model_type == TimeSeriesModelType.NHITS:
                model = NHiTSModel(**model_params)
            elif model_type == TimeSeriesModelType.TCN:
                model = TCNModel(**model_params)
            elif model_type == TimeSeriesModelType.TRANSFORMER:
                model = TransformerModel(**model_params)
            elif model_type == TimeSeriesModelType.LSTM:
                model = BlockRNNModel(
                    model='LSTM',
                    hidden_dim=hidden_size,
                    n_rnn_layers=num_layers,
                    dropout=dropout,
                    **model_params
                )
            elif model_type == TimeSeriesModelType.GRU:
                model = BlockRNNModel(
                    model='GRU',
                    hidden_dim=hidden_size,
                    n_rnn_layers=num_layers,
                    dropout=dropout,
                    **model_params
                )
            else:  # RNN
                model = BlockRNNModel(
                    model='RNN',
                    hidden_dim=hidden_size,
                    n_rnn_layers=num_layers,
                    dropout=dropout,
                    **model_params
                )
            
            # Entrenar modelo
            model.fit(train_ts, future_covariates=train_covariates, verbose=False)
            
            # Hacer predicciones
            if test_ts is not None and hasattr(test_ts, '__len__'):
                predictions = model.predict(
                    n=len(test_ts),
                    future_covariates=test_covariates
                )
                
                # Calcular métricas
                mape_score = mape(test_ts, predictions)
                smape_score = smape(test_ts, predictions)
            else:
                # Fallback si test_ts es None
                predictions = model.predict(n=10, future_covariates=test_covariates)
                mape_score = 0.0
                smape_score = 0.0
            
            # Guardar modelo
            model_path = self.models_dir / f"{model_id}_darts.pkl"
            model.save(str(model_path))
            
            # Guardar metadatos adicionales
            metadata = {
                'target_column': target_column,
                'date_column': date_column,
                'feature_columns': feature_columns,
                'train_size': train_size,
                'test_size': len(test_ts) if test_ts is not None and hasattr(test_ts, '__len__') else 0
            }
            
            metadata_path = self.models_dir / f"{model_id}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Metadatos del modelo
            training_time = (datetime.now() - start_time).total_seconds()
            
            model_metadata = {
                'model_id': model_id,
                'model_type': ModelType.TIMESERIES,
                'created_at': start_time.isoformat(),
                'target_column': target_column,
                'date_column': date_column,
                'feature_columns': feature_columns or [],
                'model_path': str(model_path),
                'metadata_path': str(metadata_path),
                'training_time': training_time,
                'darts_model_type': model_type,
                'parameters': {
                    'input_chunk_length': input_chunk_length,
                    'output_chunk_length': output_chunk_length,
                    'n_epochs': n_epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout
                },
                'metrics': {
                    'mape': float(mape_score) if isinstance(mape_score, (int, float)) else float(mape_score[0]) if isinstance(mape_score, list) and len(mape_score) > 0 else 0.0,
                    'smape': float(smape_score) if isinstance(smape_score, (int, float)) else float(smape_score[0]) if isinstance(smape_score, list) and len(smape_score) > 0 else 0.0
                }
            }
            
            self.models_metadata[model_id] = model_metadata
            self._save_models_metadata()
            
            return {
                'model_id': model_id,
                'status': 'success',
                'message': 'Modelo de series temporales entrenado exitosamente',
                'training_time': training_time,
                'metrics': model_metadata['metrics'],
                'model_info': {
                    'model_type': 'timeseries',
                    'darts_model_type': model_type,
                    'input_chunk_length': input_chunk_length,
                    'output_chunk_length': output_chunk_length,
                    'feature_columns': feature_columns or []
                }
            }
            
        except Exception as e:
            return {
                'model_id': model_id,
                'status': 'error',
                'message': f'Error entrenando modelo de series temporales: {str(e)}',
                'training_time': 0,
                'metrics': {},
                'model_info': {}
            }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Lista todos los modelos entrenados"""
        models = []
        for model_id, metadata in self.models_metadata.items():
            models.append({
                'model_id': model_id,
                'model_type': metadata['model_type'],
                'name': f"{metadata['model_type']}_{model_id[:8]}",
                'created_at': metadata['created_at'],
                'target_column': metadata['target_column'],
                'feature_columns': metadata['feature_columns'],
                'metrics': metadata.get('metrics', {}),
                'status': 'trained',
                'file_path': metadata.get('model_path', '')
            })
        return models
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información detallada de un modelo"""
        return self.models_metadata.get(model_id)
    
    async def predict(self, model_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza predicción con un modelo entrenado"""
        if model_id not in self.models_metadata:
            raise ValueError("Modelo no encontrado")
        
        metadata = self.models_metadata[model_id]
        model_type = metadata['model_type']
        
        try:
            if model_type in [ModelType.CLASSIFICATION, ModelType.REGRESSION]:
                return await self._predict_ml_model(model_id, data, metadata)
            elif model_type == ModelType.TIMESERIES:
                return await self._predict_timeseries_model(model_id, data, metadata)
            else:
                return {
                    'error': 'Tipo de modelo no soportado',
                    'model_id': model_id,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            return {
                'error': f"Error en predicción: {str(e)}",
                'model_id': model_id,
                'timestamp': datetime.now()
            }
    
    async def _predict_ml_model(self, model_id: str, data: Dict[str, Any], metadata: Dict) -> Dict[str, Any]:
        """Predicción para modelos de ML (clasificación/regresión)"""
        if not TF_AVAILABLE:
            raise ValueError("TensorFlow no está disponible")
        
        # Cargar modelo
        if hasattr(keras, 'models') and keras.models is not None:
            model = keras.models.load_model(metadata['model_path'])
        else:
            return {
                'error': 'TensorFlow/Keras no disponible para cargar modelo',
                'model_id': model_id,
                'timestamp': datetime.now()
            }
        
        # Cargar preprocessors
        with open(metadata['preprocessors_path'], 'rb') as f:
            preprocessors = pickle.load(f)
        
        # Preparar datos
        feature_columns = preprocessors['feature_columns']
        input_data = pd.DataFrame([data])[feature_columns]
        
        # Aplicar encoders
        for col, encoder in preprocessors['encoders'].items():
            if col in input_data.columns:
                input_data[col] = encoder.transform([str(data[col])])[0]
        
        # Escalar datos
        if metadata['model_type'] == ModelType.CLASSIFICATION:
            X_scaled = preprocessors['scaler'].transform(input_data)
            if model is not None and hasattr(model, 'predict'):
                prediction = model.predict(X_scaled)[0]
            else:
                return {
                    'error': 'Modelo no válido para predicción',
                    'model_id': model_id,
                    'timestamp': datetime.now()
                }
            
            if preprocessors['n_classes'] == 2:
                predicted_class = (prediction[0] > 0.5).astype(int)
                probability = float(prediction[0])
                class_name = preprocessors['label_encoder'].inverse_transform([predicted_class])[0]
                
                return {
                    'prediction': class_name,
                    'probability': {'positive': probability, 'negative': 1 - probability},
                    'confidence': max(probability, 1 - probability),
                    'model_id': model_id,
                    'timestamp': datetime.now()
                }
            else:
                predicted_class = np.argmax(prediction)
                probabilities = {
                    class_name: float(prob) for class_name, prob in 
                    zip(preprocessors['label_encoder'].classes_, prediction)
                }
                class_name = preprocessors['label_encoder'].classes_[predicted_class]
                
                return {
                    'prediction': class_name,
                    'probability': probabilities,
                    'confidence': float(prediction[predicted_class]),
                    'model_id': model_id,
                    'timestamp': datetime.now()
                }
        
        else:  # Regresión
            X_scaled = preprocessors['scaler_X'].transform(input_data)
            if model is not None and hasattr(model, 'predict'):
                prediction_scaled = model.predict(X_scaled)
            else:
                return {
                    'error': 'Modelo no válido para predicción de regresión',
                    'model_id': model_id,
                    'timestamp': datetime.now()
                }
            prediction = preprocessors['scaler_y'].inverse_transform(prediction_scaled)[0][0]
            
            return {
                'prediction': float(prediction),
                'model_id': model_id,
                'timestamp': datetime.now()
            }
    
    async def _predict_timeseries_model(self, model_id: str, data: Dict[str, Any], metadata: Dict) -> Dict[str, Any]:
        """Predicción para modelos de series temporales"""
        if not DARTS_AVAILABLE:
            raise ValueError("Darts no está disponible")
        
        # Cargar modelo
        try:
            from darts.models.forecasting.forecasting_model import ForecastingModel
            model = ForecastingModel.load(metadata['model_path'])
        except ImportError:
            raise ValueError("No se puede importar ForecastingModel de Darts")
        
        # Para series temporales, esperamos parámetros específicos
        n_periods = data.get('n_periods', 1)
        
        # Hacer predicción
        prediction = model.predict(n=n_periods)
        prediction_values = prediction.values().flatten().tolist()
        
        return {
            'prediction': prediction_values,
            'model_id': model_id,
            'timestamp': datetime.now()
        }
    
    async def batch_predict(self, model_id: str, df: pd.DataFrame) -> List[Any]:
        """Realiza predicciones en lote"""
        predictions = []
        for _, row in df.iterrows():
            try:
                result = await self.predict(model_id, row.to_dict())
                predictions.append(result['prediction'])
            except Exception as e:
                predictions.append(f"Error: {str(e)}")
        return predictions
    
    def delete_model(self, model_id: str) -> bool:
        """Elimina un modelo y sus archivos"""
        if model_id not in self.models_metadata:
            return False
        
        try:
            metadata = self.models_metadata[model_id]
            
            # Eliminar archivos
            if os.path.exists(metadata.get('model_path', '')):
                os.remove(metadata['model_path'])
            
            if os.path.exists(metadata.get('preprocessors_path', '')):
                os.remove(metadata['preprocessors_path'])
            
            if os.path.exists(metadata.get('metadata_path', '')):
                os.remove(metadata['metadata_path'])
            
            # Eliminar metadatos
            del self.models_metadata[model_id]
            self._save_models_metadata()
            
            return True
        except Exception:
            return False
    
    def get_model_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene métricas detalladas de un modelo"""
        if model_id not in self.models_metadata:
            return None
        
        metadata = self.models_metadata[model_id]
        return {
            'model_id': model_id,
            'model_type': metadata['model_type'],
            'training_time': metadata['training_time'],
            'metrics': metadata.get('metrics', {}),
            'training_history': metadata.get('training_history', {}),
            'parameters': metadata.get('parameters', {})
        }
    
    def generate_plot(self, model_id: str, plot_type: str) -> Optional[Dict[str, Any]]:
        """Genera gráficos de evaluación del modelo"""
        if model_id not in self.models_metadata:
            return None
        
        metadata = self.models_metadata[model_id]
        
        try:
            if plot_type == 'training_history':
                return self._plot_training_history(metadata)
            elif plot_type == 'confusion_matrix' and metadata['model_type'] == ModelType.CLASSIFICATION:
                return self._plot_confusion_matrix(metadata)
            elif plot_type == 'metrics_comparison':
                return self._plot_metrics_comparison(metadata)
            else:
                return None
        except Exception:
            return None
    
    def _plot_training_history(self, metadata: Dict) -> Optional[Dict[str, Any]]:
        """Genera gráfico del historial de entrenamiento"""
        history = metadata.get('training_history', {})
        if not history:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Loss', 'Accuracy/MAE'],
            vertical_spacing=0.1
        )
        
        # Plot loss
        if 'loss' in history:
            fig.add_trace(
                go.Scatter(y=history['loss'], name='Training Loss'),
                row=1, col=1
            )
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(y=history['val_loss'], name='Validation Loss'),
                row=1, col=1
            )
        
        # Plot accuracy or MAE
        if 'accuracy' in history:
            fig.add_trace(
                go.Scatter(y=history['accuracy'], name='Training Accuracy'),
                row=2, col=1
            )
        if 'val_accuracy' in history:
            fig.add_trace(
                go.Scatter(y=history['val_accuracy'], name='Validation Accuracy'),
                row=2, col=1
            )
        if 'mae' in history:
            fig.add_trace(
                go.Scatter(y=history['mae'], name='Training MAE'),
                row=2, col=1
            )
        if 'val_mae' in history:
            fig.add_trace(
                go.Scatter(y=history['val_mae'], name='Validation MAE'),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Training History',
            height=600
        )
        
        return {
            'plot_type': 'training_history',
            'plot_data': fig.to_dict(),
            'plot_config': {'displayModeBar': True}
        }
    
    def _plot_confusion_matrix(self, metadata: Dict) -> Optional[Dict[str, Any]]:
        """Genera matriz de confusión"""
        conf_matrix = metadata.get('metrics', {}).get('confusion_matrix')
        if not conf_matrix:
            return None
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        
        return {
            'plot_type': 'confusion_matrix',
            'plot_data': fig.to_dict(),
            'plot_config': {'displayModeBar': True}
        }
    
    def _plot_metrics_comparison(self, metadata: Dict) -> Optional[Dict[str, Any]]:
        """Genera gráfico de comparación de métricas"""
        metrics = metadata.get('metrics', {})
        if not metrics:
            return None
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Filtrar métricas numéricas
        numeric_metrics = []
        numeric_values = []
        for name, value in zip(metric_names, metric_values):
            if isinstance(value, (int, float)) and name != 'confusion_matrix':
                numeric_metrics.append(name)
                numeric_values.append(value)
        
        if not numeric_metrics:
            return None
        
        fig = go.Figure(data=[
            go.Bar(x=numeric_metrics, y=numeric_values, marker_color='skyblue')
        ])
        
        fig.update_layout(
            title='Model Performance Metrics',
            xaxis_title='Metrics',
            yaxis_title='Values'
        )
        
        return {
            'plot_type': 'metrics_comparison',
            'plot_data': fig.to_dict(),
            'plot_config': {'displayModeBar': True}
        }

    # Nuevas funciones de análisis visual
    async def generate_data_analysis_plots(self, dataset_id: int, plot_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera múltiples gráficos de análisis de datos según configuraciones especificadas
        """
        if not VISUALIZATION_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="Visualization libraries not available. Please install matplotlib, seaborn, and plotly."
            )
        
        try:
            # Cargar dataset
            dataset = await self._load_dataset(dataset_id)
            if dataset is None:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            plots = []
            
            for config in plot_configs:
                plot_type = config.get('type')
                plot_data = None
                
                if plot_type == 'scatter':
                    plot_data = await self._create_scatter_plot(dataset, config)
                elif plot_type == 'histogram':
                    plot_data = await self._create_histogram(dataset, config)
                elif plot_type == 'box':
                    plot_data = await self._create_box_plot(dataset, config)
                elif plot_type == 'correlation':
                    plot_data = await self._create_correlation_plot(dataset, config)
                elif plot_type == 'line':
                    plot_data = await self._create_line_plot(dataset, config)
                elif plot_type == 'bar':
                    plot_data = await self._create_bar_plot(dataset, config)
                elif plot_type == 'heatmap':
                    plot_data = await self._create_heatmap(dataset, config)
                elif plot_type == 'violin':
                    plot_data = await self._create_violin_plot(dataset, config)
                elif plot_type == 'distribution':
                    plot_data = await self._create_distribution_plot(dataset, config)
                elif plot_type == 'pair':
                    plot_data = await self._create_pair_plot(dataset, config)
                
                if plot_data:
                    plots.append(plot_data)
            
            return {
                'dataset_id': dataset_id,
                'plots': plots,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating plots: {str(e)}")

    async def _create_scatter_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea gráfico de dispersión"""
        x_col = config.get('x_axis')
        y_col = config.get('y_axis')
        color_col = config.get('color_by')
        size_col = config.get('size_by')
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Invalid x_axis or y_axis columns")
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            color=color_col if color_col and color_col in df.columns else None,
            size=size_col if size_col and size_col in df.columns else None,
            title=config.get('title', f'{x_col} vs {y_col}'),
            hover_data=config.get('hover_data', [])
        )
        
        return {
            'plot_type': 'scatter',
            'plot_data': fig.to_dict(),
            'config': config,
            'plot_config': {'displayModeBar': True}
        }

    async def _create_histogram(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea histograma"""
        column = config.get('column')
        bins = config.get('bins', 30)
        
        if not column or column not in df.columns:
            raise ValueError("Invalid column for histogram")
        
        fig = px.histogram(
            df, 
            x=column,
            nbins=bins,
            title=config.get('title', f'Distribution of {column}'),
            color_discrete_sequence=config.get('color', ['skyblue'])
        )
        
        return {
            'plot_type': 'histogram',
            'plot_data': fig.to_dict(),
            'config': config,
            'plot_config': {'displayModeBar': True}
        }

    async def _create_box_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea diagrama de caja"""
        y_col = config.get('y_axis')
        x_col = config.get('x_axis')  # Opcional para agrupación
        
        if not y_col or y_col not in df.columns:
            raise ValueError("Invalid y_axis column for box plot")
        
        fig = px.box(
            df, 
            y=y_col,
            x=x_col if x_col and x_col in df.columns else None,
            title=config.get('title', f'Box Plot of {y_col}')
        )
        
        return {
            'plot_type': 'box',
            'plot_data': fig.to_dict(),
            'config': config,
            'plot_config': {'displayModeBar': True}
        }

    async def _create_correlation_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea matriz de correlación"""
        columns = config.get('columns')
        method = config.get('method', 'pearson')
        
        # Si no se especifican columnas, usar todas las numéricas
        if not columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation")
        
        corr_matrix = df[numeric_cols].corr(method=method)
        
        fig = px.imshow(
            corr_matrix,
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title=config.get('title', f'{method.capitalize()} Correlation Matrix')
        )
        
        # Añadir valores de correlación como texto
        fig.update_traces(
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate='%{text}',
            textfont={"size": 10}
        )
        
        return {
            'plot_type': 'correlation',
            'plot_data': fig.to_dict(),
            'config': config,
            'correlation_data': corr_matrix.to_dict(),
            'plot_config': {'displayModeBar': True}
        }

    async def _create_line_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea gráfico de líneas"""
        x_col = config.get('x_axis')
        y_col = config.get('y_axis')
        group_col = config.get('group_by')
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Invalid x_axis or y_axis columns")
        
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col,
            color=group_col if group_col and group_col in df.columns else None,
            title=config.get('title', f'{y_col} over {x_col}')
        )
        
        return {
            'plot_type': 'line',
            'plot_data': fig.to_dict(),
            'config': config,
            'plot_config': {'displayModeBar': True}
        }

    async def _create_bar_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea gráfico de barras"""
        x_col = config.get('x_axis')
        y_col = config.get('y_axis')
        
        if not x_col or x_col not in df.columns:
            raise ValueError("Invalid x_axis column")
        
        # Si no hay y_col, hacer conteo
        if not y_col or y_col not in df.columns:
            value_counts = df[x_col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=config.get('title', f'Count of {x_col}')
            )
        else:
            fig = px.bar(
                df, 
                x=x_col, 
                y=y_col,
                title=config.get('title', f'{y_col} by {x_col}')
            )
        
        return {
            'plot_type': 'bar',
            'plot_data': fig.to_dict(),
            'config': config,
            'plot_config': {'displayModeBar': True}
        }

    async def _create_heatmap(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea mapa de calor"""
        columns = config.get('columns')
        
        if not columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for heatmap")
        
        fig = px.imshow(
            df[numeric_cols].values,
            aspect='auto',
            color_continuous_scale='Viridis',
            title=config.get('title', 'Data Heatmap')
        )
        
        return {
            'plot_type': 'heatmap',
            'plot_data': fig.to_dict(),
            'config': config,
            'plot_config': {'displayModeBar': True}
        }

    async def _create_violin_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea gráfico de violín"""
        y_col = config.get('y_axis')
        x_col = config.get('x_axis')
        
        if not y_col or y_col not in df.columns:
            raise ValueError("Invalid y_axis column for violin plot")
        
        fig = px.violin(
            df, 
            y=y_col,
            x=x_col if x_col and x_col in df.columns else None,
            box=True,
            title=config.get('title', f'Violin Plot of {y_col}')
        )
        
        return {
            'plot_type': 'violin',
            'plot_data': fig.to_dict(),
            'config': config,
            'plot_config': {'displayModeBar': True}
        }

    async def _create_distribution_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea gráfico de distribución con estadísticas"""
        column = config.get('column')
        
        if not column or column not in df.columns:
            raise ValueError("Invalid column for distribution plot")
        
        data = df[column].dropna()
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Histogram', 'Q-Q Plot', 'Box Plot', 'Statistics'),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "table"}]]
        )
        
        # Histograma
        fig.add_trace(
            go.Histogram(x=data, name='Distribution'),
            row=1, col=1
        )
        
        # Q-Q Plot (si scipy está disponible)
        if STATS_AVAILABLE and stats:
            sorted_data = np.sort(data)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Q-Q Plot'),
                row=1, col=2
            )
        else:
            # Gráfico alternativo si scipy no está disponible
            fig.add_trace(
                go.Histogram(x=data, name='Distribution (alt)'),
                row=1, col=2
            )
        
        # Box Plot
        fig.add_trace(
            go.Box(y=data, name='Box Plot'),
            row=2, col=1
        )
        
        # Estadísticas
        if STATS_AVAILABLE and stats:
            stats_data = {
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    f"{data.mean():.4f}",
                    f"{data.median():.4f}",
                    f"{data.std():.4f}",
                    f"{data.min():.4f}",
                    f"{data.max():.4f}",
                    f"{stats.skew(data):.4f}",
                    f"{stats.kurtosis(data):.4f}"
                ]
            }
        else:
            stats_data = {
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{data.mean():.4f}",
                    f"{data.median():.4f}",
                    f"{data.std():.4f}",
                    f"{data.min():.4f}",
                    f"{data.max():.4f}"
                ]
            }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(stats_data.keys())),
                cells=dict(values=list(stats_data.values()))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=config.get('title', f'Distribution Analysis of {column}'),
            showlegend=False
        )
        
        return {
            'plot_type': 'distribution',
            'plot_data': fig.to_dict(),
            'config': config,
            'statistics': stats_data,
            'plot_config': {'displayModeBar': True}
        }

    async def _create_pair_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crea matriz de gráficos de pares"""
        columns = config.get('columns')
        
        if not columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Limitar a 5 para rendimiento
        else:
            numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']][:5]
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for pair plot")
        
        # Crear matriz de subplots
        n_cols = len(numeric_cols)
        fig = make_subplots(
            rows=n_cols, cols=n_cols,
            subplot_titles=[f"{col1} vs {col2}" for col1 in numeric_cols for col2 in numeric_cols]
        )
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i == j:
                    # Diagonal: histograma
                    fig.add_trace(
                        go.Histogram(x=df[col1], name=f'{col1} dist', showlegend=False),
                        row=i+1, col=j+1
                    )
                else:
                    # Fuera de diagonal: scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=df[col2], 
                            y=df[col1], 
                            mode='markers',
                            name=f'{col1} vs {col2}',
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )
        
        fig.update_layout(
            title=config.get('title', 'Pair Plot Matrix'),
            showlegend=False
        )
        
        return {
            'plot_type': 'pair',
            'plot_data': fig.to_dict(),
            'config': config,
            'plot_config': {'displayModeBar': True}
        }

    async def get_dataset_summary(self, dataset_id: int) -> Dict[str, Any]:
        """Obtiene resumen estadístico del dataset"""
        try:
            dataset = await self._load_dataset(dataset_id)
            if dataset is None:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            summary = {
                'basic_info': {
                    'shape': list(dataset.shape),  # Convertir a lista para serialización JSON
                    'columns': list(dataset.columns),
                    'dtypes': dataset.dtypes.astype(str).to_dict(),
                    'memory_usage': int(dataset.memory_usage(deep=True).sum()),  # Convertir a int
                    'null_counts': dataset.isnull().sum().to_dict()
                },
                'numeric_summary': {},
                'categorical_summary': {},
                'correlations': {}
            }
            
            # Resumen numérico
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary['numeric_summary'] = dataset[numeric_cols].describe().to_dict()
                
                # Correlaciones si hay más de una columna numérica
                if len(numeric_cols) > 1:
                    summary['correlations'] = dataset[numeric_cols].corr().to_dict()
            
            # Resumen categórico
            categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    summary['categorical_summary'][col] = {
                        'unique_count': dataset[col].nunique(),
                        'unique_values': dataset[col].unique().tolist()[:20],  # Limitar a 20
                        'value_counts': dataset[col].value_counts().head(10).to_dict()
                    }
            
            return summary
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting dataset summary: {str(e)}")

    async def get_column_info(self, dataset_id: int) -> Dict[str, Any]:
        """Obtiene información detallada de las columnas del dataset"""
        try:
            dataset = await self._load_dataset(dataset_id)
            if dataset is None:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            column_info = {}
            
            for col in dataset.columns:
                col_data = dataset[col]
                info = {
                    'name': col,
                    'dtype': str(col_data.dtype),
                    'null_count': col_data.isnull().sum(),
                    'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                    'unique_count': col_data.nunique(),
                    'is_numeric': col_data.dtype in ['int64', 'float64'],
                    'is_categorical': col_data.dtype in ['object', 'category']
                }
                
                if info['is_numeric']:
                    info.update({
                        'min': float(col_data.min()) if not col_data.isnull().all() else None,
                        'max': float(col_data.max()) if not col_data.isnull().all() else None,
                        'mean': float(col_data.mean()) if not col_data.isnull().all() else None,
                        'median': float(col_data.median()) if not col_data.isnull().all() else None,
                        'std': float(col_data.std()) if not col_data.isnull().all() else None
                    })
                
                if info['is_categorical']:
                    value_counts = col_data.value_counts().head(10)
                    info.update({
                        'top_values': value_counts.to_dict(),
                        'sample_values': col_data.dropna().unique().tolist()[:10]
                    })
                
                column_info[col] = info
            
            return {
                'dataset_id': dataset_id,
                'columns': column_info,
                'numeric_columns': [col for col, info in column_info.items() if info['is_numeric']],
                'categorical_columns': [col for col, info in column_info.items() if info['is_categorical']]
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting column info: {str(e)}")