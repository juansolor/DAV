from sqlalchemy import Column, Integer, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
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
    
    # Relación con modelos de ML
    neural_network_models = relationship("NeuralNetworkModel", back_populates="dataset", cascade="all, delete-orphan")

class NeuralNetworkModel(Base):
    __tablename__ = 'neural_network_models'
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # classification, regression, timeseries
    
    # Referencias
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    dataset = relationship("Dataset", back_populates="neural_network_models")
    
    # Configuración del modelo
    target_column = Column(String, nullable=False)
    feature_columns = Column(Text)  # JSON string
    
    # Parámetros de entrenamiento
    epochs = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    hidden_layers = Column(Text)  # JSON string
    dropout_rate = Column(Float)
    
    # Parámetros específicos para series temporales
    date_column = Column(String)
    darts_model_type = Column(String)
    input_chunk_length = Column(Integer)
    output_chunk_length = Column(Integer)
    hidden_size = Column(Integer)
    num_layers = Column(Integer)
    
    # Métricas de evaluación
    training_time = Column(Float)
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    mae = Column(Float)
    mse = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    mape = Column(Float)
    smape = Column(Float)
    
    # Archivos y metadatos
    model_file_path = Column(String, nullable=False)
    preprocessors_file_path = Column(String)
    metadata_file_path = Column(String)
    training_history = Column(Text)  # JSON string
    
    # Estado y timestamps
    status = Column(String, default='trained')  # training, trained, error, deleted
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
