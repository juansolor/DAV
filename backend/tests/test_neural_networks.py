"""
Tests para el módulo de redes neuronales
"""
import pytest
import json
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from app.db.session import get_db, Base
from app.db.models import Dataset

# Configurar base de datos de prueba
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = None
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        if db:
            db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="function")
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def sample_dataset(setup_database):
    """Crear dataset de prueba"""
    # Crear datos de ejemplo
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'target': [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    # Guardar como CSV
    csv_path = "test_dataset.csv"
    df.to_csv(csv_path, index=False)
    
    # Crear registro en BD
    db = TestingSessionLocal()
    dataset = Dataset(
        name="Test Dataset",
        stored_filename=csv_path,
        original_filename="test_dataset.csv",
        n_rows=len(df),
        n_cols=len(df.columns),
        file_size=1000
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    db.close()
    
    return dataset

def test_neural_networks_endpoints_exist():
    """Verificar que los endpoints existen"""
    response = client.get("/neural-networks/models")
    assert response.status_code == 200
    assert response.json() == []

def test_train_classification_model(sample_dataset):
    """Test entrenamiento de modelo de clasificación"""
    payload = {
        "dataset_id": sample_dataset.id,
        "target_column": "target",
        "feature_columns": ["feature1", "feature2"],
        "epochs": 10,  # Pocas épocas para prueba rápida
        "batch_size": 2,
        "learning_rate": 0.01,
        "hidden_layers": [4, 2],
        "dropout_rate": 0.1,
        "validation_split": 0.2
    }
    
    response = client.post("/neural-networks/classification/train", json=payload)
    assert response.status_code == 200
    
    result = response.json()
    assert result["status"] == "success"
    assert "model_id" in result
    assert "training_time" in result
    assert "metrics" in result

def test_train_regression_model(sample_dataset):
    """Test entrenamiento de modelo de regresión"""
    # Modificar datos para regresión
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'target': [1.5, 2.8, 4.2, 5.7, 6.1, 7.9, 8.3, 9.8, 10.2, 11.5]
    }
    df = pd.DataFrame(data)
    csv_path = "test_regression_dataset.csv"
    df.to_csv(csv_path, index=False)
    
    # Actualizar dataset
    db = TestingSessionLocal()
    sample_dataset.file_path = csv_path
    db.add(sample_dataset)
    db.commit()
    db.close()
    
    payload = {
        "dataset_id": sample_dataset.id,
        "target_column": "target",
        "feature_columns": ["feature1", "feature2"],
        "epochs": 10,
        "batch_size": 2,
        "learning_rate": 0.01,
        "hidden_layers": [4, 2],
        "dropout_rate": 0.1,
        "validation_split": 0.2
    }
    
    response = client.post("/neural-networks/regression/train", json=payload)
    assert response.status_code == 200
    
    result = response.json()
    assert result["status"] == "success"
    assert "model_id" in result
    assert "metrics" in result

def test_list_models_after_training(sample_dataset):
    """Test listar modelos después de entrenar"""
    # Primero entrenar un modelo
    payload = {
        "dataset_id": sample_dataset.id,
        "target_column": "target",
        "epochs": 5,
        "batch_size": 2
    }
    
    train_response = client.post("/neural-networks/classification/train", json=payload)
    assert train_response.status_code == 200
    
    # Luego listar modelos
    response = client.get("/neural-networks/models")
    assert response.status_code == 200
    
    models = response.json()
    assert len(models) >= 1
    assert "model_id" in models[0]
    assert "model_type" in models[0]

def test_invalid_dataset_id():
    """Test con dataset_id inválido"""
    payload = {
        "dataset_id": 999,  # ID que no existe
        "target_column": "target",
        "epochs": 10
    }
    
    response = client.post("/neural-networks/classification/train", json=payload)
    assert response.status_code == 404

def test_model_service_initialization():
    """Test inicialización del servicio"""
    from app.services.neural_networks import NeuralNetworkService
    
    service = NeuralNetworkService()
    assert service.models_dir.exists()
    assert isinstance(service.models_metadata, dict)

def test_data_preprocessing():
    """Test preprocesamiento de datos"""
    from app.services.neural_networks import NeuralNetworkService
    
    service = NeuralNetworkService()
    
    # Datos con valores categóricos
    data = {
        'numeric_feature': [1, 2, 3, 4, 5],
        'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    X, y, encoders, feature_columns = service._preprocess_data(df, 'target')
    
    assert len(X) == 5
    assert len(y) == 5
    assert 'categorical_feature' in encoders
    assert len(feature_columns) == 2

def test_model_metadata_persistence():
    """Test persistencia de metadatos"""
    from app.services.neural_networks import NeuralNetworkService
    
    service = NeuralNetworkService()
    
    # Simular metadatos
    test_metadata = {
        "test_model": {
            "model_id": "test_model",
            "model_type": "classification",
            "created_at": "2023-01-01T00:00:00",
            "metrics": {"accuracy": 0.95}
        }
    }
    
    service.models_metadata = test_metadata
    service._save_models_metadata()
    
    # Crear nuevo servicio para verificar carga
    new_service = NeuralNetworkService()
    assert "test_model" in new_service.models_metadata
    assert new_service.models_metadata["test_model"]["metrics"]["accuracy"] == 0.95

@pytest.mark.asyncio
async def test_async_training_methods():
    """Test métodos asincrónicos de entrenamiento"""
    from app.services.neural_networks import NeuralNetworkService
    
    service = NeuralNetworkService()
    
    # Datos de prueba
    data = {
        'feature1': [1, 2, 3, 4, 5] * 4,  # 20 samples
        'feature2': [2, 4, 6, 8, 10] * 4,
        'target': [0, 1, 0, 1, 1] * 4
    }
    df = pd.DataFrame(data)
    
    # Test clasificación
    result = await service.train_classification_model(
        df=df,
        target_column='target',
        epochs=5,
        batch_size=4
    )
    
    assert result['status'] == 'success'
    assert 'model_id' in result
    assert 'metrics' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])