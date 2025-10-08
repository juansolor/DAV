import React, { useState, useEffect } from 'react';
import DataAnalysis from './DataAnalysis';

const NeuralNetworks = ({ datasets = [] }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [modelType, setModelType] = useState('classification');
  const [activeTab, setActiveTab] = useState('train');
  const [trainingParams, setTrainingParams] = useState({
    target_column: '',
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    hidden_layers: '128,64,32',
    dropout_rate: 0.2
  });
  const [selectedModel, setSelectedModel] = useState(null);
  const [predictionData, setPredictionData] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/neural-networks/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data);
      }
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const trainModel = async () => {
    if (!selectedDataset || !trainingParams.target_column) {
      alert('Selecciona un dataset y columna objetivo');
      return;
    }

    setLoading(true);
    try {
      let endpoint = '';
      let payload = {
        dataset_id: parseInt(selectedDataset),
        target_column: trainingParams.target_column,
        epochs: trainingParams.epochs,
        batch_size: trainingParams.batch_size,
        learning_rate: trainingParams.learning_rate,
        hidden_layers: trainingParams.hidden_layers.split(',').map(x => parseInt(x.trim())),
        dropout_rate: trainingParams.dropout_rate,
        validation_split: 0.2
      };

      if (modelType === 'classification') {
        endpoint = 'http://127.0.0.1:8000/neural-networks/classification/train';
      } else if (modelType === 'regression') {
        endpoint = 'http://127.0.0.1:8000/neural-networks/regression/train';
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        const result = await response.json();
        if (result.status === 'success') {
          alert('Modelo entrenado exitosamente');
          loadModels();
        } else {
          alert(`Error: ${result.message}`);
        }
      } else {
        alert('Error entrenando modelo');
      }
    } catch (error) {
      console.error('Error training model:', error);
      alert('Error entrenando modelo');
    } finally {
      setLoading(false);
    }
  };

  const makePrediction = async () => {
    if (!selectedModel || !predictionData) {
      alert('Selecciona un modelo e ingresa datos');
      return;
    }

    try {
      let data;
      try {
        data = JSON.parse(predictionData);
      } catch {
        alert('Formato JSON inválido');
        return;
      }

      const response = await fetch(`http://127.0.0.1:8000/neural-networks/predict/${selectedModel}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      if (response.ok) {
        const result = await response.json();
        setPredictionResult(result);
      } else {
        alert('Error realizando predicción');
      }
    } catch (error) {
      console.error('Error making prediction:', error);
      alert('Error realizando predicción');
    }
  };

  const deleteModel = async (modelId) => {
    if (!confirm('¿Estás seguro de eliminar este modelo?')) return;

    try {
      const response = await fetch(`http://127.0.0.1:8000/neural-networks/models/${modelId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        alert('Modelo eliminado');
        loadModels();
      } else {
        alert('Error eliminando modelo');
      }
    } catch (error) {
      console.error('Error deleting model:', error);
      alert('Error eliminando modelo');
    }
  };

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
        <span style={{ fontSize: '2rem' }}>🧠</span>
        <h1 style={{ fontSize: '2rem', fontWeight: 'bold', margin: 0 }}>Redes Neuronales</h1>
      </div>

      {/* Tabs */}
      <div style={{ 
        display: 'flex', 
        marginBottom: '2rem',
        borderBottom: '2px solid #e2e8f0'
      }}>
        {['train', 'models', 'predict', 'analysis'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '0.75rem 1.5rem',
              border: 'none',
              background: activeTab === tab ? '#3498db' : 'transparent',
              color: activeTab === tab ? 'white' : '#64748b',
              cursor: 'pointer',
              borderRadius: '4px 4px 0 0',
              marginRight: '0.5rem',
              fontWeight: activeTab === tab ? 'bold' : 'normal'
            }}
          >
            {tab === 'train' && '🏋️ Entrenar Modelo'}
            {tab === 'models' && '📊 Modelos Entrenados'}
            {tab === 'predict' && '🎯 Predicciones'}
            {tab === 'analysis' && '📈 Análisis Visual'}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'train' && (
        <div style={{ 
          backgroundColor: 'white', 
          padding: '2rem', 
          borderRadius: '8px', 
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)' 
        }}>
          <h2 style={{ marginBottom: '1.5rem', color: '#2c3e50' }}>🏋️ Entrenar Nuevo Modelo</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Dataset</label>
              <select 
                value={selectedDataset} 
                onChange={(e) => setSelectedDataset(e.target.value)}
                style={{ 
                  width: '100%', 
                  padding: '0.75rem', 
                  borderRadius: '4px', 
                  border: '1px solid #ddd' 
                }}
              >
                <option value="">Seleccionar dataset</option>
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Tipo de Modelo</label>
              <select 
                value={modelType} 
                onChange={(e) => setModelType(e.target.value)}
                style={{ 
                  width: '100%', 
                  padding: '0.75rem', 
                  borderRadius: '4px', 
                  border: '1px solid #ddd' 
                }}
              >
                <option value="classification">Clasificación</option>
                <option value="regression">Regresión</option>
              </select>
            </div>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Columna Objetivo</label>
            <input
              type="text"
              value={trainingParams.target_column}
              onChange={(e) => setTrainingParams({
                ...trainingParams,
                target_column: e.target.value
              })}
              placeholder="Nombre de la columna objetivo"
              style={{ 
                width: '100%', 
                padding: '0.75rem', 
                borderRadius: '4px', 
                border: '1px solid #ddd' 
              }}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Épocas</label>
              <input
                type="number"
                value={trainingParams.epochs}
                onChange={(e) => setTrainingParams({
                  ...trainingParams,
                  epochs: parseInt(e.target.value)
                })}
                min="10"
                max="1000"
                style={{ 
                  width: '100%', 
                  padding: '0.75rem', 
                  borderRadius: '4px', 
                  border: '1px solid #ddd' 
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Batch Size</label>
              <input
                type="number"
                value={trainingParams.batch_size}
                onChange={(e) => setTrainingParams({
                  ...trainingParams,
                  batch_size: parseInt(e.target.value)
                })}
                min="8"
                max="512"
                style={{ 
                  width: '100%', 
                  padding: '0.75rem', 
                  borderRadius: '4px', 
                  border: '1px solid #ddd' 
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                value={trainingParams.learning_rate}
                onChange={(e) => setTrainingParams({
                  ...trainingParams,
                  learning_rate: parseFloat(e.target.value)
                })}
                min="0.0001"
                max="0.1"
                style={{ 
                  width: '100%', 
                  padding: '0.75rem', 
                  borderRadius: '4px', 
                  border: '1px solid #ddd' 
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Dropout</label>
              <input
                type="number"
                step="0.1"
                value={trainingParams.dropout_rate}
                onChange={(e) => setTrainingParams({
                  ...trainingParams,
                  dropout_rate: parseFloat(e.target.value)
                })}
                min="0"
                max="0.8"
                style={{ 
                  width: '100%', 
                  padding: '0.75rem', 
                  borderRadius: '4px', 
                  border: '1px solid #ddd' 
                }}
              />
            </div>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Capas Ocultas (separadas por coma)</label>
            <input
              type="text"
              value={trainingParams.hidden_layers}
              onChange={(e) => setTrainingParams({
                ...trainingParams,
                hidden_layers: e.target.value
              })}
              placeholder="128, 64, 32"
              style={{ 
                width: '100%', 
                padding: '0.75rem', 
                borderRadius: '4px', 
                border: '1px solid #ddd' 
              }}
            />
          </div>

          <button 
            onClick={trainModel} 
            disabled={loading || !selectedDataset || !trainingParams.target_column}
            style={{
              width: '100%',
              padding: '1rem',
              backgroundColor: loading ? '#95a5a6' : '#3498db',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '1rem',
              fontWeight: 'bold'
            }}
          >
            {loading ? '🔄 Entrenando...' : '🚀 Entrenar Modelo'}
          </button>
        </div>
      )}

      {activeTab === 'models' && (
        <div style={{ 
          backgroundColor: 'white', 
          padding: '2rem', 
          borderRadius: '8px', 
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)' 
        }}>
          <h2 style={{ marginBottom: '1.5rem', color: '#2c3e50' }}>📊 Modelos Entrenados</h2>
          
          {models.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#7f8c8d' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🧠</div>
              <h3>No hay modelos entrenados</h3>
              <p>Entrena tu primer modelo en la pestaña anterior</p>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.5rem' }}>
              {models.map((model) => (
                <div key={model.model_id} style={{
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  padding: '1.5rem',
                  backgroundColor: '#f9f9f9',
                  transition: 'box-shadow 0.2s',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)'}
                onMouseLeave={(e) => e.currentTarget.style.boxShadow = 'none'}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h3 style={{ margin: 0, color: '#2c3e50' }}>{model.name}</h3>
                    <span style={{
                      padding: '0.25rem 0.75rem',
                      borderRadius: '12px',
                      fontSize: '0.75rem',
                      fontWeight: 'bold',
                      backgroundColor: model.model_type === 'classification' ? '#e3f2fd' : '#e8f5e8',
                      color: model.model_type === 'classification' ? '#1976d2' : '#388e3c'
                    }}>
                      {model.model_type}
                    </span>
                  </div>
                  
                  <div style={{ fontSize: '0.875rem', color: '#666', marginBottom: '1rem' }}>
                    <div>🎯 Objetivo: {model.target_column}</div>
                    <div>📅 Creado: {new Date(model.created_at).toLocaleDateString()}</div>
                  </div>

                  {model.metrics && Object.keys(model.metrics).length > 0 && (
                    <div style={{ marginBottom: '1rem' }}>
                      <h4 style={{ fontSize: '0.875rem', margin: '0 0 0.5rem 0', fontWeight: 'bold' }}>Métricas:</h4>
                      {Object.entries(model.metrics).slice(0, 3).map(([key, value]) => (
                        <div key={key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem' }}>
                          <span>{key.replace(/_/g, ' ').toUpperCase()}:</span>
                          <span style={{ fontFamily: 'monospace' }}>
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      onClick={() => setSelectedModel(model.model_id)}
                      style={{
                        flex: 1,
                        padding: '0.5rem',
                        backgroundColor: '#3498db',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.75rem'
                      }}
                    >
                      📊 Seleccionar
                    </button>
                    <button
                      onClick={() => deleteModel(model.model_id)}
                      style={{
                        padding: '0.5rem',
                        backgroundColor: '#e74c3c',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.75rem'
                      }}
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === 'predict' && (
        <div style={{ 
          backgroundColor: 'white', 
          padding: '2rem', 
          borderRadius: '8px', 
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)' 
        }}>
          <h2 style={{ marginBottom: '1.5rem', color: '#2c3e50' }}>🎯 Realizar Predicción</h2>
          
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Modelo</label>
            <select 
              value={selectedModel || ''} 
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{ 
                width: '100%', 
                padding: '0.75rem', 
                borderRadius: '4px', 
                border: '1px solid #ddd' 
              }}
            >
              <option value="">Seleccionar modelo</option>
              {models.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.name} ({model.model_type})
                </option>
              ))}
            </select>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Datos (JSON)</label>
            <textarea
              value={predictionData}
              onChange={(e) => setPredictionData(e.target.value)}
              placeholder='{"feature1": 1.0, "feature2": "value"}'
              rows={4}
              style={{ 
                width: '100%', 
                padding: '0.75rem', 
                borderRadius: '4px', 
                border: '1px solid #ddd',
                fontFamily: 'monospace'
              }}
            />
          </div>

          <button 
            onClick={makePrediction}
            disabled={!selectedModel || !predictionData}
            style={{
              width: '100%',
              padding: '1rem',
              backgroundColor: (!selectedModel || !predictionData) ? '#95a5a6' : '#27ae60',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: (!selectedModel || !predictionData) ? 'not-allowed' : 'pointer',
              fontSize: '1rem',
              fontWeight: 'bold',
              marginBottom: '1.5rem'
            }}
          >
            🎯 Predecir
          </button>

          {predictionResult && (
            <div style={{
              backgroundColor: '#f8f9fa',
              border: '1px solid #e9ecef',
              borderRadius: '8px',
              padding: '1.5rem'
            }}>
              <h3 style={{ margin: '0 0 1rem 0', color: '#2c3e50' }}>Resultado de Predicción</h3>
              
              <div style={{ marginBottom: '1rem' }}>
                <strong>Predicción: </strong>
                <span style={{ color: '#3498db', fontSize: '1.2rem', fontWeight: 'bold' }}>
                  {predictionResult.prediction}
                </span>
              </div>
              
              {predictionResult.probability && (
                <div style={{ marginBottom: '1rem' }}>
                  <strong>Probabilidades:</strong>
                  <div style={{ marginTop: '0.5rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.5rem' }}>
                    {Object.entries(predictionResult.probability).map(([key, value]) => (
                      <div key={key} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        backgroundColor: '#ffffff',
                        padding: '0.5rem',
                        borderRadius: '4px',
                        border: '1px solid #dee2e6'
                      }}>
                        <span>{key}:</span>
                        <span style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>
                          {(value * 100).toFixed(2)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {predictionResult.confidence && (
                <div style={{ marginBottom: '1rem' }}>
                  <strong>Confianza: </strong>
                  <span style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>
                    {(predictionResult.confidence * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              
              <div style={{ fontSize: '0.75rem', color: '#6c757d' }}>
                Timestamp: {new Date(predictionResult.timestamp).toLocaleString()}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Analysis Tab */}
      {activeTab === 'analysis' && (
        <DataAnalysis datasets={datasets} />
      )}
    </div>
  );
};

export default NeuralNetworks;