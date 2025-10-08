import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

const DataAnalysis = ({ datasets = [] }) => {
  const [selectedDataset, setSelectedDataset] = useState('');
  const [columnInfo, setColumnInfo] = useState(null);
  const [datasetSummary, setDatasetSummary] = useState(null);
  const [plotConfigs, setPlotConfigs] = useState([]);
  const [generatedPlots, setGeneratedPlots] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [availablePlotTypes, setAvailablePlotTypes] = useState([]);
  
  // Estados para gráfico rápido
  const [quickPlot, setQuickPlot] = useState({
    type: 'scatter',
    x_column: '',
    y_column: '',
    color_column: '',
    title: ''
  });

  useEffect(() => {
    loadAvailablePlotTypes();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      loadDatasetInfo();
      loadDatasetSummary();
    }
  }, [selectedDataset]);

  const loadAvailablePlotTypes = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/neural-networks/analysis/plot-types');
      if (response.ok) {
        const data = await response.json();
        setAvailablePlotTypes(data.plot_types);
      }
    } catch (error) {
      console.error('Error loading plot types:', error);
    }
  };

  const loadDatasetInfo = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/neural-networks/analysis/dataset/${selectedDataset}/columns`);
      if (response.ok) {
        const data = await response.json();
        setColumnInfo(data);
      }
    } catch (error) {
      console.error('Error loading dataset info:', error);
    }
  };

  const loadDatasetSummary = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/neural-networks/analysis/dataset/${selectedDataset}/summary`);
      if (response.ok) {
        const data = await response.json();
        setDatasetSummary(data);
      }
    } catch (error) {
      console.error('Error loading dataset summary:', error);
    }
  };

  const addPlotConfig = () => {
    const newConfig = {
      id: Date.now(),
      type: 'scatter',
      title: '',
      x_axis: '',
      y_axis: '',
      color_by: '',
      size_by: '',
      group_by: '',
      column: '',
      columns: [],
      bins: 30,
      method: 'pearson',
      color: [],
      hover_data: []
    };
    setPlotConfigs([...plotConfigs, newConfig]);
  };

  const updatePlotConfig = (id, field, value) => {
    setPlotConfigs(plotConfigs.map(config => 
      config.id === id ? { ...config, [field]: value } : config
    ));
  };

  const removePlotConfig = (id) => {
    setPlotConfigs(plotConfigs.filter(config => config.id !== id));
  };

  const generatePlots = async () => {
    if (!selectedDataset || plotConfigs.length === 0) {
      alert('Selecciona un dataset y agrega al menos una configuración de gráfico');
      return;
    }

    setLoading(true);
    try {
      const cleanConfigs = plotConfigs.map(config => {
        const cleaned = { ...config };
        delete cleaned.id;
        // Limpiar campos vacíos
        Object.keys(cleaned).forEach(key => {
          if (cleaned[key] === '' || (Array.isArray(cleaned[key]) && cleaned[key].length === 0)) {
            delete cleaned[key];
          }
        });
        return cleaned;
      });

      const response = await fetch('http://127.0.0.1:8000/neural-networks/analysis/plots', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: parseInt(selectedDataset),
          plot_configs: cleanConfigs
        })
      });

      if (response.ok) {
        const data = await response.json();
        setGeneratedPlots(data.plots);
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error generating plots:', error);
      alert('Error generando gráficos');
    } finally {
      setLoading(false);
    }
  };

  const generateQuickPlot = async () => {
    if (!selectedDataset || !quickPlot.type) {
      alert('Selecciona un dataset y tipo de gráfico');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/neural-networks/analysis/quick-plot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: parseInt(selectedDataset),
          plot_type: quickPlot.type,
          x_column: quickPlot.x_column || null,
          y_column: quickPlot.y_column || null,
          color_column: quickPlot.color_column || null,
          title: quickPlot.title || null
        })
      });

      if (response.ok) {
        const data = await response.json();
        setGeneratedPlots([{
          plot_type: quickPlot.type,
          plot_data: data.plot_data,
          plot_config: data.plot_config,
          config: { type: quickPlot.type, title: quickPlot.title }
        }]);
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error generating quick plot:', error);
      alert('Error generando gráfico rápido');
    } finally {
      setLoading(false);
    }
  };

  const generateCorrelationMatrix = async () => {
    if (!selectedDataset) {
      alert('Selecciona un dataset');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/neural-networks/analysis/dataset/${selectedDataset}/correlations?method=pearson`);
      
      if (response.ok) {
        const data = await response.json();
        setGeneratedPlots([{
          plot_type: 'correlation',
          plot_data: data.plot_data,
          plot_config: { displayModeBar: true },
          config: { type: 'correlation', title: 'Correlation Matrix' },
          correlation_data: data.correlation_matrix
        }]);
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error generating correlation matrix:', error);
      alert('Error generando matriz de correlación');
    } finally {
      setLoading(false);
    }
  };

  const renderDatasetOverview = () => {
    if (!datasetSummary || !columnInfo) return <div>Selecciona un dataset para ver el resumen</div>;

    return (
      <div style={{ display: 'grid', gap: '1rem' }}>
        {/* Información básica */}
        <div style={{ 
          backgroundColor: '#f8f9fa', 
          padding: '1rem', 
          borderRadius: '8px',
          border: '1px solid #dee2e6'
        }}>
          <h4 style={{ margin: '0 0 1rem 0', color: '#495057' }}>📊 Información Básica</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div>
              <strong>Dimensiones:</strong> {datasetSummary.basic_info.shape[0]} filas × {datasetSummary.basic_info.shape[1]} columnas
            </div>
            <div>
              <strong>Memoria:</strong> {(datasetSummary.basic_info.memory_usage / 1024 / 1024).toFixed(2)} MB
            </div>
            <div>
              <strong>Columnas numéricas:</strong> {columnInfo.numeric_columns.length}
            </div>
            <div>
              <strong>Columnas categóricas:</strong> {columnInfo.categorical_columns.length}
            </div>
          </div>
        </div>

        {/* Resumen de valores nulos */}
        <div style={{ 
          backgroundColor: '#fff3cd', 
          padding: '1rem', 
          borderRadius: '8px',
          border: '1px solid #ffeaa7'
        }}>
          <h4 style={{ margin: '0 0 1rem 0', color: '#856404' }}>⚠️ Valores Nulos</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.5rem' }}>
            {Object.entries(datasetSummary.basic_info.null_counts).map(([column, nullCount]) => (
              <div key={column} style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                padding: '0.25rem 0'
              }}>
                <span>{column}:</span>
                <span style={{ 
                  fontWeight: 'bold',
                  color: nullCount > 0 ? '#dc3545' : '#28a745'
                }}>
                  {nullCount}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Estadísticas numéricas */}
        {Object.keys(datasetSummary.numeric_summary).length > 0 && (
          <div style={{ 
            backgroundColor: '#d1ecf1', 
            padding: '1rem', 
            borderRadius: '8px',
            border: '1px solid #bee5eb'
          }}>
            <h4 style={{ margin: '0 0 1rem 0', color: '#0c5460' }}>📈 Estadísticas Numéricas</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                <thead>
                  <tr style={{ backgroundColor: '#b3d7e0' }}>
                    <th style={{ padding: '0.5rem', textAlign: 'left' }}>Columna</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right' }}>Media</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right' }}>Std</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right' }}>Min</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right' }}>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(datasetSummary.numeric_summary).map(([column, stats]) => (
                    <tr key={column}>
                      <td style={{ padding: '0.5rem', fontWeight: 'bold' }}>{column}</td>
                      <td style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'monospace' }}>
                        {stats.mean?.toFixed(4)}
                      </td>
                      <td style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'monospace' }}>
                        {stats.std?.toFixed(4)}
                      </td>
                      <td style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'monospace' }}>
                        {stats.min?.toFixed(4)}
                      </td>
                      <td style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'monospace' }}>
                        {stats.max?.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Resumen categórico */}
        {Object.keys(datasetSummary.categorical_summary).length > 0 && (
          <div style={{ 
            backgroundColor: '#d4edda', 
            padding: '1rem', 
            borderRadius: '8px',
            border: '1px solid #c3e6cb'
          }}>
            <h4 style={{ margin: '0 0 1rem 0', color: '#155724' }}>🏷️ Variables Categóricas</h4>
            <div style={{ display: 'grid', gap: '1rem' }}>
              {Object.entries(datasetSummary.categorical_summary).map(([column, info]) => (
                <div key={column} style={{ 
                  backgroundColor: '#ffffff', 
                  padding: '1rem', 
                  borderRadius: '4px',
                  border: '1px solid #c3e6cb'
                }}>
                  <h5 style={{ margin: '0 0 0.5rem 0' }}>{column}</h5>
                  <div style={{ marginBottom: '0.5rem' }}>
                    <strong>Valores únicos:</strong> {info.unique_count}
                  </div>
                  <div>
                    <strong>Top valores:</strong>
                    <div style={{ marginTop: '0.25rem', display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                      {Object.entries(info.value_counts).slice(0, 5).map(([value, count]) => (
                        <span key={value} style={{
                          backgroundColor: '#e9ecef',
                          padding: '0.25rem 0.5rem',
                          borderRadius: '4px',
                          fontSize: '0.75rem'
                        }}>
                          {value}: {count}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderPlotConfigurator = () => {
    return (
      <div style={{ display: 'grid', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h4 style={{ margin: 0 }}>🎨 Configurador de Gráficos</h4>
          <button
            onClick={addPlotConfig}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            + Agregar Gráfico
          </button>
        </div>

        {plotConfigs.map((config) => (
          <div key={config.id} style={{
            backgroundColor: '#f8f9fa',
            padding: '1rem',
            borderRadius: '8px',
            border: '1px solid #dee2e6'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h5 style={{ margin: 0 }}>Gráfico #{config.id}</h5>
              <button
                onClick={() => removePlotConfig(config.id)}
                style={{
                  padding: '0.25rem 0.5rem',
                  backgroundColor: '#dc3545',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                ✕
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                  Tipo de Gráfico:
                </label>
                <select
                  value={config.type}
                  onChange={(e) => updatePlotConfig(config.id, 'type', e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.375rem',
                    border: '1px solid #ced4da',
                    borderRadius: '4px'
                  }}
                >
                  {availablePlotTypes.map(plotType => (
                    <option key={plotType.type} value={plotType.type}>
                      {plotType.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                  Título:
                </label>
                <input
                  type="text"
                  value={config.title}
                  onChange={(e) => updatePlotConfig(config.id, 'title', e.target.value)}
                  placeholder="Título del gráfico"
                  style={{
                    width: '100%',
                    padding: '0.375rem',
                    border: '1px solid #ced4da',
                    borderRadius: '4px'
                  }}
                />
              </div>

              {/* Columnas específicas según el tipo de gráfico */}
              {(config.type === 'scatter' || config.type === 'line') && (
                <>
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                      Eje X:
                    </label>
                    <select
                      value={config.x_axis}
                      onChange={(e) => updatePlotConfig(config.id, 'x_axis', e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.375rem',
                        border: '1px solid #ced4da',
                        borderRadius: '4px'
                      }}
                    >
                      <option value="">Seleccionar columna</option>
                      {columnInfo?.columns && Object.keys(columnInfo.columns).map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                      Eje Y:
                    </label>
                    <select
                      value={config.y_axis}
                      onChange={(e) => updatePlotConfig(config.id, 'y_axis', e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.375rem',
                        border: '1px solid #ced4da',
                        borderRadius: '4px'
                      }}
                    >
                      <option value="">Seleccionar columna</option>
                      {columnInfo?.columns && Object.keys(columnInfo.columns).map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                </>
              )}

              {(config.type === 'histogram' || config.type === 'distribution') && (
                <div>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                    Columna:
                  </label>
                  <select
                    value={config.column}
                    onChange={(e) => updatePlotConfig(config.id, 'column', e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.375rem',
                      border: '1px solid #ced4da',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="">Seleccionar columna</option>
                    {columnInfo?.numeric_columns && columnInfo.numeric_columns.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              )}

              {(config.type === 'box' || config.type === 'violin') && (
                <div>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                    Eje Y:
                  </label>
                  <select
                    value={config.y_axis}
                    onChange={(e) => updatePlotConfig(config.id, 'y_axis', e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.375rem',
                      border: '1px solid #ced4da',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="">Seleccionar columna</option>
                    {columnInfo?.numeric_columns && columnInfo.numeric_columns.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              )}

              {config.type === 'correlation' && (
                <div>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                    Método:
                  </label>
                  <select
                    value={config.method}
                    onChange={(e) => updatePlotConfig(config.id, 'method', e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.375rem',
                      border: '1px solid #ced4da',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="pearson">Pearson</option>
                    <option value="spearman">Spearman</option>
                    <option value="kendall">Kendall</option>
                  </select>
                </div>
              )}

              {/* Opciones de color para scatter plots */}
              {config.type === 'scatter' && (
                <div>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                    Color por:
                  </label>
                  <select
                    value={config.color_by}
                    onChange={(e) => updatePlotConfig(config.id, 'color_by', e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.375rem',
                      border: '1px solid #ced4da',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="">Sin colorear</option>
                    {columnInfo?.columns && Object.keys(columnInfo.columns).map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          </div>
        ))}

        {plotConfigs.length > 0 && (
          <button
            onClick={generatePlots}
            disabled={loading}
            style={{
              padding: '0.75rem 1.5rem',
              backgroundColor: loading ? '#6c757d' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '1rem',
              fontWeight: 'bold'
            }}
          >
            {loading ? '🔄 Generando...' : '📊 Generar Gráficos'}
          </button>
        )}
      </div>
    );
  };

  const renderQuickAnalysis = () => {
    return (
      <div style={{ display: 'grid', gap: '1rem' }}>
        <h4 style={{ margin: 0 }}>⚡ Análisis Rápido</h4>
        
        {/* Gráfico rápido */}
        <div style={{
          backgroundColor: '#f8f9fa',
          padding: '1rem',
          borderRadius: '8px',
          border: '1px solid #dee2e6'
        }}>
          <h5 style={{ margin: '0 0 1rem 0' }}>Gráfico Rápido</h5>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                Tipo:
              </label>
              <select
                value={quickPlot.type}
                onChange={(e) => setQuickPlot({...quickPlot, type: e.target.value})}
                style={{
                  width: '100%',
                  padding: '0.375rem',
                  border: '1px solid #ced4da',
                  borderRadius: '4px'
                }}
              >
                {availablePlotTypes.map(plotType => (
                  <option key={plotType.type} value={plotType.type}>
                    {plotType.name}
                  </option>
                ))}
              </select>
            </div>

            {(quickPlot.type === 'scatter' || quickPlot.type === 'line') && (
              <>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                    Eje X:
                  </label>
                  <select
                    value={quickPlot.x_column}
                    onChange={(e) => setQuickPlot({...quickPlot, x_column: e.target.value})}
                    style={{
                      width: '100%',
                      padding: '0.375rem',
                      border: '1px solid #ced4da',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="">Seleccionar</option>
                    {columnInfo?.columns && Object.keys(columnInfo.columns).map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                    Eje Y:
                  </label>
                  <select
                    value={quickPlot.y_column}
                    onChange={(e) => setQuickPlot({...quickPlot, y_column: e.target.value})}
                    style={{
                      width: '100%',
                      padding: '0.375rem',
                      border: '1px solid #ced4da',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="">Seleccionar</option>
                    {columnInfo?.columns && Object.keys(columnInfo.columns).map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              </>
            )}

            <div>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                Título:
              </label>
              <input
                type="text"
                value={quickPlot.title}
                onChange={(e) => setQuickPlot({...quickPlot, title: e.target.value})}
                placeholder="Título opcional"
                style={{
                  width: '100%',
                  padding: '0.375rem',
                  border: '1px solid #ced4da',
                  borderRadius: '4px'
                }}
              />
            </div>
          </div>

          <button
            onClick={generateQuickPlot}
            disabled={loading}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: loading ? '#6c757d' : '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              marginRight: '0.5rem'
            }}
          >
            {loading ? '🔄 Generando...' : '⚡ Generar'}
          </button>
        </div>

        {/* Matriz de correlación rápida */}
        <div style={{
          backgroundColor: '#e3f2fd',
          padding: '1rem',
          borderRadius: '8px',
          border: '1px solid #bbdefb'
        }}>
          <h5 style={{ margin: '0 0 1rem 0' }}>🔗 Matriz de Correlación</h5>
          <p style={{ margin: '0 0 1rem 0', color: '#1565c0' }}>
            Genera automáticamente una matriz de correlación con todas las variables numéricas
          </p>
          <button
            onClick={generateCorrelationMatrix}
            disabled={loading || !columnInfo?.numeric_columns?.length}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: loading || !columnInfo?.numeric_columns?.length ? '#6c757d' : '#2196f3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading || !columnInfo?.numeric_columns?.length ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? '🔄 Generando...' : '🔗 Generar Correlaciones'}
          </button>
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: '1rem' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ margin: '0 0 1rem 0', color: '#343a40' }}>📊 Análisis de Datos</h2>
        
        {/* Selector de dataset */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
            Seleccionar Dataset:
          </label>
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            style={{
              width: '100%',
              maxWidth: '400px',
              padding: '0.5rem',
              border: '1px solid #ced4da',
              borderRadius: '4px',
              fontSize: '1rem'
            }}
          >
            <option value="">Seleccionar dataset...</option>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.filename} ({dataset.name})
              </option>
            ))}
          </select>
        </div>

        {/* Pestañas */}
        <div style={{ borderBottom: '2px solid #dee2e6', marginBottom: '1rem' }}>
          {[
            { id: 'overview', label: '📋 Resumen', icon: '📋' },
            { id: 'quick', label: '⚡ Análisis Rápido', icon: '⚡' },
            { id: 'advanced', label: '🎨 Configurador', icon: '🎨' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '0.75rem 1.5rem',
                marginRight: '0.25rem',
                backgroundColor: activeTab === tab.id ? '#007bff' : 'transparent',
                color: activeTab === tab.id ? 'white' : '#495057',
                border: 'none',
                borderBottom: activeTab === tab.id ? '2px solid #007bff' : '2px solid transparent',
                cursor: 'pointer',
                fontSize: '0.95rem',
                fontWeight: activeTab === tab.id ? 'bold' : 'normal'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Contenido de las pestañas */}
      {activeTab === 'overview' && renderDatasetOverview()}
      {activeTab === 'quick' && renderQuickAnalysis()}
      {activeTab === 'advanced' && renderPlotConfigurator()}

      {/* Visualización de gráficos generados */}
      {generatedPlots.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ margin: '0 0 1rem 0' }}>📈 Gráficos Generados</h3>
          <div style={{ display: 'grid', gap: '2rem' }}>
            {generatedPlots.map((plot, index) => (
              <div key={index} style={{
                backgroundColor: '#ffffff',
                padding: '1rem',
                borderRadius: '8px',
                border: '1px solid #dee2e6',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <h4 style={{ margin: '0 0 1rem 0', color: '#495057' }}>
                  {plot.config?.title || `${plot.plot_type.charAt(0).toUpperCase() + plot.plot_type.slice(1)} Plot`}
                </h4>
                <div style={{ width: '100%', height: '500px' }}>
                  <Plot
                    data={plot.plot_data.data}
                    layout={{
                      ...plot.plot_data.layout,
                      autosize: true,
                      responsive: true
                    }}
                    config={plot.plot_config}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler={true}
                  />
                </div>
                
                {/* Información adicional */}
                {plot.correlation_data && (
                  <div style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#6c757d' }}>
                    <strong>Matriz de correlación generada con {Object.keys(plot.correlation_data).length} variables</strong>
                  </div>
                )}
                
                {plot.statistics && (
                  <div style={{ marginTop: '1rem' }}>
                    <h5>Estadísticas:</h5>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.5rem', fontSize: '0.875rem' }}>
                      {Object.entries(plot.statistics).map(([stat, values]) => (
                        <div key={stat} style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <strong>{stat}:</strong>
                          <span>{Array.isArray(values) ? values.join(', ') : values}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DataAnalysis;