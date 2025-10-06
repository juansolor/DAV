import React, { useState } from 'react';
import SummaryTable from './SummaryTable';
import CorrelationAnalysis from './CorrelationAnalysis';

const API_BASE = 'http://127.0.0.1:8000';

export default function AnalysisPanel({ dataset, analysisData, setAnalysisData }) {
  const [loading, setLoading] = useState({});
  const [activeAnalysis, setActiveAnalysis] = useState('basic');

  const runAnalysis = async (type) => {
    setLoading(prev => ({ ...prev, [type]: true }));
    try {
      const res = await fetch(`${API_BASE}/analyze/${type}/${dataset.id}`);
      if (res.ok) {
        const data = await res.json();
        setAnalysisData(prev => ({ ...prev, [type]: data }));
      } else {
        const error = await res.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (e) {
      alert(`Error: ${e.message}`);
    } finally {
      setLoading(prev => ({ ...prev, [type]: false }));
    }
  };

  const loadPreview = async () => {
    setLoading(prev => ({ ...prev, preview: true }));
    try {
      const res = await fetch(`${API_BASE}/datasets/${dataset.id}/preview?n=20`);
      if (res.ok) {
        const data = await res.json();
        setAnalysisData(prev => ({ ...prev, preview: data }));
      }
    } catch (e) {
      alert(`Error: ${e.message}`);
    } finally {
      setLoading(prev => ({ ...prev, preview: false }));
    }
  };

  const analysisTypes = [
    { key: 'basic', label: '📊 Estadísticas', desc: 'Resumen estadístico básico' },
    { key: 'correlation', label: '🔗 Correlaciones', desc: 'Matriz de correlaciones' },
    { key: 'missing', label: '❓ Datos faltantes', desc: 'Análisis de valores nulos' },
    { key: 'preview', label: '👁️ Preview', desc: 'Primeras 20 filas' }
  ];

  return (
    <div>
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ color: '#2c3e50', margin: 0 }}>📊 {dataset.name}</h2>
        <div style={{ color: '#7f8c8d', fontSize: '0.9rem', marginTop: '0.5rem' }}>
          {dataset.n_rows} filas × {dataset.n_cols} columnas
        </div>
      </div>

      {/* Analysis Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        {analysisTypes.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveAnalysis(key)}
            style={{
              padding: '0.5rem 1rem',
              border: 'none',
              borderRadius: '6px',
              backgroundColor: activeAnalysis === key ? '#3498db' : '#ecf0f1',
              color: activeAnalysis === key ? 'white' : '#2c3e50',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Run Analysis Button */}
      <div style={{ marginBottom: '1.5rem' }}>
        {activeAnalysis === 'preview' ? (
          <button
            onClick={loadPreview}
            disabled={loading.preview}
            style={{
              backgroundColor: '#27ae60',
              color: 'white',
              border: 'none',
              padding: '0.75rem 1.5rem',
              borderRadius: '6px',
              cursor: loading.preview ? 'not-allowed' : 'pointer'
            }}
          >
            {loading.preview ? '⏳ Cargando...' : '▶️ Cargar Preview'}
          </button>
        ) : (
          <button
            onClick={() => runAnalysis(activeAnalysis)}
            disabled={loading[activeAnalysis]}
            style={{
              backgroundColor: '#3498db',
              color: 'white',
              border: 'none',
              padding: '0.75rem 1.5rem',
              borderRadius: '6px',
              cursor: loading[activeAnalysis] ? 'not-allowed' : 'pointer'
            }}
          >
            {loading[activeAnalysis] ? '⏳ Analizando...' : '▶️ Ejecutar Análisis'}
          </button>
        )}
      </div>

      {/* Results */}
      <div style={{ backgroundColor: '#f8f9fa', padding: '1.5rem', borderRadius: '8px' }}>
        {activeAnalysis === 'basic' && analysisData.basic && (
          <div>
            <h3>📊 Estadísticas descriptivas</h3>
            <SummaryTable summary={analysisData.basic.summary} />
          </div>
        )}

        {activeAnalysis === 'correlation' && (
          <CorrelationAnalysis 
            dataset={dataset} 
            analysisData={analysisData.correlation}
            setAnalysisData={setAnalysisData}
            loading={loading.correlation}
            setLoading={setLoading}
          />
        )}

        {activeAnalysis === 'missing' && analysisData.missing && (
          <div>
            <h3>❓ Datos faltantes</h3>
            <div style={{ display: 'grid', gap: '0.5rem' }}>
              {Object.entries(analysisData.missing.missing_percentages).map(([col, pct]) => (
                <div key={col} style={{ 
                  backgroundColor: 'white', 
                  padding: '0.5rem', 
                  borderRadius: '4px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  border: pct > 20 ? '2px solid #e74c3c' : pct > 5 ? '2px solid #f39c12' : '1px solid #bdc3c7'
                }}>
                  <span><strong>{col}</strong></span>
                  <span>{pct}% ({analysisData.missing.missing_counts[col]} valores)</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeAnalysis === 'preview' && analysisData.preview && (
          <div>
            <h3>👁️ Preview del dataset</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ 
                width: '100%', 
                borderCollapse: 'collapse', 
                backgroundColor: 'white',
                fontSize: '0.85rem'
              }}>
                <thead>
                  <tr style={{ backgroundColor: '#34495e', color: 'white' }}>
                    {analysisData.preview.preview_rows.length > 0 && 
                      Object.keys(analysisData.preview.preview_rows[0]).map(col => (
                        <th key={col} style={{ padding: '0.5rem', textAlign: 'left' }}>{col}</th>
                      ))
                    }
                  </tr>
                </thead>
                <tbody>
                  {analysisData.preview.preview_rows.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #ecf0f1' }}>
                      {Object.values(row).map((val, j) => (
                        <td key={j} style={{ padding: '0.5rem' }}>
                          {val === null || val === undefined ? (
                            <span style={{ color: '#e74c3c', fontStyle: 'italic' }}>null</span>
                          ) : (
                            String(val)
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {!analysisData[activeAnalysis] && (
          <div style={{ textAlign: 'center', color: '#7f8c8d', padding: '2rem' }}>
            <p>👆 Ejecuta el análisis para ver los resultados aquí</p>
          </div>
        )}
      </div>
    </div>
  );
}