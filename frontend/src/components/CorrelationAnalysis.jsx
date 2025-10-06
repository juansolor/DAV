import React, { useState } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

export default function CorrelationAnalysis({ dataset, analysisData, setAnalysisData, loading, setLoading }) {
  const [method, setMethod] = useState('pearson');
  const [threshold, setThreshold] = useState(0.7);
  const [includeCategorical, setIncludeCategorical] = useState(false);

  const runCorrelationAnalysis = async () => {
    setLoading(prev => ({ ...prev, correlation: true }));
    try {
      const params = new URLSearchParams({
        threshold: threshold.toString(),
        method: method,
        include_categorical: includeCategorical.toString()
      });
      const res = await fetch(`${API_BASE}/analyze/correlation/${dataset.id}?${params}`);
      if (res.ok) {
        const data = await res.json();
        setAnalysisData(prev => ({ ...prev, correlation: data }));
      } else {
        const error = await res.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (e) {
      alert(`Error: ${e.message}`);
    } finally {
      setLoading(prev => ({ ...prev, correlation: false }));
    }
  };

  return (
    <div>
      <h3>🔗 Análisis de correlaciones</h3>
      
      {/* Controles de configuración */}
      <div style={{ 
        backgroundColor: '#ecf0f1', 
        padding: '1rem', 
        borderRadius: '6px', 
        marginBottom: '1rem',
        display: 'grid',
        gap: '1rem',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))'
      }}>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold', color: '#2c3e50' }}>
            📊 Método de correlación:
          </label>
          <select 
            value={method} 
            onChange={(e) => setMethod(e.target.value)}
            style={{
              width: '100%',
              padding: '0.5rem',
              border: '1px solid #bdc3c7',
              borderRadius: '4px',
              backgroundColor: 'white'
            }}
          >
            <option value="pearson">Pearson (lineal)</option>
            <option value="spearman">Spearman (ordinal)</option>
            <option value="kendall">Kendall (tau)</option>
          </select>
          <div style={{ fontSize: '0.8rem', color: '#7f8c8d', marginTop: '0.25rem' }}>
            {method === 'pearson' && 'Correlación lineal estándar'}
            {method === 'spearman' && 'Correlación de rangos (no lineal)'}
            {method === 'kendall' && 'Correlación robusta tau de Kendall'}
          </div>
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold', color: '#2c3e50' }}>
            🎯 Umbral de correlación:
          </label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#7f8c8d' }}>
            <span>0.1</span>
            <span><strong>{threshold}</strong></span>
            <span>1.0</span>
          </div>
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold', color: '#2c3e50' }}>
            🔢 Tipos de datos:
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input
              type="checkbox"
              id="include-categorical"
              checked={includeCategorical}
              onChange={(e) => setIncludeCategorical(e.target.checked)}
            />
            <label htmlFor="include-categorical" style={{ fontSize: '0.9rem', color: '#2c3e50' }}>
              Incluir categóricas codificadas
            </label>
          </div>
          <div style={{ fontSize: '0.8rem', color: '#7f8c8d', marginTop: '0.25rem' }}>
            {includeCategorical ? 'Numéricos + categóricos codificados' : 'Solo variables numéricas'}
          </div>
        </div>
      </div>

      {/* Botón ejecutar */}
      <button
        onClick={runCorrelationAnalysis}
        disabled={loading}
        style={{
          backgroundColor: '#3498db',
          color: 'white',
          border: 'none',
          padding: '0.75rem 1.5rem',
          borderRadius: '6px',
          cursor: loading ? 'not-allowed' : 'pointer',
          marginBottom: '1.5rem'
        }}
      >
        {loading ? '⏳ Analizando...' : '▶️ Ejecutar Análisis de Correlación'}
      </button>

      {/* Resultados */}
      {analysisData && (
        <div>
          <div style={{ 
            backgroundColor: '#d5dbdb', 
            padding: '0.75rem', 
            borderRadius: '4px', 
            marginBottom: '1rem',
            fontSize: '0.9rem'
          }}>
            <strong>Configuración:</strong> {analysisData.method} | 
            Umbral: {threshold} | 
            Tipos: {analysisData.data_types_used.join(', ')} | 
            Columnas: {analysisData.columns_analyzed.length}
          </div>

          {analysisData.high_correlations.length > 0 ? (
            <div>
              <h4>Correlaciones altas (≥{threshold}):</h4>
              <div style={{ display: 'grid', gap: '0.5rem' }}>
                {analysisData.high_correlations.map((corr, i) => (
                  <div key={i} style={{ 
                    backgroundColor: 'white', 
                    padding: '0.75rem', 
                    borderRadius: '4px',
                    border: `2px solid ${corr.strength === 'strong' ? '#e74c3c' : '#f39c12'}`,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}>
                    <div>
                      <strong>{corr.var1}</strong> ↔ <strong>{corr.var2}</strong>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ 
                        fontSize: '1.1rem', 
                        fontWeight: 'bold',
                        color: corr.correlation > 0 ? '#27ae60' : '#e74c3c'
                      }}>
                        {corr.correlation}
                      </span>
                      <span style={{ 
                        fontSize: '0.8rem',
                        color: corr.strength === 'strong' ? '#e74c3c' : '#f39c12',
                        backgroundColor: corr.strength === 'strong' ? '#fadbd8' : '#fef9e7',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '12px'
                      }}>
                        {corr.strength}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ 
              textAlign: 'center', 
              padding: '2rem', 
              color: '#7f8c8d', 
              backgroundColor: 'white',
              borderRadius: '4px',
              border: '1px dashed #bdc3c7'
            }}>
              <p>🔍 No se encontraron correlaciones ≥ {threshold}</p>
              <p style={{ fontSize: '0.9rem' }}>Prueba reducir el umbral o cambiar el método</p>
            </div>
          )}

          {/* Información adicional */}
          <div style={{ 
            marginTop: '1rem', 
            fontSize: '0.85rem', 
            color: '#7f8c8d',
            backgroundColor: '#f8f9fa',
            padding: '0.75rem',
            borderRadius: '4px'
          }}>
            <strong>Columnas analizadas ({analysisData.columns_analyzed.length}):</strong> {analysisData.columns_analyzed.join(', ')}
          </div>
        </div>
      )}
    </div>
  );
}