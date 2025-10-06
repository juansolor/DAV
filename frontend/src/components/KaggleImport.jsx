import React, { useState } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

export default function KaggleImport({ onImported }) {
  const [datasetName, setDatasetName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImport = async () => {
    if (!datasetName.trim()) return;
    setLoading(true); setError('');
    
    try {
      const res = await fetch(`${API_BASE}/external/kaggle/import?dataset=${encodeURIComponent(datasetName.trim())}`, {
        method: 'POST'
      });
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail || 'Error importando desde Kaggle');
      }
      const data = await res.json();
      onImported && onImported();
      setDatasetName('');
      alert(`✅ Importados ${data.length} archivos desde Kaggle`);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h4 style={{ marginBottom: '1rem', color: '#2c3e50' }}>🌐 Importar de Kaggle</h4>
      <div style={{ 
        border: '2px dashed #bdc3c7', 
        borderRadius: '8px', 
        padding: '1rem',
        backgroundColor: '#f9f9f9'
      }}>
        <input 
          type="text"
          placeholder="owner/dataset-name"
          value={datasetName}
          onChange={e => setDatasetName(e.target.value)}
          style={{
            width: '100%',
            padding: '0.5rem',
            border: '1px solid #bdc3c7',
            borderRadius: '4px',
            marginBottom: '0.5rem',
            boxSizing: 'border-box'
          }}
        />
        <p style={{ fontSize: '0.8rem', color: '#7f8c8d', margin: '0.5rem 0' }}>
          Ej: zynicide/wine-reviews
        </p>
        <button 
          disabled={!datasetName.trim() || loading} 
          onClick={handleImport}
          style={{
            backgroundColor: datasetName.trim() && !loading ? '#3498db' : '#95a5a6',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: '4px',
            cursor: datasetName.trim() && !loading ? 'pointer' : 'not-allowed',
            width: '100%'
          }}
        >
          {loading ? '⏳ Importando...' : '🔽 Importar'}
        </button>
      </div>
      {error && <p style={{color: '#e74c3c', fontSize: '0.9rem', marginTop: '0.5rem'}}>{error}</p>}
    </div>
  );
}