import React, { useState } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

export default function UploadArea({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true); setError('');
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/datasets/upload`, { method: 'POST', body: form });
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail || 'Error subiendo archivo');
      }
      const result = await res.json();
      const count = Array.isArray(result) ? result.length : 1;
      if (count > 1) {
        alert(`✅ Se cargaron ${count} hojas del archivo Excel`);
      }
      onUploaded && onUploaded();
      setFile(null);
      // Reset input
      document.getElementById('file-input').value = '';
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h4 style={{ marginBottom: '1rem', color: '#2c3e50' }}>📁 Subir archivo</h4>
      <div style={{ 
        border: '2px dashed #bdc3c7', 
        borderRadius: '8px', 
        padding: '1rem', 
        textAlign: 'center',
        backgroundColor: file ? '#e8f5e8' : '#f9f9f9'
      }}>
        <input 
          id="file-input"
          type="file" 
          accept=".csv,.txt,.xlsx,.xls" 
          onChange={e => setFile(e.target.files[0])}
          style={{ marginBottom: '0.5rem' }}
        />
        <div style={{ fontSize: '0.85rem', color: '#7f8c8d', margin: '0.5rem 0' }}>
          Formatos: CSV, TXT, Excel (.xlsx, .xls)
        </div>
        <button 
          disabled={!file || loading} 
          onClick={handleUpload}
          style={{
            backgroundColor: file && !loading ? '#27ae60' : '#95a5a6',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: '4px',
            cursor: file && !loading ? 'pointer' : 'not-allowed'
          }}
        >
          {loading ? '⏳ Subiendo...' : '📤 Subir archivo'}
        </button>
      </div>
      {error && <p style={{color: '#e74c3c', fontSize: '0.9rem', marginTop: '0.5rem'}}>{error}</p>}
    </div>
  );
}
