import React, { useState, useEffect } from 'react';
import UploadArea from './components/UploadArea';
import KaggleImport from './components/KaggleImport';
import DatasetCard from './components/DatasetCard';
import AnalysisPanel from './components/AnalysisPanel';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [analysisData, setAnalysisData] = useState({});
  const [activeTab, setActiveTab] = useState('upload');

  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${API_BASE}/datasets/`);
      if (res.ok) {
        const data = await res.json();
        setDatasets(data);
      }
    } catch (e) {
      console.error('Error fetching datasets:', e);
    }
  };

  const selectDataset = (dataset) => {
    setSelectedDataset(dataset);
    setAnalysisData({}); // Reset analysis
  };

  useEffect(() => { fetchDatasets(); }, []);

  return (
    <div style={{ fontFamily: 'system-ui', minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      <header style={{ backgroundColor: '#2c3e50', color: 'white', padding: '1rem 2rem' }}>
        <h1 style={{ margin: 0 }}>🔬 Data Analytics Integrator</h1>
        <p style={{ margin: '0.5rem 0 0 0', opacity: 0.8 }}>Análisis inteligente de datasets</p>
      </header>

      <div style={{ display: 'flex', gap: '1rem', padding: '1rem' }}>
        {/* Sidebar */}
        <div style={{ width: '300px', backgroundColor: 'white', borderRadius: '8px', padding: '1rem', height: 'fit-content', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
          
          {/* Tabs */}
          <div style={{ display: 'flex', marginBottom: '1rem' }}>
            <button 
              onClick={() => setActiveTab('upload')}
              style={{ 
                flex: 1, padding: '0.5rem', border: 'none', 
                backgroundColor: activeTab === 'upload' ? '#3498db' : '#ecf0f1',
                color: activeTab === 'upload' ? 'white' : '#2c3e50',
                cursor: 'pointer', borderRadius: '4px 0 0 4px'
              }}
            >
              📁 Subir
            </button>
            <button 
              onClick={() => setActiveTab('kaggle')}
              style={{ 
                flex: 1, padding: '0.5rem', border: 'none',
                backgroundColor: activeTab === 'kaggle' ? '#3498db' : '#ecf0f1',
                color: activeTab === 'kaggle' ? 'white' : '#2c3e50',
                cursor: 'pointer', borderRadius: '0 4px 4px 0'
              }}
            >
              🌐 Kaggle
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'upload' && <UploadArea onUploaded={fetchDatasets} />}
          {activeTab === 'kaggle' && <KaggleImport onImported={fetchDatasets} />}

          {/* Datasets List */}
          <div style={{ marginTop: '1.5rem' }}>
            <h3 style={{ color: '#2c3e50', marginBottom: '1rem' }}>📊 Datasets ({datasets.length})</h3>
            {datasets.length === 0 ? (
              <p style={{ color: '#7f8c8d', fontStyle: 'italic' }}>No hay datasets cargados</p>
            ) : (
              <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                {datasets.map(d => (
                  <DatasetCard 
                    key={d.id} 
                    dataset={d} 
                    isSelected={selectedDataset?.id === d.id}
                    onClick={() => selectDataset(d)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div style={{ flex: 1, backgroundColor: 'white', borderRadius: '8px', padding: '1.5rem', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
          {selectedDataset ? (
            <AnalysisPanel 
              dataset={selectedDataset} 
              analysisData={analysisData}
              setAnalysisData={setAnalysisData}
            />
          ) : (
            <div style={{ textAlign: 'center', color: '#7f8c8d', marginTop: '4rem' }}>
              <h2>👈 Selecciona un dataset para análisis</h2>
              <p>Una vez seleccionado, podrás ver estadísticas, correlaciones y datos faltantes.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
