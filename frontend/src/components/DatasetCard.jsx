import React from 'react';

export default function DatasetCard({ dataset, isSelected, onClick }) {
  const formatFileSize = (bytes) => {
    if (!bytes) return 'N/A';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('es-ES', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getFileTypeIcon = (name) => {
    if (name.includes('.xlsx') || name.includes('.xls') || name.includes(':')) {
      return '📊'; // Excel
    } else if (name.includes('.csv')) {
      return '📋'; // CSV
    } else if (name.includes('kaggle')) {
      return '🌐'; // Kaggle
    }
    return '📄'; // Default
  };

  const getDatasetType = (name) => {
    if (name.includes(':') && !name.includes('kaggle')) {
      return 'Excel Sheet';
    } else if (name.includes('kaggle')) {
      return 'Kaggle';
    } else {
      return 'CSV';
    }
  };

  return (
    <div 
      onClick={onClick}
      style={{
        border: isSelected ? '2px solid #3498db' : '1px solid #ecf0f1',
        borderRadius: '6px',
        padding: '0.75rem',
        marginBottom: '0.5rem',
        cursor: 'pointer',
        backgroundColor: isSelected ? '#ebf3fd' : '#fafafa',
        transition: 'all 0.2s ease'
      }}
      onMouseEnter={e => {
        if (!isSelected) e.target.style.backgroundColor = '#f0f0f0';
      }}
      onMouseLeave={e => {
        if (!isSelected) e.target.style.backgroundColor = '#fafafa';
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.25rem' }}>
        <span style={{ fontSize: '1.2rem', marginRight: '0.5rem' }}>
          {getFileTypeIcon(dataset.name)}
        </span>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 'bold', color: '#2c3e50', fontSize: '0.9rem' }}>
            {dataset.name.length > 25 ? `${dataset.name.substring(0, 25)}...` : dataset.name}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#95a5a6' }}>
            {getDatasetType(dataset.name)}
          </div>
        </div>
      </div>
      <div style={{ fontSize: '0.8rem', color: '#7f8c8d' }}>
        📊 {dataset.n_rows || 'N/A'} filas × {dataset.n_cols || 'N/A'} cols
      </div>
      <div style={{ fontSize: '0.8rem', color: '#7f8c8d' }}>
        📁 {formatFileSize(dataset.file_size)}
      </div>
      {dataset.created_at && (
        <div style={{ fontSize: '0.75rem', color: '#95a5a6', marginTop: '0.25rem' }}>
          🕒 {formatDate(dataset.created_at)}
        </div>
      )}
    </div>
  );
}