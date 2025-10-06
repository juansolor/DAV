# 🧠 Data Analytics Platform with Neural Networks

Una plataforma completa de análisis de datos con capacidades avanzadas de machine learning y redes neuronales, construida con **FastAPI** y **React**.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2+-blue.svg)](https://reactjs.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

## 🚀 Características Principales

### 📊 **Análisis de Datos**
- ✅ Carga de datasets (CSV, Excel, JSON)
- ✅ Análisis estadístico automático
- ✅ Visualizaciones interactivas
- ✅ Limpieza y preprocesamiento de datos

### 🧠 **Machine Learning & Redes Neuronales**
- ✅ **Clasificación** con TensorFlow/Keras
- ✅ **Regresión** con redes neuronales profundas
- ✅ **Series Temporales** con Darts (LSTM, GRU, N-BEATS, Transformer)
- ✅ Entrenamiento automático con hiperparámetros configurables
- ✅ Evaluación y métricas de modelos
- ✅ Predicciones en tiempo real

### 🌐 **Integración Externa**
- ✅ Importación desde **Kaggle API**
- ✅ Soporte para múltiples formatos de datos
- ✅ API REST completa y documentada

### 🎯 **Interfaz de Usuario**
- ✅ Dashboard interactivo en React
- ✅ Gestión visual de modelos
- ✅ Interfaz de entrenamiento de IA
- ✅ Visualización de resultados en tiempo real

## 🏗️ Arquitectura

```
📁 Data Analytics Platform
├── 🔧 backend/          # FastAPI + Python
│   ├── app/
│   │   ├── routers/     # API endpoints
│   │   │   ├── datasets.py
│   │   │   ├── analyze.py
│   │   │   ├── external.py
│   │   │   └── neural_networks.py  # 🧠 ML/AI Module
│   │   ├── services/    # Lógica de negocio
│   │   │   └── neural_networks.py  # ML Service Layer
│   │   ├── db/          # Base de datos & modelos
│   │   │   └── models.py  # SQLAlchemy models
│   │   └── schemas/     # Validación Pydantic
│   ├── models/          # Modelos ML entrenados
│   ├── tests/           # Tests automatizados
│   └── requirements.txt # Dependencias Python
│
├── 🎨 frontend/         # React + Vite
│   ├── src/
│   │   ├── components/  # Componentes React
│   │   │   ├── Dashboard.jsx
│   │   │   ├── DatasetManager.jsx
│   │   │   └── NeuralNetworks.jsx  # 🧠 ML Interface
│   │   └── services/    # API client
│   ├── package.json     # Dependencias Node.js
│   └── public/          # Assets estáticos
│
└── 📚 README.md         # Documentación principal
```

## ⚡ Instalación Rápida

### 1. **Clonar el repositorio**
```bash
git clone https://github.com/juansolor/DAV.git
cd DAV
```

### 2. **Backend Setup (Python + FastAPI)**
```bash
cd backend

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate     # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Inicializar base de datos
python init_db.py

# Arrancar servidor
uvicorn main:app --reload
```

### 3. **Frontend Setup (React + Vite)**
```bash
cd ../frontend

# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm run dev
```

### 4. **Acceder a la aplicación**
- 🌐 **Frontend:** http://localhost:5173
- 📚 **API Docs:** http://localhost:8000/docs
- 🔍 **API Redoc:** http://localhost:8000/redoc

## 🧠 Módulo de Redes Neuronales

### Endpoints Disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/neural-networks/classification/train` | Entrenar modelo de clasificación |
| `POST` | `/neural-networks/regression/train` | Entrenar modelo de regresión |
| `POST` | `/neural-networks/timeseries/train` | Entrenar modelo de series temporales |
| `GET` | `/neural-networks/models` | Listar modelos entrenados |
| `POST` | `/neural-networks/predict/{model_id}` | Realizar predicciones |
| `DELETE` | `/neural-networks/models/{model_id}` | Eliminar modelo |

### Ejemplos de Uso

#### 🎯 Entrenar Modelo de Clasificación
```json
POST /neural-networks/classification/train
{
  "dataset_id": 1,
  "target_column": "species",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "hidden_layers": [128, 64, 32],
  "dropout_rate": 0.2
}
```

#### 📈 Entrenar Modelo de Regresión
```json
POST /neural-networks/regression/train
{
  "dataset_id": 2,
  "target_column": "price",
  "epochs": 150,
  "hidden_layers": [256, 128, 64],
  "learning_rate": 0.0005
}
```

#### 📅 Series Temporales con Darts
```json
POST /neural-networks/timeseries/train
{
  "dataset_id": 3,
  "target_column": "sales",
  "date_column": "date",
  "model_type": "LSTM",
  "input_chunk_length": 12,
  "n_epochs": 200
}
```

## 🛠️ Stack Tecnológico

### Backend
- **Framework:** FastAPI 0.104+
- **Base de Datos:** SQLAlchemy + SQLite/PostgreSQL
- **ML/AI:** TensorFlow 2.13+, PyTorch 2.0+, Scikit-learn
- **Series Temporales:** Darts 0.25+
- **Visualización:** Matplotlib, Seaborn, Plotly
- **Testing:** Pytest + AsyncIO

### Frontend
- **Framework:** React 18.2 + Vite 5.0
- **Styling:** CSS Modules + Vanilla CSS
- **HTTP Client:** Fetch API nativo
- **Charts:** Plotly.js (integración futura)

### Machine Learning
- **Deep Learning:** TensorFlow/Keras, PyTorch Lightning
- **Classical ML:** Scikit-learn
- **Time Series:** Darts (N-BEATS, NHiTS, TCN, Transformer)
- **Interpretability:** SHAP
- **Metrics:** Accuracy, Precision, Recall, F1, MAPE, SMAPE

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Ejecutar tests específicos
pytest tests/test_neural_networks.py -v

# Tests con cobertura
pytest --cov=app tests/
```

## 🚀 Deployment

### Variables de entorno
Copia `.env.example` a `.env` y configura:
```bash
DATABASE_URL=sqlite:///./database.db
KAGGLE_USERNAME=tu_usuario_kaggle
KAGGLE_KEY=tu_api_key_kaggle
```

### Docker (Próximamente)
```bash
# Desarrollo local con Docker Compose
docker-compose up --build
```

### Manual
```bash
# Backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
npm run build
# Servir archivos estáticos
```

## 📁 Funcionalidades Principales

### 📊 **Análisis de Datos**
- ✅ Subida de archivos CSV, Excel, JSON
- ✅ Importación directa desde Kaggle API
- ✅ Análisis estadístico automático
- ✅ Correlaciones (Pearson, Spearman, Kendall)
- ✅ Detección de datos faltantes
- ✅ Visualizaciones interactivas

### 🧠 **Machine Learning**
- ✅ Clasificación multiclase y binaria
- ✅ Regresión con redes neuronales
- ✅ Series temporales avanzadas
- ✅ Hyperparameter tuning
- ✅ Model evaluation & metrics
- ✅ Real-time predictions

## 🎯 Roadmap

### 🔮 Próximas Funcionalidades
- [ ] **AutoML** - Optimización automática de hiperparámetros
- [ ] **Model Registry** - Versionado de modelos
- [ ] **Real-time Streaming** - Predicciones en tiempo real
- [ ] **A/B Testing** - Comparación de modelos
- [ ] **Advanced Visualizations** - Dashboard con D3.js
- [ ] **Multi-tenancy** - Soporte multiusuario

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📜 Licencia

Este proyecto está bajo la Licencia MIT.

---

**🚀 ¡Comienza a entrenar tus modelos de IA hoy mismo!**
