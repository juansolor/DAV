# Data Analytics Integrator

Plataforma fullstack para análisis exploratorio, correlaciones y visualización de datasets (CSV, Excel, Kaggle, etc.)

## Estructura del proyecto

```
DAV/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── app/
│   │   ├── routers/
│   │   ├── db/
│   │   ├── core/
│   │   └── ...
│   └── data/uploads/   # (ignorado por git)
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
├── .gitignore
├── .env.example
└── README.md
```

## Primeros pasos

### 1. Clonar el repositorio
```bash
git clone https://github.com/juansolor/DAV.git
cd DAV
```

### 2. Configurar entorno backend (Python)
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # (Windows)
pip install -r requirements.txt
```

### 3. Configurar entorno frontend (Node/React)
```bash
cd ../frontend
npm install
```

### 4. Variables de entorno

Copia `.env.example` a `.env` y ajusta según tus credenciales/local.

### 5. Levantar servidores

**Backend:**
```bash
cd backend
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm run dev
```

Abre: http://localhost:5173

## Funcionalidades principales
- Subida de archivos CSV, TXT, Excel (.xlsx, .xls)
- Importación directa desde Kaggle
- Análisis estadístico, correlaciones (Pearson, Spearman, Kendall)
- Análisis de datos faltantes
- Vista previa de datasets
- Interfaz React moderna y responsiva

## Buenas prácticas
- No subir archivos de datos reales ni credenciales (`.env`, `backend/data/uploads/` están en `.gitignore`)
- Usa ramas para nuevas features
- Haz PRs descriptivos

## CI/CD y Deployment

### GitHub Actions
El proyecto incluye un pipeline de CI/CD que ejecuta:
- **Linting**: ruff (Python) + eslint (JavaScript)
- **Testing**: pytest (backend) + placeholder frontend
- **Security**: bandit security scan
- **Build**: Docker images multi-arquitectura
- **Deploy**: automático en push a main

### Docker
```bash
# Desarrollo local con Docker Compose
docker-compose up --build

# Build manual
docker build -t data-analytics-backend ./backend
docker build -t data-analytics-frontend ./frontend
```

### Variables de entorno (GitHub Secrets)
Para el deployment automático con Docker Hub, configurar:
- `DOCKERHUB_USERNAME`: usuario de Docker Hub
- `DOCKERHUB_TOKEN`: token de acceso

**Nota**: Los secrets son opcionales. El pipeline funciona sin ellos, solo omite el push a Docker Hub.

### Testing
```bash
# Backend
cd backend
pytest tests/ -v

# Frontend
cd frontend
npm test
```

## Badges
![CI/CD](https://github.com/juansolor/DAV/workflows/CI/CD%20Pipeline/badge.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![React](https://img.shields.io/badge/react-18-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Licencia
MIT
