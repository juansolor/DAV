from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import datasets, analyze, external

app = FastAPI(title="Data Analytics Integrator", version="0.1.0")

# 🔐 CORS (en producción restringir dominios)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: parametrizar
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"])
def root():
    return {"message": "Backend Data Analysis listo 🚀", "version": app.version}

# Rutas de dominio
app.include_router(datasets.router)
app.include_router(analyze.router)
app.include_router(external.router)

# Punto de entrada uvicorn (ej: uvicorn backend.main:app --reload)
