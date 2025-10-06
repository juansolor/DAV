"""
Script para inicializar la base de datos con todas las tablas
"""
import sys
import os

# Agregar el directorio backend al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.db.session import engine
from app.db.models import Base

def init_db():
    """Crear todas las tablas en la base de datos"""
    print("Creando tablas en la base de datos...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tablas creadas exitosamente")
    print("📊 Tablas disponibles:")
    print("   - datasets")
    print("   - neural_network_models")

if __name__ == "__main__":
    init_db()