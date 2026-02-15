#!/usr/bin/env python3
"""
Prueba local del flujo Glue: solo conexión a RDS + creación de tablas (sin scraper).

Uso (desde la raíz del proyecto):
  export DB_HOST=tu-rds.amazonaws.com
  export DB_NAME=postgres
  export DB_USER=postgres
  export DB_PASSWORD=tu_password
  python test_glue_local.py

O con un .env o variables ya definidas. No hace falta Playwright ni ejecutar el scraper.
"""

import os
import sys

# Asegurar que el proyecto está en el path (igual que en Glue local)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    for key in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        if not os.environ.get(key):
            print(f"Falta variable de entorno: {key}")
            print("Ejemplo: export DB_HOST=xxx DB_NAME=postgres DB_USER=postgres DB_PASSWORD=xxx")
            sys.exit(1)

    print("Conectando a RDS y creando tablas si no existen...")
    from src.database.operations import DatabaseOperations

    db_ops = DatabaseOperations(use_postgres=True, chunk_size=5000)
    db_ops.ensure_tables()
    print("OK: Tablas verificadas/creadas. Conexión y DDL funcionan.")
    # Opcional: contar (debería ser 0 si está vacío)
    from src.database.models import User
    n = db_ops.count(User)
    print(f"Registros en 'users': {n}")

if __name__ == "__main__":
    main()
