import psycopg2
import os

# Configuración de la conexión a PostgreSQL
def connect_db():
    return psycopg2.connect(
        host=os.getenv("HOST_DB_PROCESS"),
        port=os.getenv("PORT_DB_PROCESS"),
        user=os.getenv("DB_USER"),
        password=os.getenv("PASSWORD_DB_PROCESS"),
        database=os.getenv("DATABASE_DB_PROCESS")
        )
