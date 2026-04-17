import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd

# --- Configuración de Base de Datos (Supabase) ---
# He mantenido tus credenciales según los archivos proporcionados
USER = "postgres.hzzukkgqmvgdbbmpuvgv"
PASSWORD = "Facundo12*12"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

def get_connection():
    """Establece la conexión con PostgreSQL."""
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

def save_prediction(sepal_l, sepal_w, petal_l, petal_w, prediction, confidence):
    """Inserta los datos de entrada y el resultado en la tabla."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Asegura que la tabla exista con una columna de fecha para el historial
        cur.execute("""
            CREATE TABLE IF NOT EXISTS iris_history (
                id SERIAL PRIMARY KEY,
                sepal_length FLOAT,
                sepal_width FLOAT,
                petal_length FLOAT,
                petal_width FLOAT,
                prediction TEXT,
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        query = """
            INSERT INTO iris_history (sepal_length, sepal_width, petal_length, petal_width, prediction, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (sepal_l, sepal_w, petal_l, petal_w, prediction, confidence))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error al guardar en BD: {e}")

def get_history():
    """Obtiene los registros ordenados por fecha descendente."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT sepal_length,
