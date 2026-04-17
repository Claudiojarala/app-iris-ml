
import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd 

# --- Configuración de Base de Datos ---
USER = "postgres.hzzukkgqmvgdbbmpuvgv"
PASSWORD = "Facundo12*12"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

def save_prediction(sepal_l, sepal_w, petal_l, petal_w, prediction, confidence):
    try:
        conn = get_connection()
        cur = conn.cursor()
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
        st.error(f"Error al guardar: {e}")

def get_history():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, created_at FROM iris_history ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        return []

# --- Configuración de la página ---
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸", layout="wide")

@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except Exception:
        st.error("Archivos del modelo no encontrados.")
        return None, None, None

st.title("🌸 Predictor de Especies de Iris")

model, scaler, model_info = load_models()

if model is not None:
    # CORRECCIÓN AQUÍ: Definimos explícitamente 2 columnas
    col1, col2 = st.columns(2) 

    with col1:
        st.header("Entrada de Datos")
        sepal_length = st.number_input("Longitud Sépalo", 0.0, 10.0, 5.0)
        sepal_width = st.number_input("Ancho Sépalo", 0.0, 10.0, 3.0)
        petal_length = st.number_input("Longitud Pétalo", 0.0, 10.0, 4.0)
        petal_width = st.number_input("Ancho Pétalo", 0.0, 10.0, 1.0)
        
        if st.button("Predecir", use_container_width=True):
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            features_scaled = scaler.transform(features)
            
            prediction_idx = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            
            target_names = model_info['target_names']
            predicted_species = target_names[prediction_idx]
            confidence = float(max(probabilities))
            
            # Guardar en DB
            save_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence)
            
            st.success(f"**{predicted_species}** ({confidence:.1%})")

    with col2:
        st.header("Historial (Recientes primero)")
        data = get_history()
        if data:
            df = pd.DataFrame(data, columns=["SepalL", "SepalW", "PetalL", "PetalW", "Especie", "Confianza", "Fecha"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay registros aún.")
