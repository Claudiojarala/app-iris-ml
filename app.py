import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd

# --- Configuración de Base de Datos (Supabase/PostgreSQL) ---
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
        # Crear la tabla si no existe
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
        # Insertar los valores ingresados y la predicción
        query = """
            INSERT INTO iris_history (sepal_length, sepal_width, petal_length, petal_width, prediction, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (sepal_l, sepal_w, petal_l, petal_w, prediction, confidence))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error al guardar en la base de datos: {e}")

def get_history():
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Consulta para obtener el historial en orden descendente por fecha
        cur.execute("SELECT sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, created_at FROM iris_history ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception:
        return []

# --- Configuración de la Interfaz ---
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸", layout="wide")

@st.cache_resource
def load_models():
    try:
        # Carga desde la carpeta 'components/' confirmada en tus imágenes
        model = joblib.load('components/iris_model.pkl') [cite: 2, 6]
        scaler = joblib.load('components/iris_scaler.pkl') [cite: 2, 6]
        with open('components/model_info.pkl', 'rb') as f: [cite: 2, 6]
            model_info = pickle.load(f) [cite: 2]
        return model, scaler, model_info
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None, None

st.title("🌸 Predictor de Especies de Iris")

model, scaler, model_info = load_models()

if model is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Nueva Predicción")
        sepal_length = st.number_input("Longitud Sépalo (cm)", 0.0, 10.0, 5.0, step=0.1) [cite: 1]
        sepal_width = st.number_input("Ancho Sépalo (cm)", 0.0, 10.0, 3.0, step=0.1) [cite: 1]
        petal_length = st.number_input("Longitud Pétalo (cm)", 0.0, 10.0, 4.0, step=0.1) [cite: 1]
        petal_width = st.number_input("Ancho Pétalo (cm)", 0.0, 10.0, 1.0, step=0.1) [cite: 1]
        
        if st.button("Predecir e Insertar", use_container_width=True): [cite: 1]
            # Preparar datos y escalar
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]]) [cite: 1]
            features_scaled = scaler.transform(features) [cite: 1]
            
            # Realizar predicción
            prediction_idx = int(model.predict(features_scaled)) [cite: 1]
            probabilities = model.predict_proba(features_scaled) [cite: 1]
            
            # Obtener nombre de la especie
            target_names = model_info.get('target_names', ['Setosa', 'Versicolor', 'Virginica']) [cite: 1, 3]
            predicted_species = target_names[prediction_idx] [cite: 1]
            confidence = float(max(probabilities)) [cite: 1]
            
            # 1. Guardar en la base de datos
            save_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence)
            
            st.success(f"**Especie:** {predicted_species}")
            st.info(f"**Confianza:** {confidence:.1%}")

    with col2:
        st.header("Historial de Consultas")
        history_data = get_history()
        if history_data:
            df = pd.DataFrame(history_data, columns=["Sépalo L", "Sépalo W", "Pétalo L", "Pétalo W", "Especie", "Confianza", "Fecha"])
            # Formatear la visualización
            df['Confianza'] = df['Confianza'].apply(lambda x: f"{x:.1%}")
            st.dataframe(df, use_container_width=True)
        else:
            st.write("Aún no hay datos registrados.")
