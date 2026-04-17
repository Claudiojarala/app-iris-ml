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
    """Establece la conexión con la base de datos de Supabase."""
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

def save_prediction(sl, sw, pl, pw, species, confidence):
    """Guarda los datos de entrada y la predicción en la base de datos."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Crear la tabla si no existe con la estructura de tu historial
        cur.execute("""
            CREATE TABLE IF NOT EXISTS iris_history (
                id SERIAL PRIMARY KEY,
                sl FLOAT,
                sw FLOAT,
                pl FLOAT,
                pw FLOAT,
                especie TEXT,
                confianza FLOAT,
                fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Insertar los valores
        cur.execute("""
            INSERT INTO iris_history (sl, sw, pl, pw, especie, confianza)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (sl, sw, pl, pw, species, confidence))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error al guardar en BD: {e}")

def get_history():
    """Consulta el historial ordenado por fecha de forma descendente."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT sl, sw, pl, pw, especie, confianza, fecha FROM iris_history ORDER BY fecha DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception:
        return []

# --- Configuración de la Página ---
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸", layout="wide")

@st.cache_resource
def load_models():
    """Carga los modelos desde la carpeta 'components'."""
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos en 'components/'")
        return None, None, None

st.title("🌸 Predictor de Especies de Iris")

model, scaler, model_info = load_models()

if model is not None:
    # Usamos columnas para que se parezca a tu diseño
    col_input, col_hist = st.columns([1, 1.5])

    with col_input:
        st.header("Entrada de Datos")
        sl = st.number_input("SL (cm)", 0.0, 10.0, 5.0)
        sw = st.number_input("SW (cm)", 0.0, 10.0, 3.0)
        pl = st.number_input("PL (cm)", 0.0, 10.0, 4.0)
        pw = st.number_input("PW (cm)", 0.0, 10.0, 1.0)
        
        if st.button("Predecir Especie"):
            # Preparar datos
            features = np.array([[sl, sw, pl, pw]])
            features_scaled = scaler.transform(features)
            
            # CORRECCIÓN DEL ERROR: Accedemos al primer elemento antes de int()
            prediction_idx = int(model.predict(features_scaled))
            probabilities = model.predict_proba(features_scaled)
            
            target_names = model_info['target_names']
            predicted_species = target_names[prediction_idx]
            confidence = float(max(probabilities))
            
            # Guardar en base de datos
            save_prediction(sl, sw, pl, pw, predicted_species, confidence)
            
            st.success(f"Especie predicha: **{predicted_species}**")
            st.write(f"Confianza: **{confidence:.2f}**")

    with col_hist:
        st.header("📜 Historial de Predicciones (Descendente)")
        history_data = get_history()
        if history_data:
            # Crear DataFrame con los nombres exactos de tu imagen
            df = pd.DataFrame(history_data, columns=["SL", "SW", "PL", "PW", "Especie", "Confianza", "Fecha"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay registros previos.")
