import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# --- Configuración de Credenciales ---
USER = "postgres.hzzukkgqmvgdbbmpuvgv"
PASSWORD = "Facundo12*12"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# --- Funciones de Base de Datos ---
def get_connection():
    return psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )

def save_to_db(l_s, a_s, l_p, a_p, prediccion):
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Nota: l_p es parte de la PK según tu SQL, no puede ser NULL
        query = """
            INSERT INTO ml.tb_iris (l_s, a_s, l_p, a_p, prediccion)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(query, (l_s, a_s, l_p, a_p, prediccion))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error al guardar: {e}")

def get_history():
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT created_at, l_s, a_s, l_p, a_p, prediccion FROM ml.tb_iris ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        return []

# --- Carga de Modelos ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except Exception:
        return None, None, None

# --- UI Principal ---
st.title("🌸 Predictor de Iris con Historial")

model, scaler, model_info = load_models()

if model:
    # Inputs organizados
    col1, col2 = st.columns(2)
    with col1:
        sepal_l = st.number_input("Longitud Sépalo", value=5.0)
        sepal_w = st.number_input("Ancho Sépalo", value=3.0)
    with col2:
        petal_l = st.number_input("Longitud Pétalo", value=4.0)
        petal_w = st.number_input("Ancho Pétalo", value=1.0)

    if st.button("Predecir y Guardar"):
        # Predicción
        features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        scaled = scaler.transform(features)
        
        # FIX: Aseguramos que el índice sea un entero puro de Python
        prediction_raw = model.predict(scaled)
        prediction_idx = int(prediction_raw) 
        
        target_names = model_info['target_names']
        especie = str(target_names[prediction_idx])
        
        # Mostrar resultado
        st.success(f"Especie: **{especie}**")
        
        # Guardar en Supabase
        save_to_db(sepal_l, sepal_w, petal_l, petal_w, especie)
        st.balloons()

    # --- Mostrar Historial ---
    st.markdown("---")
    st.subheader("📜 Historial de Predicciones (Descendente)")
    datos = get_history()
    if datos:
        st.dataframe(datos, use_container_width=True)
    else:
        st.info("No hay registros previos.")
