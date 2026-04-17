import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuración de las credenciales (Se recomienda usar st.secrets para mayor seguridad)
USER = "postgres.hzzukkgqmvgdbbmpuvgv"
PASSWORD = "Facundo12*12"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# --- FUNCIONES DE BASE DE DATOS ---

def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

def save_prediction(l_s, a_s, l_p, a_p, prediction):
    try:
        conn = get_connection()
        cur = conn.cursor()
        # El esquema es 'ml' y la tabla 'tb_iris'
        query = """
            INSERT INTO ml.tb_iris (l_s, a_s, l_p, a_p, prediccion)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(query, (l_s, a_s, l_p, a_p, prediction))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error al guardar en BD: {e}")

def get_history():
    try:
        conn = get_connection()
        # Usamos RealDictCursor para que sea más fácil de leer en un DataFrame
        cur = conn.cursor(cursor_factory=RealDictCursor)
        query = "SELECT created_at, l_s, a_s, l_p, a_p, prediccion FROM ml.tb_iris ORDER BY created_at DESC"
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Error al obtener historial: {e}")
        return []

# --- CARGA DE MODELOS ---

@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'")
        return None, None, None

# --- INTERFAZ DE USUARIO ---

st.title("🌸 Predictor de Especies de Iris")

model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0)
        sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0)
    with col2:
        petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0)
        petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0)
    
    if st.button("Predecir e Insertar"):
        # 1. Preparar y Predecir
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        prediction_idx = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        target_names = model_info['target_names']
        predicted_species = target_names[prediction_idx]
        
        # 2. Mostrar Resultado
        st.success(f"Especie predicha: **{predicted_species}** (Confianza: {max(probabilities):.1%})")
        
        # 3. Guardar en Base de Datos
        save_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species)
        st.info("Datos guardados en Supabase correctamente.")

    # --- SECCIÓN DE HISTORIAL ---
    st.divider()
    st.header("📜 Historial de Predicciones")
    
    historial = get_history()
    if historial:
        st.table(historial)
    else:
        st.write("Aún no hay registros en la base de datos.")
