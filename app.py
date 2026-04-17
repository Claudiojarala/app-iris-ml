import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd

# --- Configuración de Base de Datos (Supabase) ---
# He mantenido tus credenciales según los archivos proporcionados [cite: 1]
USER = "postgres.hzzukkgqmvgdbbmpuvgv"
PASSWORD = "Facundo12*12"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

def get_connection():
    """Establece la conexión con PostgreSQL[cite: 1]."""
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

def save_prediction(sepal_l, sepal_w, petal_l, petal_w, prediction, confidence):
    """Inserta los datos de entrada y el resultado en la tabla[cite: 1]."""
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
    """Obtiene los registros ordenados por fecha descendente[cite: 1]."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT sepal_length, sepal_width, petal_length, petal_width, prediction, confidence, created_at 
            FROM iris_history 
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception:
        return []

# --- Configuración de Streamlit ---
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸", layout="wide")

@st.cache_resource
def load_models():
    """Carga el modelo, escalador e información desde la carpeta 'components'[cite: 1, 1102, 1103]."""
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except Exception as e:
        st.error(f"No se pudieron cargar los modelos: {e}")
        return None, None, None

st.title("🌸 Predictor de Especies de Iris")

model, scaler, model_info = load_models()

if model is not None:
    # Dividimos la pantalla en dos columnas para ver el formulario y el historial al lado
    col1, col2 = st.columns(2)

    with col1:
        st.header("Entrada de Datos")
        # Inputs para las características de la flor [cite: 1]
        sepal_length = st.number_input("Longitud del Sépalo (cm)", 0.0, 10.0, 5.0)
        sepal_width = st.number_input("Ancho del Sépalo (cm)", 0.0, 10.0, 3.0)
        petal_length = st.number_input("Longitud del Pétalo (cm)", 0.0, 10.0, 4.0)
        petal_width = st.number_input("Ancho del Pétalo (cm)", 0.0, 10.0, 1.0)
        
        if st.button("Predecir y Guardar", use_container_width=True):
            # Preparación de datos y escalado [cite: 1]
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            features_scaled = scaler.transform(features)
            
            # Predicción y cálculo de confianza [cite: 1]
            prediction_idx = int(model.predict(features_scaled))
            probabilities = model.predict_proba(features_scaled)
            confidence = float(max(probabilities))
            
            # Obtener nombre de la especie desde model_info [cite: 1103]
            target_names = model_info.get('target_names', ['setosa', 'versicolor', 'virginica'])
            predicted_species = target_names[prediction_idx]
            
            # Guardar en la base de datos de Supabase
            save_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence)
            
            st.success(f"Especie predicha: **{predicted_species}**")
            st.write(f"Confianza: **{confidence:.1%}**")

    with col2:
        st.header("Historial (Más recientes primero)")
        history = get_history()
        if history:
            # Convertimos a DataFrame para una mejor visualización 
            df = pd.DataFrame(history, columns=[
                "Sepal L", "Sepal W", "Petal L", "Petal W", "Resultado", "Confianza", "Fecha"
            ])
            # Formatear la confianza como porcentaje
            df['Confianza'] = df['Confianza'].apply(lambda x: f"{x:.1%}")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Aún no hay registros en la base de datos.")
