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

# --- Funciones de Base de Datos ---
def get_connection():
    return psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )

def save_to_db(l_s, a_s, l_p, a_p, prediccion):
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Se especifica el esquema 'ml' según tu definición de tabla
        query = """
            INSERT INTO ml.tb_iris (l_s, a_s, l_p, a_p, prediccion)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(query, (l_s, a_s, l_p, a_p, prediccion))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error al insertar en la base de datos: {e}")

def get_history():
    try:
        conn = get_connection()
        # RealDictCursor devuelve los resultados como diccionarios (clave: valor)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Consulta con orden descendente por fecha de creación
        cur.execute("SELECT created_at, l_s, a_s, l_p, a_p, prediccion FROM ml.tb_iris ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Error al obtener historial: {e}")
        return []

# --- Lógica de la Aplicación ---
st.title("🌸 Predictor de Iris con Persistencia")

# (Asumiendo que ya tienes cargados model, scaler y model_info)
# ... código de carga de modelos ...

if model:
    st.header("Ingresa las características:")
    sepal_l = st.number_input("Longitud Sépalo", value=5.0)
    sepal_w = st.number_input("Ancho Sépalo", value=3.0)
    petal_l = st.number_input("Longitud Pétalo", value=4.0)
    petal_w = st.number_input("Ancho Pétalo", value=1.0)

    if st.button("Predecir y Guardar"):
        # 1. Preparar datos
        features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        scaled = scaler.transform(features)
        
        # 2. Predicción (CORRECCIÓN DEL ERROR)
        prediction_raw = model.predict(scaled)
        # Usamos .item() para obtener el valor escalar de forma segura, sea un array o lista
        prediction_idx = int(prediction_raw) 
        
        target_names = model_info['target_names']
        especie_predicha = str(target_names[prediction_idx])
        
        # 3. Guardar en Supabase
        save_to_db(sepal_l, sepal_w, petal_l, petal_w, especie_predicha)
        
        st.success(f"Especie detectada: **{especie_predicha}**")
        st.info("Resultado guardado en la base de datos.")

    # --- Sección de Historial ---
    st.divider()
    st.subheader("📜 Historial de Predicciones (Recientes primero)")
    historial = get_history()
    
    if historial:
        # Se muestra como una tabla interactiva
        st.dataframe(historial, use_container_width=True)
    else:
        st.write("No hay registros disponibles.")
