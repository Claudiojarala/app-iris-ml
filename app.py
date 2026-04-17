import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd

# Variables Supabase
USER = "postgres.hzzukkgqmvgdbbmpuvgv"
PASSWORD = "Facundo12*12"
HOST = "aws-1-us-east-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# Configuración página
st.set_page_config(
    page_title="Predictor de Iris",
    page_icon="🌸",
    layout="wide"
)

# Cargar modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)

        return model, scaler, model_info

    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo")
        return None, None, None


# Guardar predicción
def guardar_prediccion(sl, sw, pl, pw, especie):

    try:
        conn = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )

        cursor = conn.cursor()

        query = """
        INSERT INTO ml.tb_iris 
        (l_p, l_s, a_p, a_s, prediccion)
        VALUES (%s, %s, %s, %s, %s)
        """

        cursor.execute(query, (
            pl,
            sl,
            pw,
            sw,
            especie
        ))

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        st.error(f"Error al guardar: {e}")


# Obtener historial
def obtener_historial():

    try:
        conn = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )

        query = """
        SELECT 
        l_s as "SL",
        a_s as "SW",
        l_p as "PL",
        a_p as "PW",
        prediccion as "Especie",
        created_at as "Fecha"
        FROM ml.tb_iris
        ORDER BY created_at DESC
        """

        df = pd.read_sql(query, conn)

        conn.close()

        return df

    except Exception as e:
        st.error(f"Error al obtener historial: {e}")
        return None


# Borrar historial
def borrar_historial():

    try:
        conn = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )

        cursor = conn.cursor()

        query = "DELETE FROM ml.tb_iris"

        cursor.execute(query)

        conn.commit()
        cursor.close()
        conn.close()

        st.success("Historial borrado correctamente")

    except Exception as e:
        st.error(f"Error al borrar: {e}")


# Titulo
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelo
model, scaler, model_info = load_models()

if model is not None:

    st.subheader("Ingresa las características de la flor")

    col1, col2 = st.columns(2)

    with col1:

        sepal_length = st.number_input(
            "Longitud del Sépalo (cm)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

        sepal_width = st.number_input(
            "Ancho del Sépalo (cm)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1
        )

    with col2:

        petal_length = st.number_input(
            "Longitud del Pétalo (cm)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1
        )

        petal_width = st.number_input(
            "Ancho del Pétalo (cm)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )


    # Botón predicción
    if st.button("Predecir Especie", use_container_width=True):

        features = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        probabilities = model.predict_proba(features_scaled)[0]

        target_names = model_info['target_names']

        predicted_species = target_names[prediction]

        confianza = float(max(probabilities))

        st.success(f"Especie predicha: {predicted_species}")
        st.info(f"Confianza: {confianza:.2f}")

        guardar_prediccion(
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            predicted_species
        )

        st.subheader("Probabilidades")

        for species, prob in zip(target_names, probabilities):
            st.write(f"{species}: {prob:.2f}")


# Historial
st.markdown("---")

st.subheader("📜 Historial de Predicciones")

historial = obtener_historial()

if historial is not None:

    col1, col2 = st.columns([3,1])

    with col1:

        especies = ["Todas"] + list(historial["Especie"].unique())

        filtro = st.selectbox(
            "Filtrar por especie",
            especies
        )

    with col2:

        if st.button("🗑️ Borrar Historial"):
            borrar_historial()
            st.rerun()


    # Aplicar filtro
    if filtro != "Todas":
        historial = historial[historial["Especie"] == filtro]

    st.dataframe(
        historial,
        use_container_width=True,
        hide_index=True
    )
