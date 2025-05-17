import streamlit as st
from joblib import load
import os

# Títulos y texto
st.title("Predicción de Calidad del Vino 🍷")

# Lista de características en español
nombres = [
    "Acidez fija",
    "Acidez volátil",
    "Ácido cítrico",
    "Azúcar residual",
    "Cloruros",
    "Dióxido de azufre libre",
    "Dióxido de azufre total",
    "Densidad",
    "pH",
    "Sulfatos",
    "Alcohol"
]

# Ajustamos base_dir solo un nivel arriba, porque models está dentro de src
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scaler_path = os.path.join(base_dir, 'models', 'scaler_knn.sav')
modelo_path = os.path.join(base_dir, 'models', 'modelo_final_knn.sav')


# DEBUG
print("Base directory:", base_dir)
print("Ruta scaler:", scaler_path)
print("Ruta modelo:", modelo_path)
print("¿Existe scaler_knn.sav?:", os.path.isfile(scaler_path))
print("¿Existe modelo_final_knn.sav?:", os.path.isfile(modelo_path))

# Cargar scaler y modelo
scaler = load(scaler_path)
modelo = load(modelo_path)

# Crear inputs para las características
caracteristicas = []
for nombre in nombres:
    valor = st.number_input(nombre, format="%.4f")
    caracteristicas.append(valor)

if st.button('Predecir'):
    try:
        caracteristicas_scaled = scaler.transform([caracteristicas])
        pred = modelo.predict(caracteristicas_scaled)[0]

        if pred == 0:
            st.success("Este vino probablemente sea de baja calidad 🍷")
        elif pred == 1:
            st.info("Este vino probablemente sea de calidad media 🍷")
        else:
            st.success("Este vino probablemente sea de alta calidad 🍷")

    except Exception as e:
        st.error(f"Error en la predicción: {e}")
