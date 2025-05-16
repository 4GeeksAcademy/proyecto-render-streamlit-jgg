from flask import Flask, request, render_template
from joblib import load
import os

app = Flask(__name__)

# Lista de nombres de las características en español
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

# Definir rutas absolutas para los modelos
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scaler_path = os.path.join(base_dir, 'models', 'scaler_knn.sav')
modelo_path = os.path.join(base_dir, 'models', 'modelo_final_knn.sav')

# Cargar scaler y modelo
scaler = load(scaler_path)
modelo = load(modelo_path)

@app.route('/')
def index():
    return render_template('index.html', nombres=nombres)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener y convertir características a float
        features = [float(request.form[f'feature{i}']) for i in range(1, 12)]
        # Escalar características
        features_scaled = scaler.transform([features])
        # Predecir con el modelo
        pred = modelo.predict(features_scaled)[0]

        if pred == 0:
            resultado = "Este vino probablemente sea de baja calidad 🍷"
        elif pred == 1:
            resultado = "Este vino probablemente sea de calidad media 🍷"
        else:
            resultado = "Este vino probablemente sea de alta calidad 🍷"
    except Exception as e:
        resultado = f"Error en la predicción: {e}"

    return render_template('index.html', resultado=resultado, nombres=nombres)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
