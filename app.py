from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Intentar cargar el modelo con manejo de errores
try:
    modelo = joblib.load("modelo_mantenimiento.pkl")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

# Ruta de prueba para verificar si la API está en línea
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API en funcionamiento"})

# Ruta para realizar predicciones
@app.route("/predict", methods=["POST"])
def predict():
    if modelo is None:
        return jsonify({"error": "Modelo no cargado"}), 500
    
    try:
        data = request.get_json()
        valores = np.array([data["temperatura"], data["vibracion"], data["presion"], data["horas_uso"]]).reshape(1, -1)
        prediccion = modelo.predict(valores)
        resultado = "Falla" if prediccion[0] == 1 else "No Falla"
        return jsonify({"prediccion": resultado})
    except Exception as e:
        return jsonify({"error": f"Error en la predicción: {str(e)}"}), 400

# Iniciar la aplicación en el puerto dinámico de Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render asigna automáticamente un puerto
    app.run(host="0.0.0.0", port=port)

