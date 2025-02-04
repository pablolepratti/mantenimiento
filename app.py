from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load("modelo_mantenimiento.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    valores = np.array([data["temperatura"], data["vibracion"], data["presion"], data["horas_uso"]]).reshape(1, -1)
    prediccion = modelo.predict(valores)
    resultado = "Falla" if prediccion[0] == 1 else "No Falla"
    return jsonify({"prediccion": resultado})

if __name__ == "__main__":
    app.run(debug=True)
