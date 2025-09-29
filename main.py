from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models
reg_model = joblib.load("water_level_predictor.pkl")
cls_model = joblib.load("flood_classifier.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌊 Flood Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
        "water_level_percent": 45,
        "soil_moisture_percent": 60
    }
    """
    data = request.get_json()

    try:
        water_level = float(data.get("water_level_percent", 0))
        soil_moisture = float(data.get("soil_moisture_percent", 0))

        X = np.array([[water_level, soil_moisture]])

        # Predictions
        predicted_level = reg_model.predict(X)[0]
        flood_pred = int(cls_model.predict(X)[0])

        return jsonify({
            "predicted_water_level": round(predicted_level, 2),
            "flood_alert": flood_pred
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
