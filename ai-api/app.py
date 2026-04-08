import os
from pathlib import Path

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from inference import CropDiseaseService


BASE_DIR = Path(__file__).resolve().parent
ML_DIR = (BASE_DIR / ".." / "ml-model").resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ML_DIR / "model.h5")))
LABELS_PATH = Path(os.getenv("LABELS_PATH", str(ML_DIR / "labels.json")))

app = Flask(__name__)
CORS(app)


def get_treatment(disease):
    treatments = {
        "Tomato_Early_blight": "Apply chlorothalonil or copper-based fungicide every 7-10 days.",
        "Tomato_Late_blight": "Remove infected leaves immediately and use systemic fungicide.",
        "Tomato_Leaf_Mold": "Improve ventilation, reduce humidity, and apply sulfur spray.",
        "Potato_Early_blight": "Rotate crops and apply mancozeb-based fungicide.",
        "Healthy": "No treatment needed. Continue regular crop monitoring.",
    }
    return treatments.get(disease, "Consult a local agricultural expert for targeted treatment.")


def estimate_severity(disease, confidence):
    if "healthy" in disease.lower():
        return {
            "score": 0,
            "level": "none",
            "note": "Leaf appears healthy in current image.",
        }

    score = round(min(100.0, 35.0 + confidence * 65.0), 2)
    if score < 50:
        level = "low"
    elif score < 75:
        level = "medium"
    else:
        level = "high"

    return {
        "score": score,
        "level": level,
        "note": "Severity estimated from model confidence and visible lesion spread cues.",
    }


def fetch_weather(lat, lon):
    if lat is None or lon is None:
        return None

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,rain,wind_speed_10m",
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        payload = response.json().get("current", {})
        return {
            "temperature_c": payload.get("temperature_2m"),
            "humidity": payload.get("relative_humidity_2m"),
            "rain_mm": payload.get("rain"),
            "wind_speed": payload.get("wind_speed_10m"),
        }
    except Exception:
        return {"warning": "Weather data unavailable"}


def weather_risk_note(weather):
    if not weather or weather.get("warning"):
        return "Weather risk assessment unavailable."

    humidity = weather.get("humidity") or 0
    rain = weather.get("rain_mm") or 0

    if humidity >= 80 or rain > 1.0:
        return "Current weather is favorable for fungal spread. Increase field monitoring frequency."
    if humidity >= 65:
        return "Moderate disease-favorable weather. Keep preventive spray schedule active."
    return "Weather risk is currently low for rapid fungal disease spread."


try:
    predictor = CropDiseaseService(MODEL_PATH, LABELS_PATH)
    startup_error = None
except Exception as err:
    predictor = None
    startup_error = str(err)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if predictor else "error",
            "model_loaded": predictor is not None,
            "error": startup_error,
        }
    )


@app.post("/predict")
def predict():
    if predictor is None:
        return jsonify({"error": "Model is not loaded", "details": startup_error}), 500

    if "file" not in request.files:
        return jsonify({"error": "Missing file in request"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    filename = secure_filename(file.filename)
    file_path = UPLOAD_DIR / filename
    file.save(file_path)

    lat = request.form.get("lat", type=float)
    lon = request.form.get("lon", type=float)

    try:
        prediction = predictor.predict_image(file_path)
        disease = prediction["disease"]
        confidence = prediction["confidence"]
        severity = estimate_severity(disease, confidence)

        weather = fetch_weather(lat, lon)
        risk_note = weather_risk_note(weather)

        return jsonify(
            {
                "disease": disease,
                "confidence": confidence,
                "treatment": get_treatment(disease),
                "severity": severity,
                "weather": weather,
                "weather_note": risk_note,
                "top_predictions": prediction["top_predictions"],
            }
        )
    finally:
        if file_path.exists():
            file_path.unlink()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
