"""
Flood Prediction — Early Warning System
Flask Web Application
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load model & scaler ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
model      = joblib.load(os.path.join(BASE_DIR, "flood_xgb_model.pkl"))
scaler     = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

FEATURE_ORDER = ['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'YEAR']


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate required fields
        required = ["ANNUAL", "Jan_Feb", "Mar_May", "Jun_Sep", "Oct_Dec", "YEAR"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = np.array([[
            float(data["ANNUAL"]),
            float(data["Jan_Feb"]),
            float(data["Mar_May"]),
            float(data["Jun_Sep"]),
            float(data["Oct_Dec"]),
            float(data["YEAR"]),
        ]])

        prediction  = model.predict(features)[0]
        proba       = model.predict_proba(features)[0]
        confidence  = float(max(proba)) * 100
        flood_prob  = float(proba[1]) * 100

        if prediction == 1:
            label     = "🌊 FLOOD RISK DETECTED"
            risk      = "HIGH" if flood_prob >= 75 else "MODERATE"
            color     = "danger" if flood_prob >= 75 else "warning"
            advice    = ("Immediate action required. Issue evacuation advisories "
                         "and deploy disaster response teams to the region.")
        else:
            label     = "✅ NO FLOOD RISK"
            risk      = "LOW"
            color     = "success"
            advice    = ("No immediate flood threat detected. Continue monitoring "
                         "weather patterns and maintain regular preparedness protocols.")

        return jsonify({
            "prediction"  : int(prediction),
            "label"       : label,
            "risk_level"  : risk,
            "color"       : color,
            "confidence"  : f"{confidence:.2f}%",
            "flood_prob"  : f"{flood_prob:.2f}%",
            "advice"      : advice,
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
