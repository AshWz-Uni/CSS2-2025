from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ─── load the trained (or dummy) model ────────────────────────────────────────
MODEL_PATH = "rf_alzheimer.pkl"
MODEL = joblib.load(MODEL_PATH)

# The feature names the model expects (recorded at fit time)
TRAIN_COLS = list(MODEL.feature_names_in_)     # scikit-learn ≥1.2

# ─── mapping from UI field names → training column names  ─────────────────────
## NEW: adapt this once; leave UI labels unchanged
UI_TO_TRAIN = {
    "Age":    "Age",        # if Age existed when training
    "Gender": "M/F",        # 0 = Female, 1 = Male  (encoded same way)
    "Educ":   "EDUC",
    "MMSE":   "MMSE",
    "SES":    "SES",
    "nWBV":   "nWBV",
    "ASF":    "ASF"
}

# ─── routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()           # JSON dict from front-end

        # 1) Rename UI keys to the names used during training
        renamed = {UI_TO_TRAIN[k]: v for k, v in data.items() if k in UI_TO_TRAIN}

        # 2) Create a one-row DataFrame
        df = pd.DataFrame([renamed])

        # 3) Add any missing training columns (set to zero or NaN)
        for col in TRAIN_COLS:
            if col not in df.columns:
                df[col] = 0

        # 4) Re-order columns exactly as the model saw them
        df = df[TRAIN_COLS]

        # 5) Make prediction
        pred = MODEL.predict(df)[0]
        prob = MODEL.predict_proba(df)[0][pred]

        result = {
            "label": "Alzheimer’s" if pred == 1 else "Control / No AD",
            "prob":  float(prob)
        }
        return jsonify(result)

    except Exception as e:
        # Log server-side for debugging, but return generic message to client
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed."}), 500

# ─── entry-point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
