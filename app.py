from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

#load the trained (or dummy) model
MODEL_PATH = "rf_alzheimer.pkl"
MODEL = joblib.load(MODEL_PATH)
TRAIN_COLS = list(MODEL.feature_names_in_)    

#mapping from UI field names
UI_TO_TRAIN = {
    "Age":    "Age",      
    "Gender": "M/F",        
    "Educ":   "EDUC",
    "MMSE":   "MMSE",
    "SES":    "SES",
    "nWBV":   "nWBV",
    "ASF":    "ASF"
}

#routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()         

        #Rename UI keys to the names used during training
        renamed = {UI_TO_TRAIN[k]: v for k, v in data.items() if k in UI_TO_TRAIN}

        #Create a one-row DataFrame
        df = pd.DataFrame([renamed])

        #Add any missing training columns (set to zero or NaN)
        for col in TRAIN_COLS:
            if col not in df.columns:
                df[col] = 0

        #Re-order columns exactly as the model saw them
        df = df[TRAIN_COLS]

        #Make prediction
        pred = MODEL.predict(df)[0]
        prob = MODEL.predict_proba(df)[0][pred]

        result = {
            "label": "Has Alzheimer’s Disease" if pred == 1 else "Control / No AD",
            "prob":  float(prob)
        }
        return jsonify(result)

    except Exception as e:
        # Log server-side for debugging
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed."}), 500

if __name__ == "__main__":
    app.run(debug=True)
