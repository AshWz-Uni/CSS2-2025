from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ----- load trained Random-Forest -----
MODEL = joblib.load("rf_alzheimer.pkl")
FEATURES = ["Age","Gender","Educ","MMSE","SES","nWBV","ASF"]

# ----- routes -----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Ensure correct order
    df = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    pred = MODEL.predict(df)[0]
    prob = MODEL.predict_proba(df)[0][pred]
    return jsonify({
        "label": "Alzheimer’s" if pred==1 else "Control / No AD",
        "prob":  float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)
