from flask import Flask, request, jsonify
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features]
        pred = model.predict(df)[0]
        pred_proba = model.predict_proba(df)[0][1]
        return jsonify({"ChurnPrediction": int(pred), "ChurnProbability": float(pred_proba)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)