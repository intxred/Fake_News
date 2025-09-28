# ==========================
# Fake News Detection API
# ==========================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

# Step 1: Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Step 2: Preprocessing (same as in training!)
def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Step 3: Flask app
app = Flask(__name__)
CORS(app)

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "API is running"}), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Preprocess
        cleaned_text = preprocess_text(text)
        user_tfidf = vectorizer.transform([cleaned_text])

        # Predict
        user_pred = model.predict(user_tfidf)[0]
        user_confidence = model.predict_proba(user_tfidf)[0].max()

        return jsonify({
            "prediction": user_pred,  # keep exactly as model output (Fake / Real / etc.)
            "confidence": float(user_confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
