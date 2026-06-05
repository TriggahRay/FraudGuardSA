"""
FraudGuard SA - Fraudulent Job Posting Detection System
Flask backend with Random Forest classifier (TF-IDF features)
"""
import pickle as pk
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    pipeline = pk.load(f)

def classify_posting(text):
    proba = pipeline.predict_proba([text])[0]
    fraud_prob = proba[1]
    legit_prob = proba[0]

    if fraud_prob >= 0.60:
        label = "Fraudulent"
        risk = "HIGH RISK"
        colour = "red"
        confidence = round(fraud_prob * 100, 1)
        advice = "This posting shows strong indicators of fraud. Do not submit personal documents, banking details, or pay any fees. Report this posting to the job platform immediately."
    elif fraud_prob >= 0.35:
        label = "Suspicious"
        risk = "MEDIUM RISK"
        colour = "orange"
        confidence = round(fraud_prob * 100, 1)
        advice = "This posting contains some characteristics associated with fraudulent ads. Proceed with caution and verify the company independently before submitting any personal information."
    else:
        label = "Legitimate"
        risk = "LOW RISK"
        colour = "green"
        confidence = round(legit_prob * 100, 1)
        advice = "This posting appears legitimate based on its content. Always verify the company independently and never share banking details or pay upfront fees."

    fraud_keywords = [
        "send id", "bank details", "banking details", "registration fee",
        "no experience needed", "work from home", "guaranteed income",
        "urgent hiring", "send passport", "deposit required", "whatsapp",
        "weekly pay", "daily earnings", "limited slots", "upfront fee",
        "joining fee", "no qualifications needed", "earn r"
    ]
    lower_text = text.lower()
    found = [kw for kw in fraud_keywords if kw in lower_text]

    return {
        "label": label,
        "risk": risk,
        "colour": colour,
        "confidence": confidence,
        "fraud_prob": round(fraud_prob * 100, 1),
        "legit_prob": round(legit_prob * 100, 1),
        "indicators": found[:5],
        "advice": advice
    }

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyse", methods=["POST"])
def analyse():
    text = request.form.get("posting", "").strip()
    if not text or len(text) < 20:
        return render_template("index.html", error="Please paste a job posting of at least 20 characters.")
    result = classify_posting(text)
    snippet = text[:300] + ("..." if len(text) > 300 else "")
    return render_template("result.html", result=result, posting=snippet)

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(classify_posting(text))

if __name__ == "__main__":
    app.run(debug=True, port=5000)





