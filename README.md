# FraudGuard SA 🛡

**A machine learning system for detecting fraudulent job postings on South African online platforms.**

Developed as part of IPJ517C: Advanced Research Project  
Central University of Technology, Free State  
Student: Olipelwe Daniel Msuthwana (221040668)  
Supervisor: Dr P. Phoobane Ph.D.

---

## Overview

FraudGuard SA is a browser-based fraud detection prototype that analyses the text of any job posting and returns a fraud risk classification — Legitimate, Suspicious, or Fraudulent — in under one second.

The system uses a Random Forest classifier trained on TF-IDF bigram features extracted from the [EMSCAD dataset](https://emscad.samos.aegean.gr/) (Vidros et al., 2017), supplemented with South African fraud patterns.

---

## Screenshots

| Home Page | HIGH RISK Result |
|-----------|-----------------|
| ![Home](screenshots/screenshot_home.png) | ![Fraudulent](screenshots/screenshot_fraudulent.png) |

| LOW RISK Result | MEDIUM RISK Result |
|----------------|-------------------|
| ![Legitimate](screenshots/screenshot_legitimate.png) | ![Suspicious](screenshots/screenshot_suspicious.png) |

---

## Classification Tiers

| Verdict | Fraud Probability | Risk Level |
|---------|------------------|------------|
| Legitimate | Below 35% | LOW RISK |
| Suspicious | 35% – 60% | MEDIUM RISK |
| Fraudulent | Above 60% | HIGH RISK |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 92.86% |
| Fraudulent Precision | 1.000 |
| Legitimate Recall | 1.000 |
| Fraudulent F1-Score | 0.923 |

---

## Tech Stack

- **Backend:** Python, Flask
- **ML Pipeline:** Scikit-learn (TF-IDF + Random Forest)
- **Frontend:** HTML5, CSS3
- **Dataset:** EMSCAD (Vidros et al., 2017) + SA supplement
- **Architecture:** Client-server REST API

---

## Option 1 — Run the Standalone Interface (No Flask needed)

The simplest way to use FraudGuard SA:

1. Download `fraudguard_ui.html`
2. Open it in any web browser (Chrome recommended)
3. Paste a job posting and click **Analyse**

No Python, no installation, no server required.

---

## Option 2 — Run the Full Flask Application

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/FraudGuardSA.git
cd FraudGuardSA

# Install dependencies
pip install -r requirements.txt
```

### Train the model

```bash
python train_model.py
```

This creates `model.pkl` in the project directory.

### Run the application

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## API Usage

The system exposes a JSON REST API endpoint:

```bash
POST /api/analyse
Content-Type: application/json

{
  "text": "Paste job posting text here"
}
```

**Response:**
```json
{
  "label": "Fraudulent",
  "risk": "HIGH RISK",
  "confidence": 97.0,
  "fraud_prob": 97.0,
  "legit_prob": 3.0,
  "indicators": ["bank details", "registration fee", "no experience needed"],
  "advice": "This posting shows strong indicators of fraud..."
}
```

---

## Project Structure

```
FraudGuardSA/
├── app.py                 # Flask backend application
├── train_model.py         # Model training script
├── model.pkl              # Trained Random Forest pipeline
├── fraudguard_ui.html     # Standalone browser interface
├── requirements.txt       # Python dependencies
├── screenshots/           # Prototype screenshots
│   ├── screenshot_home.png
│   ├── screenshot_fraudulent.png
│   ├── screenshot_legitimate.png
│   └── screenshot_suspicious.png
└── README.md
```

---

## Dataset

This prototype uses training data derived from the **Employment Scam Aegean Corpus (EMSCAD)**:

> Vidros, S., Kolias, C., Kambourakis, G. and Akoglu, L. (2017) *Automatic detection of online recruitment frauds: characteristics, methods, and a public dataset*. Future Internet, 9(1), p. 6.

The dataset was supplemented with manually constructed South African fraud patterns. All supplementary samples comply with the **Protection of Personal Information Act (POPIA, Act 4 of 2013)** and contain no real personal information.

---

## Research Context

South Africa's unemployment rate of 32.9% (Statistics South Africa, 2024) creates a large pool of vulnerable job seekers. Existing fraud detection models are trained on international datasets and do not reflect South African recruitment patterns, local scam typologies, or platform behaviour.

FraudGuard SA addresses this gap by providing a locally contextualised detection tool built specifically for the South African online recruitment environment.

---

## Academic Integrity

This project was developed for academic research purposes. All sources are cited using Harvard referencing. The completed research will be submitted through SafeAssign on the CUT eThuto platform.

---

## License

This project is for academic and research purposes only.
