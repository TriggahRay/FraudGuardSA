# FraudGuard SA

A machine learning tool that helps South African job seekers check whether a job posting is fraudulent before they respond to it. Paste any posting text, click Analyse, and get one of three verdicts: **Legitimate**, **Suspicious** or **Fraudulent**, with a confidence score, probability breakdown, detected fraud indicators, and safety advice.

Developed for **IPJ527C: Advanced Research Project B (Alpha Version)** .

**Author:** Olipelwe Daniel Msuthwana (221040668)  
**Supervisor:** Dr. P. Phoobane, Ph.D.
---

## Functions Implemented


|**ID**|**Requirement**|**Status**|**Evidence**|
|---|---|---|---|
|**R1–R6**|Paste text, three-ter verdict, confdence, probability split, fraud indicators, safety advice|Implemented|Live demo|
|**R7**|Flask REST API returning a JSON classifcaton|Implemented|app.py|
|**R8–R9**|TF-IDF bigram features feeding a Random Forest classifer|Implemented|train_model.py|
|**R10**|Training on the full EMSCAD dataset|Implemented|14,303 postngs|
|**R11**|South African training data|Implemented|138 additonal|
|**R12**|File upload (.txt, .pdf, .docx) as a second input path|Implemented|Live demo|
|**R13**|Comparison against Logistc Regression, Naive Bayes, LSTM|Deferred|Next stage, RO2|
|**R14–R15**|Job seeker survey and user evaluaton|Deferred|Needs ethics clearance|
|**R16**|Deploying the applicaton on PythonAnywhere|Deferred|Next stage<br>4|
|**R17**|POPIA compliance: no personal data processed|Implemented|Public postings only|




## Quick Setup

### 1. Download and Extract

If you downloaded the project as a ZIP file:

1. Right-click `FraudGuardSA.zip`.
2. Select **Extract All**.
3. Open the extracted `FraudGuardSA` folder.

### 2. Install and Run

Requires **Python 3.8+**.

```bash
cd FraudGuardSA
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

> **Note:** If `model.pkl` is not already included, running `train_model.py` will generate it.

---

## Standalone Interface

For a simple demonstration without running the Flask application:

1. Extract the ZIP file.
2. Open the `FraudGuardSA` folder.
3. Double-click `index.html`.
4. Enter a job advertisement and select **Analyse**.

---

# Architecture & Training Data

The backend uses a Scikit-learn machine learning pipeline combining **TF-IDF feature extraction** with a **Random Forest Classifier**.

```text
Job Advertisement
        ↓
Text Processing
        ↓
TF-IDF Vectorization
        ↓
Random Forest Classifier
        ↓
Fraud Probability
        ↓
Risk Classification
        ↓
Indicators + Safety Advice
```

### Model Configuration

- TF-IDF unigrams and bigrams
- Maximum 5,000 features
- English stop-word removal
- Random Forest Classifier
- 200 estimators
- Balanced class weighting

---

# Dataset Composition

### EMSCAD Dataset

The **EMSCAD dataset (Vidros et al., 2017)** contains:

- **17,880 annotated job postings**

It serves as the primary benchmark dataset for the project.

### Supplementary Data

The project includes **138 legitimate job postiongs  from South African job postings platforms(PNet, CareerJunction), and synthetic South African job postings**

All entity details in the datasets contains no personal data, complaint with POPIA.

---

# Empirical Performance

## Model Validation

**80/20 Stratified Split, 3,576 EMSCAD Test Postings**

| Performance Metric | Score |
|---|---:|
| Overall Accuracy | **98.0%** |
| ROC-AUC | **0.992** |
| Fraudulent Precision | **1.000** |
| Fraudulent Recall | **0.590** |

## Three-Tier Evaluation

| Actual Posting | Fraudulent >60% | Suspicious 35–60% | Legitimate <35% |
|---|---:|---:|---:|
| Fraudulent (n=173) | 53.2% | 14.5% | 32.4% |
| Legitimate (n=3,403) | 0.0% | 0.2% | 99.8% |

The system flagged approximately **68% of actual fraudulent postings as Suspicious or higher**, while producing **0% false positives for the Fraudulent classification** during primary testing.

---

# Limitations

**Miss rate on subtle fraud** - Roughly a third of fraudulent postings score below the Suspicious threshold. Inherent to lexical features. 

**Over-flagging unfamiliar sectors** - Legitimate postings from sectors absent from training tend toward Suspicious.

**Text features only** - Company registration, poster history and contact domain are ignored, though research shows they help.

**English only** - The model has not been tested on other South African official languages.


---

# Next Steps

|**1**|Train and compare all four algorithms<br>Random Forest, Logistc Regression, Naive Bayes and LSTM on the full dataset|_Targets the 31.8% miss rate  ·  RO2, R13_|
|---|---|---|
|**2**|**Broaden legitmate SA training coverage**<br>Add postngs across more sectors so unfamiliar industries stop drifing into Suspicious|_Targets the over-fagging defect  ·  R11_|
|**3**|**Collect real South African postngs**<br>Publicly visible postngs gathered manually, POPIA-compliant, replacing synthetc data|_Strengthens local validity  ·  R11_|
|**4**|**Deploy on PythonAnywhere**<br>Deploy the application on PythonAnywhere|_Completes RO1 and RO3  ·  R14, R15_|


---
# 📁 Project Structure

```text
FraudGuardSA/
│
├── app.py
├── train_model.py
├── generate_sa_dataset.py
├── requirements.txt
├── fake_job_postings.csv
├── sa_job_postings.csv
├── model.pkl
├── index.html
└── README.md
```
---
