"""
FraudGuard SA - Model Training Script (alpha, final)

Data sources
  1. EMSCAD          fake_job_postings.csv            17,880 postings (Vidros et al., 2017)
  2. Synthetic SA    sa_job_postings.csv              400 postings, template-generated
                                                      from documented SA scam patterns
  3. Additionaly SA data         sa_job_postings_with_descriptions.csv
                                                      138 legitimate postings collected
                                                      from CareerJunction and
                                                      PNet, each traceable to a live job ID

Evaluation design
  - EMSCAD held-out test split (20%, stratified) for overall metrics.
  - 61 highest-fidelity real SA postings (full job-page text and PNet results text)
    are excluded from training and used as an independent real-world validation set.
  - The deployment model is then refitted on everything.

Pipeline: TF-IDF (unigrams + bigrams, 5,000 features, English stop words)
          -> Random Forest (200 estimators, balanced class weights)
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
HIGH_FIDELITY = ["job page (full text)", "results page (PNet)",
                 "job page (full text, mirrored from twin posting)"]


def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                  stop_words="english")),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=42, n_jobs=-1)),
    ])


# ── 1. EMSCAD ───────────────────────────────────────────────────
df = pd.read_csv("fake_job_postings.csv")
df[TEXT_COLS] = df[TEXT_COLS].fillna("")
df["text"] = df[TEXT_COLS].agg(" ".join, axis=1).str.strip()
df = df[df["text"].str.len() >= 20]
Etr_X, Ete_X, Etr_y, Ete_y = train_test_split(
    df["text"].tolist(), df["fraudulent"].tolist(),
    test_size=0.2, random_state=42, stratify=df["fraudulent"].tolist())

# ── 2. Synthetic SA ─────────────────────────────────────────────
syn = pd.read_csv("sa_job_postings.csv")

# ── 3.Additional SA sata ──────────────────────────────────────────────────
real = pd.read_csv("sa_job_postings_with_descriptions.csv")
real = real[real["description"].notna()
            & (real["description"].astype(str).str.len() > 80)].copy()
real["text"] = (real["title"].astype(str) + " " + real["company"].astype(str) + " "
                + real["location"].astype(str) + " " + real["description"].astype(str))
hold = real[real["description_source"].isin(HIGH_FIDELITY)]
train_real = real[~real["description_source"].isin(HIGH_FIDELITY)]

print(f"EMSCAD train {len(Etr_X)} / test {len(Ete_X)}")
print(f"Synthetic SA {len(syn)}  |  Additinal SA data usable {len(real)} "
      f"(validation hold-out {len(hold)}, trainable {len(train_real)})\n")

# ── 4. Evaluation model ─────────────────────────────────────────
model = make_pipeline()
model.fit(Etr_X + syn["text"].tolist() + train_real["text"].tolist(),
          Etr_y + syn["fraudulent"].tolist() + [0] * len(train_real))

prob = model.predict_proba(Ete_X)[:, 1]
y = np.array(Ete_y)
print(f"EMSCAD accuracy {accuracy_score(y, (prob >= .5).astype(int)):.4f} | "
      f"ROC-AUC {roc_auc_score(y, prob):.4f}\n")
print(classification_report(y, model.predict(Ete_X),
      target_names=["Legitimate", "Fraudulent"], digits=3))

f, l = prob[y == 1], prob[y == 0]
print("Three-tier evaluation on EMSCAD test (0.35 / 0.60 thresholds):")
print(f"  Fraudulent ({len(f)}): Fraudulent {(f > .60).mean()*100:.1f}% | "
      f"Suspicious {((f > .35) & (f <= .60)).mean()*100:.1f}% | missed {(f <= .35).mean()*100:.1f}%")
print(f"  Legitimate ({len(l)}): correct {(l <= .35).mean()*100:.1f}% | "
      f"Suspicious {((l > .35) & (l <= .60)).mean()*100:.1f}% | "
      f"wrongly Fraudulent {(l > .60).mean()*100:.1f}%")

hp = model.predict_proba(hold["text"].tolist())[:, 1]
n = len(hp)
print(f"\nAdditional SA data validation set ({n} postings, never seen in training):")
print(f"  Legitimate {(hp <= .35).sum()/n*100:.1f}% | "
      f"Suspicious {((hp > .35) & (hp <= .60)).sum()/n*100:.1f}% | "
      f"Fraudulent {(hp > .60).sum()/n*100:.1f}% | mean fraud probability {hp.mean()*100:.1f}%")

# ── 5. Deployment model, fitted on everything ───────────────────
final = make_pipeline()
final.fit(Etr_X + syn["text"].tolist() + real["text"].tolist(),
          Etr_y + syn["fraudulent"].tolist() + [0] * len(real))
with open("model.pkl", "wb") as fh:
    pickle.dump(final, fh)
print(f"\nDeployment model trained on {len(Etr_X) + len(syn) + len(real)} samples. Saved model.pkl")
