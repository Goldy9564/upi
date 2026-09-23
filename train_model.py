import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

DATA_PATH = "data/upi_transactions.csv"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET = "is_fraud"
DROP_COLS = ["transaction_id", TARGET]
features = [c for c in df.columns if c not in DROP_COLS]

X = df[features].copy()
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Handle imbalance without changing the test distribution.
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / max(pos, 1)

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
threshold = 0.50
pred = (proba >= threshold).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "pr_auc": float(average_precision_score(y_test, proba)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist()
}

print("\n=== TEST RESULTS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\nClassification report:")
print(classification_report(y_test, pred, digits=4, zero_division=0))

joblib.dump(model, MODEL_DIR / "fraud_model.pkl")
joblib.dump(features, MODEL_DIR / "feature_columns.pkl")

with open(REPORT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(MODEL_DIR / "fraud_model.pkl")
print(MODEL_DIR / "feature_columns.pkl")
print(REPORT_DIR / "metrics.json")
