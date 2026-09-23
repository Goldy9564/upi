# UPI Fraud Detection & Risk Scoring

An end-to-end machine learning project for detecting potentially fraudulent UPI transactions.

## Features

- Synthetic UPI transaction dataset generator
- Exploratory-ready CSV
- Feature engineering
- XGBoost classification
- Class-imbalance handling with `scale_pos_weight`
- Accuracy, precision, recall, F1, ROC-AUC and PR-AUC
- Saved Joblib model
- Streamlit real-time risk scoring dashboard

## Important

This repository uses synthetic data for educational purposes. It is not a production banking fraud system.

Do not claim 99% accuracy unless it is actually obtained on an untouched test set. For fraud detection, recall, precision, F1 and PR-AUC are usually more informative than accuracy alone.

## Installation

```bash
pip install -r requirements.txt
```

## 1. Generate dataset

```bash
python generate_dataset.py
```

## 2. Train model

```bash
python train_model.py
```

The following files are created:

```text
models/fraud_model.pkl
models/feature_columns.pkl
reports/metrics.json
```

## 3. Run dashboard

```bash
streamlit run app.py
```

## Suggested interview explanation

The project treats fraud detection as an imbalanced binary classification problem. Transaction and behavioral features are engineered from the raw transaction data. XGBoost is trained with a class-weighting strategy so that the minority fraud class receives more importance. The final application returns a probability-based risk score rather than treating the model as an absolute fraud verdict.

## Resume title

UPI Fraud Detection & Real-Time Transaction Risk Scoring

## Resume bullet

Developed an end-to-end UPI fraud-risk detection system using Python, Pandas, Scikit-learn and XGBoost; engineered transaction velocity, amount-deviation, device, beneficiary and location features, handled class imbalance, evaluated using precision/recall/F1/ROC-AUC/PR-AUC, and deployed a Streamlit dashboard for real-time transaction risk scoring.
