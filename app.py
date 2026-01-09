from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle

# Load model, scaler, threshold
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
threshold = pickle.load(open("threshold.pkl", "rb"))

app = FastAPI()

@app.post("/predict")
def predict_customer(data: dict):

    df = pd.DataFrame([data])

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])

    proba = model.predict_proba(df)[0][1]

    prediction = 1 if proba >= threshold else 0

    return {
        "churn_probability": float(proba),
        "prediction": int(prediction)
    }