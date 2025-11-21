import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration ---
MODEL_DIR = "ml_models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "xgb_pipeline_rfe_top6.pkl") 

TRAINING_FEATURES = [
    'Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Age', 'BMI', 'Sex', 
    'Pregnancy', 'Smoking', 'Physical_activity', 'salt_content_in_the_diet', 
    'alcohol_consumption_per_day', 'Level_of_Stress', 'Chronic_kidney_disease', 
    'Adrenal_and_thyroid_disorders'
]

def load_model():
    with open(PIPELINE_PATH, "rb") as f:
        return pickle.load(f)

def get_patient_input():
    print("\nEnter patient details:")
    values = []
    for feature in TRAINING_FEATURES:
        val = input(f"{feature}: ")
        values.append(float(val))
    return values

def predict(patient_data, pipeline):
    df = pd.DataFrame([patient_data], columns=TRAINING_FEATURES)
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0, 1]
    label = "Abnormal" if pred == 1 else "Normal"
    print("\n" + "="*40)
    print(f" Prediction: {label}")
    print(f" Probability of Abnormality: {prob*100:.2f}%")
    print("="*40)

# --- Run Prediction ---
if __name__ == "__main__":
    pipeline = load_model()
    patient_data = get_patient_input()
    predict(patient_data, pipeline)
