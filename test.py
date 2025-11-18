# ==========================================================
# Hypertension Prediction — Fixed Test Script for RFE Pipeline
# ==========================================================
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


# ---------------- Constants ----------------
MODEL_DIR = "ml_models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "xgb_pipeline_rfe_top6_tuned.pkl")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.pkl")
ID_COL = "Patient_Number"
TARGET = "Blood_Pressure_Abnormality"
SYNTHETIC_OUTPUT_PATH = "synthetic_predictions_top6.csv"

# ---------------- 1️⃣ Load Pipeline & Feature Order ----------------
with open(PIPELINE_PATH, "rb") as f:
    pipeline = pickle.load(f)

with open(FEATURE_ORDER_PATH, "rb") as f:
    FEATURE_ORDER = pickle.load(f)

# ---------------- 2️⃣ Load Data ----------------
df_raw = pd.read_csv("data_preprocessed.csv")
synthetic_data = pd.read_csv("synthetic_test_data.csv")

# ---------------- 3️⃣ Preprocessing ----------------
def preprocess(df):
    df_proc = df.copy()
    df_proc['Pregnancy'] = df_proc['Pregnancy'].fillna(0)
    df_proc['Genetic_Pedigree_Coefficient'] = df_proc['Genetic_Pedigree_Coefficient'].fillna(
        df_proc['Genetic_Pedigree_Coefficient'].median())
    df_proc['alcohol_consumption_per_day'] = df_proc['alcohol_consumption_per_day'].fillna(
        df_proc['alcohol_consumption_per_day'].median())
    
    stress_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    df_proc['Level_of_Stress'] = df_proc['Level_of_Stress'].map(stress_mapping)
    df_proc['Level_of_Stress'] = df_proc['Level_of_Stress'].fillna(df_proc['Level_of_Stress'].median())

    Q1, Q3 = df_proc["Level_of_Hemoglobin"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_proc["Level_of_Hemoglobin"] = df_proc["Level_of_Hemoglobin"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    return df_proc

# ---------------- 4️⃣ Evaluate on Test Set ----------------
df_prepped = preprocess(df_raw)
X_full = df_prepped.drop([ID_COL, TARGET], axis=1)
y_full = df_prepped[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.3, random_state=42, stratify=y_full
)

# Ensure correct feature order
X_test = X_test[FEATURE_ORDER]

y_test_pred = pipeline.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Plot and save
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Saves the image
plt.show()
print("\n=== Test Set Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", cm)

# Get predicted probabilities for the positive class
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_curve.png")  # Saves image for report
plt.show()

# ---------------- 5️⃣ Predict on Synthetic Data ----------------
synthetic_prepped = preprocess(synthetic_data.drop([ID_COL], axis=1, errors='ignore'))
synthetic_prepped = synthetic_prepped[FEATURE_ORDER]  # same order as training

y_synth_pred = pipeline.predict(synthetic_prepped)
y_synth_prob = pipeline.predict_proba(synthetic_prepped)[:, 1]

synthetic_data["Predicted_BP_Abnormality"] = y_synth_pred
synthetic_data["Prediction_Probability"] = y_synth_prob

synthetic_data.to_csv(SYNTHETIC_OUTPUT_PATH, index=False)
print(f"\n✅ Predictions saved to '{SYNTHETIC_OUTPUT_PATH}'")
print(synthetic_data.head())

