
# Train XGBoost Model with RFE (Top Features) — Final Version

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- Setup ----------------
MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_FILE = "data_preprocessed.csv"

if not os.path.exists(DATA_FILE):
    print(f" Error: '{DATA_FILE}' not found.")
    exit()

# ---------------- Load Data ----------------
df = pd.read_csv(DATA_FILE)
ID_COL = "Patient_Number"
TARGET = "Blood_Pressure_Abnormality"
X = df.drop([ID_COL, TARGET], axis=1)
y = df[TARGET]

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------- Compute scale_pos_weight ----------------
pos_count = y_train.value_counts()[1]
neg_count = y_train.value_counts()[0]
scale_pos_weight = neg_count / pos_count if pos_count != 0 else 1.0
print(f"⚖️ scale_pos_weight = {scale_pos_weight:.2f}")

# ---------------- RFE with XGBoost ----------------
xgb_base = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100
)

n_features_to_select = 6
rfe = RFE(estimator=xgb_base, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_].tolist()
print("\n🌟 Top features selected by RFE:")
print(selected_features)

# ---------------- Prepare selected data ----------------
X_train_sel = X_train[selected_features].copy()
X_test_sel = X_test[selected_features].copy()

# ---------------- Scale numeric features ----------------
numeric_cols = X_train_sel.select_dtypes(include=["float64", "int64"]).columns.tolist()
scaler = StandardScaler()
if numeric_cols:
    X_train_sel[numeric_cols] = scaler.fit_transform(X_train_sel[numeric_cols])
    X_test_sel[numeric_cols] = scaler.transform(X_test_sel[numeric_cols])

# ---------------- Final Model ----------------
final_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=300,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2,
    reg_alpha=0.3,
    scale_pos_weight=scale_pos_weight,
    gamma=0.1,  # prevents over-splitting
    min_child_weight=5
)
final_model.fit(X_train_sel, y_train)
print("\n✅ Model training complete!")

# ---------------- Evaluate ----------------
y_train_pred = final_model.predict(X_train_sel)
y_probs = final_model.predict_proba(X_test_sel)[:, 1]
threshold = 0.55  # instead of 0.45
y_test_pred = (y_probs >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
print(f"\nThreshold set to {threshold:.2f}")
print(f"False Negatives: {fn}, False Positives: {fp}")
print(f"\nTraining Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
# ---------------- Save Model Bundle ----------------
bundle = {
    "model": final_model,
    "scaler": scaler,
    "features": selected_features,
    "threshold": threshold
}

bundle_path = os.path.join(MODEL_DIR, "xgb_pipeline_rfe_top6.pkl")
joblib.dump(bundle, bundle_path)
print(f"\n✅ Full pipeline (model + scaler + features) saved at '{bundle_path}'")

# Optional: Save test data for future explanation
joblib.dump({"X_test": X_test_sel, "y_test": y_test}, os.path.join(MODEL_DIR, "test_data.pkl"))

print("\n🎯 Training complete and model ready for deployment!")
