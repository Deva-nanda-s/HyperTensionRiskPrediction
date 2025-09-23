# train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

# ---------------- Step 1: Load Data ----------------
df = pd.read_csv("data.csv")  # update with your dataset name

# ---------------- Step 2: Handle Missing Values ----------------
df['Pregnancy'] = df['Pregnancy'].fillna(0)
df['Genetic_Pedigree_Coefficient'] = df['Genetic_Pedigree_Coefficient'].fillna(df['Genetic_Pedigree_Coefficient'].median())
df['alcohol_consumption_per_day'] = df['alcohol_consumption_per_day'].fillna(df['alcohol_consumption_per_day'].median())

# ---------------- Step 3: Outlier Treatment (Hemoglobin example) ----------------
Q1, Q3 = df["Level_of_Hemoglobin"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df["Level_of_Hemoglobin"] = df["Level_of_Hemoglobin"].clip(lower, upper)

# ---------------- Step 4: Encode Categorical ----------------
df = pd.get_dummies(df, columns=["Level_of_Stress"], drop_first=True)

# ---------------- Step 5: Features & Target ----------------
ID_COL, TARGET = "Patient_Number", "Blood_Pressure_Abnormality"
X, y = df.drop([ID_COL, TARGET], axis=1), df[TARGET]

# ---------------- Step 6: Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------- Step 7: Handle Imbalance ----------------
print("Class distribution before SMOTE:", y_train.value_counts())
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", y_train.value_counts())

# ---------------- Step 8: Feature Scaling ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Step 9: Hyperparameter Tuning ----------------
param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5]
}

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("\nBest Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# ---------------- Step 10: Evaluate ----------------
y_pred = best_model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------- Step 11: Save Model & Scaler ----------------
os.makedirs("ml_models", exist_ok=True)

with open("ml_models/xgb_hypertension_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("ml_models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nBest model and scaler saved in 'ml_models/' folder successfully!")
