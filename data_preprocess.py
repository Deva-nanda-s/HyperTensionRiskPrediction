
# preprocess_data.py — Clean and prepare hypertension dataset
import pandas as pd
import numpy as np

DATA_PATH = "data.csv"
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------- Clean column names ----------------
df.columns = df.columns.str.strip().str.replace(" ", "_")

# ---------------- Drop duplicates ----------------
df = df.drop_duplicates()

# ---------------- Fix numeric scale ----------------
def fix_scale(series):
    if series.max() > 1000:
        print(f"⚙️ Fixing scale for {series.name}")
        return series / 1000
    return series

df["salt_content_in_the_diet"] = fix_scale(df["salt_content_in_the_diet"])
df["alcohol_consumption_per_day"] = fix_scale(df["alcohol_consumption_per_day"])

# ---------------- Handle missing values ----------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
        print(f"🔹 Filled missing values in '{col}' with median.")

# ---------------- Encode categorical features ----------------
if df["Sex"].dtype == "object":
    df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})
if df["Level_of_Stress"].dtype == "object":
    df["Level_of_Stress"] = df["Level_of_Stress"].map({"Low": 1, "Medium": 2, "High": 3}).fillna(2)

# ---------------- Pregnancy-Sex consistency ----------------
df.loc[df["Sex"] == 0, "Pregnancy"] = 0

# ---------------- Handle outliers ----------------
def cap_outliers(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return np.clip(series, lower, upper)

cols_to_cap = ["Level_of_Hemoglobin", "BMI", "salt_content_in_the_diet", "alcohol_consumption_per_day", "Age"]
for col in cols_to_cap:
    df[col] = cap_outliers(df[col])

# ---------------- Remove invalid values ----------------
df = df[df["Age"] > 0]
df = df[df["BMI"] > 0]

# ---------------- Target check ----------------
target = "Blood_Pressure_Abnormality"
balance = df[target].value_counts()
print("\n📊 Target Balance:")
print(balance)
ratio = balance[1] / balance[0]
print(f"➡️ Ratio (1:0) = {ratio:.2f}")

# ---------------- Save cleaned data ----------------
CLEAN_PATH = "data_preprocessed.csv"
df.to_csv(CLEAN_PATH, index=False)
print(f"\n✅ Cleaned dataset saved to '{CLEAN_PATH}'")
print("✅ Final dataset shape:", df.shape)
