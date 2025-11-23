import os
import joblib
import pandas as pd
from flask import Flask, request, render_template
from xgboost import XGBClassifier, Booster

app = Flask(__name__)

# ---------------- Paths ----------------
MODEL_DIR = "ml_models"
MODEL_JSON = os.path.join(MODEL_DIR, "xgb_booster.json")
PIPELINE_META = os.path.join(MODEL_DIR, "xgb_pipeline_meta.pkl")

# ---------------- Load pipeline ----------------
try:
    booster = Booster()
    booster.load_model(MODEL_JSON)
    pipeline_bundle = joblib.load(PIPELINE_META)
    scaler = pipeline_bundle["scaler"]
    features = pipeline_bundle["features"]
    threshold = pipeline_bundle.get("threshold", 0.55)
    print("✅ Model and pipeline loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    booster, scaler, features, threshold = None, None, [], 0.55

# Helper: wrap Booster for sklearn-like predict
class XGBWrapper:
    def __init__(self, booster):
        self.booster = booster
    def predict_proba(self, X):
        import numpy as np
        dmatrix = XGBClassifier()._dmatrix(X)
        preds = self.booster.predict(dmatrix)
        return np.vstack([1 - preds, preds]).T

xgb_model = XGBWrapper(booster) if booster else None
# --------------------------
# HELPER FUNCTIONS
# --------------------------

def apply_health_override(data, prob, threshold):
    is_clearly_healthy = (
        data["Age"] < 30
        and 18.5 <= data["BMI"] <= 25
        and data["Level_of_Stress"] == 1.0
        and data["Smoking"] == 0
        and data["Chronic_kidney_disease"] == 0
        and data["Adrenal_and_thyroid_disorders"] == 0
    )

    if is_clearly_healthy:
        corrected_prob = min(prob, 0.5)
        return corrected_prob, "Normal"
    
    pred = int(prob >= threshold)
    result = "Abnormal" if pred == 1 else "Normal"

    return prob, result


def generate_health_tips(data, result):
    tips = []
    
    if data["Smoking"] == 1:
        tips.append("Avoid smoking to reduce hypertension risk.")

    if data["Level_of_Stress"] >= 2.0:
        tips.append("Try stress-reducing activities like meditation or yoga.")

    if data["salt_content_in_the_diet"] >= 2.0:
        tips.append("Reduce daily salt intake.")

    if data["alcohol_consumption_per_day"] >= 2.0:
        tips.append("Limit alcohol consumption.")

    if data["Physical_activity"] <= 1.0:
        tips.append("Engage in at least 30 minutes of exercise daily.")

    if data["BMI"] < 18.5:
        tips.append("Maintain a healthy body weight (underweight).")
    elif data["BMI"] > 25:
        tips.append("Maintain a healthy body weight (overweight).")

    if data["Pregnancy"] == 1:
        tips.append(
            "As you are pregnant, consult your doctor regularly for BP monitoring."
        )

    if data["Level_of_Hemoglobin"] > 16.0:
        tips.append("High hemoglobin detected; consult your doctor for advice.")

    if data["Genetic_Pedigree_Coefficient"] == 3.0:
        tips.append(
            "High genetic risk detected; regular blood pressure monitoring recommended."
        )
    elif data["Genetic_Pedigree_Coefficient"] == 2.0:
        tips.append(
            "Moderate genetic risk; maintain healthy lifestyle and monitor BP."
        )

    if not tips:
        if result == "Normal":
            tips.append("Continue healthy eating habits and regular BP check-ups.")
        else:
            tips.append(
                "Monitor your blood pressure regularly and consult a healthcare professional."
            )
            
    return tips


# --------------------------
# ROUTES
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")


@app.route("/learn")
def learn():
    return render_template("learn.html")


@app.route("/about")
def about():
    return render_template("about.html")


--------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "❌ Error: Machine Learning model is not loaded.", 500

    try:
        form = request.form

        # ---------------- Safe parsing ----------------
        def safe_cast(val, to_type, default):
            try:
                return to_type(val)
            except (TypeError, ValueError):
                return default

        # ---------------- Gather form data ----------------
        data = {
            "Sex": safe_cast(form.get("Sex"), int, 0),
            "Pregnancy": safe_cast(form.get("Pregnancy") or form.get("Pregnancy_hidden"), int, 0),
            "Smoking": safe_cast(form.get("Smoking"), int, 0),
            "Chronic_kidney_disease": safe_cast(form.get("Chronic_kidney_disease"), int, 0),
            "Adrenal_and_thyroid_disorders": safe_cast(form.get("Adrenal_and_thyroid_disorders"), int, 0),
            "Level_of_Hemoglobin": safe_cast(form.get("Level_of_Hemoglobin"), float, 0.0),
            "Age": safe_cast(form.get("Age"), float, 0.0),
            "BMI": safe_cast(form.get("BMI"), float, 0.0),
            "Genetic_Pedigree_Coefficient": form.get("Genetic_Pedigree_Coefficient", "Medium"),
            "Level_of_Stress": form.get("Level_of_Stress", "Medium"),
            "salt_content_in_the_diet": form.get("salt_content_in_the_diet", "Medium"),
            "alcohol_consumption_per_day": form.get("alcohol_consumption_per_day", "Medium"),
            "Physical_activity": form.get("Physical_activity", "Medium"),
        }

        # ---------------- Encode categorical features ----------------
        encode_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
        for key in [
            "Genetic_Pedigree_Coefficient",
            "Level_of_Stress",
            "salt_content_in_the_diet",
            "alcohol_consumption_per_day",
            "Physical_activity",
        ]:
            data[key] = encode_map.get(data[key], 2.0)  # Default to Medium

        # ---------------- Prepare input ----------------
        X_input = pd.DataFrame([data], columns=features)
        numeric_cols = X_input.select_dtypes(include=["float64", "int64"]).columns
        if len(numeric_cols) > 0:
            X_input[numeric_cols] = scaler.transform(X_input[numeric_cols])

        # ---------------- Predict probability ----------------
        prob = model.predict_proba(X_input)[:, 1][0]

        # ---------------- Override logic for clearly healthy individuals ----------------
        def apply_override(data, prob, threshold):
            healthy = (
                data["Age"] < 30
                and 18.5 <= data["BMI"] <= 25
                and data["Level_of_Stress"] == 1.0
                and data["Smoking"] == 0
                and data["Chronic_kidney_disease"] == 0
                and data["Adrenal_and_thyroid_disorders"] == 0
            )
            if healthy:
                return min(prob, 0.5), "Normal"
            return prob, "Abnormal" if prob >= threshold else "Normal"

        prob, result = apply_override(data, prob, threshold)

        # ---------------- Generate dynamic health tips ----------------
        def generate_tips(data, result):
            tips = []
            if data["Smoking"] == 1:
                tips.append("Avoid smoking to reduce hypertension risk.")
            if data["Level_of_Stress"] >= 2.0:
                tips.append("Practice stress-reducing activities like meditation or yoga.")
            if data["salt_content_in_the_diet"] >= 2.0:
                tips.append("Reduce daily salt intake.")
            if data["alcohol_consumption_per_day"] >= 2.0:
                tips.append("Limit alcohol consumption.")
            if data["Physical_activity"] <= 1.0:
                tips.append("Engage in at least 30 minutes of daily exercise.")
            if data["BMI"] < 18.5:
                tips.append("Maintain a healthy body weight (underweight).")
            elif data["BMI"] > 25:
                tips.append("Maintain a healthy body weight (overweight).")
            if data["Pregnancy"] == 1:
                tips.append("Consult your doctor regularly for BP monitoring during pregnancy.")
            if data["Level_of_Hemoglobin"] > 16.0:
                tips.append("High hemoglobin detected; consult your doctor.")
            if data["Genetic_Pedigree_Coefficient"] == 3.0:
                tips.append("High genetic risk; monitor BP regularly.")
            elif data["Genetic_Pedigree_Coefficient"] == 2.0:
                tips.append("Moderate genetic risk; maintain healthy lifestyle.")

            if not tips:
                tips.append(
                    "Continue healthy habits and regular BP check-ups." if result == "Normal"
                    else "Monitor your blood pressure and consult a healthcare professional."
                )
            return tips

        tips = generate_tips(data, result)

        # ---------------- Render result ----------------
        return render_template(
            "result.html",
            prediction=result,
            probability=round(prob * 100, 2),
            tips=tips
        )

    except Exception as e:
        print(f"[PREDICTION ERROR] {e}")
        return render_template(
            "error.html",
            error_message="⚠️ An internal error occurred during prediction. Ensure all fields are correct."
        )




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)