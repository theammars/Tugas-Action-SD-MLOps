import joblib, pandas as pd
from model.infer import load_model, predict_dict

def load_model(path="model/artifacts/latest/stroke_pipeline.joblib"):
    return joblib.load(path)

def predict_dict(model, features: dict):
    df = pd.DataFrame([features])
    proba = float(model.predict_proba(df)[0][1])
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": proba}

if __name__ == "__main__":
    m = load_model()
    sample = {
        "gender": "Female",
        "age": 67.0,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }
    print(predict_dict(m, sample))
