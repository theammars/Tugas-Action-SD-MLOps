from model.inferences import load_model, predict_dict

def test_predict_runs():
    m = load_model()
    sample = {
        "gender": "Female", "age": 67.0, "hypertension": 0, "heart_disease": 0,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 120.5, "bmi": 27.1, "smoking_status": "never smoked"
    }
    out = predict_dict(m, sample)
    assert "prediction" in out and "probability" in out
