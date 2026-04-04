from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Literal

# --- Input shape ---
class PassengerRequest(BaseModel):
    Pclass:   int                                    # 1, 2, or 3
    Sex:      Literal['male', 'female']              # exact string
    Age:      float                                  # used to derive AgeGroup
    SibSp:    int                                    # siblings/spouses aboard
    Parch:    int                                    # parents/children aboard
    Embarked: Literal['S', 'C', 'Q']                # port of embarkation
    Title:    Literal['Mr', 'Mrs', 'Miss', 'Master', 'Rare']  # passenger title

# --- Output shape ---
class SurvivalResponse(BaseModel):
    survived:    bool
    probability: float
    message:     str

# --- App ---
app = FastAPI(title="Titanic Survival Predictor")

# --- Load model once ---
model = joblib.load("model.pkl")

def get_age_group(age: float) -> str:
    if age <= 2:
        return 'Infant'
    elif age <= 18:
        return 'Young'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Old'

@app.get("/")
def home():
    return {"status": "Titanic API is running"}

@app.post("/predict", response_model=SurvivalResponse)
def predict(request: PassengerRequest):
    # Derive AgeGroup from Age — same logic as training
    age_group = get_age_group(request.Age)

    input_df = pd.DataFrame([{
        "Pclass":   request.Pclass,
        "Sex":      request.Sex,
        "SibSp":    request.SibSp,
        "Parch":    request.Parch,
        "Embarked": request.Embarked,
        "Title":    request.Title,
        "AgeGroup": age_group
    }])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return SurvivalResponse(
        survived    = bool(prediction),
        probability = round(float(probability), 3),
        message     = "Survived" if prediction else "Did not survive"
    )   