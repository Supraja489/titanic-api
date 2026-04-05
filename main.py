from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import pandas as pd

# ─── Input ───────────────────────────────────────────────────
class PassengerRequest(BaseModel):
    Pclass:   int                      # 1, 2, or 3
    Sex:      Literal['male', 'female']
    Age:      float                    # actual age in years
    SibSp:    int                      # siblings/spouses aboard
    Parch:    int                      # parents/children aboard
    Fare:     float                    # ticket price
    Embarked: Literal['S', 'C', 'Q']
    Cabin:    str = 'Unknown'          # optional — cabin number

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Pclass": 1,
                "Sex": "female",
                "Age": 25,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 71.28,
                "Embarked": "C",
                "Cabin": "C85"
            }]
        }
    }

# ─── Output ──────────────────────────────────────────────────
class SurvivalResponse(BaseModel):
    survived:    bool
    probability: float
    message:     str
    confidence:  str

# ─── App ─────────────────────────────────────────────────────
app = FastAPI(
    title="Titanic Survival Predictor",
    description="XGBoost model — 82% CV accuracy.",
    version="2.0.0"
)

model = joblib.load("model.pkl")

# ─── Feature engineering (mirrors train.py exactly) ──────────
def get_age_group(age: float) -> str:
    if age <= 2:    return 'Infant'
    elif age <= 18: return 'Young'
    elif age <= 60: return 'Adult'
    else:           return 'Old'

def get_fare_band(fare: float) -> str:
    if fare <= 7.91:    return 'Low'
    elif fare <= 14.45: return 'Medium'
    elif fare <= 31.0:  return 'High'
    else:               return 'VeryHigh'

def get_deck(cabin: str) -> str:
    if cabin and cabin != 'Unknown':
        return cabin[0]
    return 'Unknown'

def get_confidence(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:   return 'High'
    elif prob >= 0.60 or prob <= 0.40: return 'Medium'
    else:                               return 'Low'

# ─── Endpoints ───────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status":      "Titanic API v2.0 is running",
        "model":       "XGBoost — no Title feature",
        "cv_accuracy": "81.8%"
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/predict", response_model=SurvivalResponse)
def predict(request: PassengerRequest):

    # Derive all features internally
    family_size = request.SibSp + request.Parch + 1
    is_alone    = 1 if family_size == 1 else 0
    age_group   = get_age_group(request.Age)
    fare_band   = get_fare_band(request.Fare)
    deck        = get_deck(request.Cabin)

    input_df = pd.DataFrame([{
        "Pclass":     request.Pclass,
        "Sex":        request.Sex,
        "FamilySize": family_size,
        "IsAlone":    is_alone,
        "Embarked":   request.Embarked,
        "AgeGroup":   age_group,
        "FareBand":   fare_band,
        "Deck":       deck
    }])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return SurvivalResponse(
        survived    = bool(prediction),
        probability = round(float(probability), 3),
        message     = "Survived" if prediction else "Did not survive",
        confidence  = get_confidence(float(probability))
    )