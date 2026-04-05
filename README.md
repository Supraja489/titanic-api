# Titanic Survival Predictor API

A production-style ML API built with XGBoost, FastAPI, Docker, and deployed on Google Cloud Run.

## Live API
https://titanic-api-943335569374.us-central1.run.app/docs

## What it does
Predicts whether a Titanic passenger would have survived based on
passenger details. Returns a prediction, probability, and confidence level.

## Model performance
- Algorithm: XGBoost
- CV Accuracy (5-fold): 81.8%
- Trained on: Full Titanic dataset (891 passengers)

## What makes this production-ready
- sklearn Pipeline — preprocessing and model bundled together,
  no drift between training and serving
- All features derived internally — user sends raw inputs only
- MLflow experiment tracking — every training run logged with
  parameters, metrics, and model artifacts
- Docker containerised — runs identically anywhere
- Deployed to Google Cloud Run — auto-scaling, HTTPS, always on

## Feature engineering
The API accepts raw passenger data and derives all features internally:

| Raw input | Derived feature |
|-----------|----------------|
| SibSp + Parch | FamilySize, IsAlone |
| Age | AgeGroup (Infant/Young/Adult/Old) |
| Fare | FareBand (Low/Medium/High/VeryHigh) |
| Cabin | Deck (A-G or Unknown) |

## Tech stack
- Model: XGBoost with ColumnTransformer Pipeline
- Experiment tracking: MLflow
- API: FastAPI + Pydantic validation
- Container: Docker
- Cloud: Google Cloud Run
- Version control: GitHub

## API

POST /predict

Input:
```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 25,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 71.28,
  "Embarked": "C",
  "Cabin": "C85"
}
```

Output:
```json
{
  "survived": true,
  "probability": 0.91,
  "message": "Survived",
  "confidence": "High"
}
```

## How to run locally

### Without Docker
```bash
conda activate mlapi
python train.py
uvicorn main:app --reload
```

### With Docker
```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

Then open http://localhost:8000/docs

## Experiment history
All training runs tracked in MLflow. Key experiments:

| Run | CV Accuracy | Notes |
|-----|-------------|-------|
| Baseline | 77.0% | Original features |
| + Feature engineering | 80.5% | FamilySize, IsAlone, FareBand, Deck |
| + XGBoost | 81.8% | Algorithm upgrade |
| Ablation — no Title | 81.8% | Title removed, minimal impact |