# Titanic Survival Predictor API

A production-style ML API built with FastAPI, scikit-learn, and Docker.

## What it does
Predicts whether a Titanic passenger would have survived based on
passenger class, sex, age, title, siblings, parents, and embarkation port.

## Tech stack
- Model: Random Forest with sklearn Pipeline + ColumnTransformer
- API: FastAPI + Pydantic validation
- Container: Docker
- Accuracy: 77% on test set

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

Then open http://localhost:8000/docs to test the API.

## API

POST /predict

Input:
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 25,
  "SibSp": 1,
  "Parch": 0,
  "Embarked": "C",
  "Title": "Mrs"
}

Output:
{
  "survived": true,
  "probability": 0.91,
  "message": "Survived"
}