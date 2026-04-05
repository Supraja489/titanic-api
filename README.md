# Titanic Survival Predictor API

A production-grade ML API built with XGBoost, FastAPI, Docker, and deployed on Google Cloud Run.

## Live API
**https://titanic-api-943335569374.us-central1.run.app/docs**

Test it directly in the browser — no setup needed.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost |
| CV Accuracy (5-fold) | **82.49%** |
| F1 Score | 0.800 |
| Training data | Full Titanic dataset (891 passengers) |
| Baseline (original notebook) | 77.2% |
| Improvement | +5.29% through systematic engineering |

---

## What makes this production-ready

**sklearn Pipeline** — preprocessing and model bundled together. No drift between training and serving. The scaler and encoder apply identically at prediction time as they did during training.

**Feature engineering** — four engineered features derived internally from raw inputs:

| Raw input | Derived feature | Why |
|-----------|----------------|-----|
| SibSp + Parch | FamilySize, IsAlone | Total family size is clearer signal than two separate numbers |
| Age | AgeGroup (Infant/Young/Adult/Old) | Binned age reduces noise |
| Fare | FareBand (Low/Medium/High/VeryHigh) | Raw fare has outliers up to £512 |
| Cabin | Deck (A–G or Unknown) | Deck letter correlates with class and survival |

**Hyperparameter tuning** — RandomizedSearchCV with 30 iterations × 5-fold CV. Best parameters found: `learning_rate=0.26, max_depth=6, n_estimators=93, subsample=0.77, colsample_bytree=0.85, min_child_weight=4`

**MLflow experiment tracking** — every training run logged with parameters, metrics, and model artifacts. Full audit trail from baseline to production.

**Docker** — containerised with a lean image using `.dockerignore` to exclude training scripts and experiment history.

**Clean API contract** — accepts raw passenger inputs only. All feature engineering happens internally. The caller doesn't need to know how the model works.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Model | XGBoost with sklearn Pipeline + ColumnTransformer |
| Experiment tracking | MLflow |
| API | FastAPI + Pydantic validation |
| Container | Docker |
| Cloud | Google Cloud Run |
| Version control | GitHub |

---

## API Reference

### POST /predict

**Input:**
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

**Output:**
```json
{
  "survived": true,
  "probability": 0.91,
  "message": "Survived",
  "confidence": "High"
}
```

**Confidence levels:**
- High — probability ≥ 0.75 or ≤ 0.25
- Medium — probability ≥ 0.60 or ≤ 0.40
- Low — probability between 0.40 and 0.60

### GET /

Returns API status and model information.

### GET /health

Health check endpoint.

---

## Run Locally

### Without Docker
```bash
conda activate mlapi
python train.py          # retrain and save model.pkl
uvicorn main:app --reload
# open http://localhost:8000/docs
```

### With Docker
```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
# open http://localhost:8000/docs
```

---

## Project Structure

```
titanic-api/
├── main.py             # FastAPI app — /predict endpoint
├── train.py            # Full preprocessing pipeline + model training
├── tune.py             # RandomizedSearchCV hyperparameter tuning
├── model.pkl           # Saved Pipeline (ColumnTransformer + XGBoost)
├── Dockerfile          # Container recipe
├── .dockerignore       # Excludes training files from container
├── requirements.txt    # Pinned library versions
└── README.md
```

---

## Experiment History

All runs tracked in MLflow. Key milestones:

| Run | CV Accuracy | Notes |
|-----|------------|-------|
| Baseline | 77.2% | Original notebook, basic features |
| + Feature engineering | 80.4% | FamilySize, IsAlone, FareBand, Deck |
| + XGBoost default | 82.04% | Algorithm upgrade from Random Forest |
| + RandomizedSearchCV | **82.49%** | Optimal hyperparameters, 30 iterations |

---

## Series

This project is the basis for a technical article series on Medium:
*From Titanic to the Cloud — A Practical ML Engineering Starter Kit*

Each article covers one layer of the ML engineering stack with working code from this project.

---

## Author

GitHub: [Supraja489](https://github.com/Supraja489)