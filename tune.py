import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import numpy as np

# ─── Load and prepare data ───────────────────────────────────
df = pd.read_csv('train.csv')

df['Age']      = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare']     = df['Fare'].fillna(df['Fare'].median())

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
df['FareBand']   = pd.qcut(df['Fare'], q=4,
                            labels=['Low','Medium','High','VeryHigh'])
df['Deck']       = df['Cabin'].apply(
                            lambda x: x[0] if pd.notna(x) else 'Unknown')
df['AgeGroup']   = pd.cut(df['Age'], bins=[0,2,18,60,100],
                           labels=['Infant','Young','Adult','Old'])

FEATURES    = ['Pclass','Sex','FamilySize','IsAlone',
               'Embarked','AgeGroup','FareBand','Deck']
categorical = ['Sex','Embarked','AgeGroup','FareBand','Deck']
numerical   = ['Pclass','FamilySize','IsAlone']

X = df[FEATURES]
y = df['Survived']

# ─── Build pipeline ──────────────────────────────────────────
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numerical)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ))
])

# ─── Parameter distributions ─────────────────────────────────
# uniform(a, b) samples from a to a+b
# randint(a, b) samples integers from a to b
param_distributions = {
    'model__n_estimators':  randint(50, 500),      # 50 to 500 trees
    'model__learning_rate': uniform(0.01, 0.29),   # 0.01 to 0.30
    'model__max_depth':     randint(3, 8),          # depth 3 to 7
    'model__subsample':     uniform(0.6, 0.4),     # 0.6 to 1.0
    'model__colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
    'model__min_child_weight': randint(1, 6)        # 1 to 5
}

# ─── RandomizedSearchCV ──────────────────────────────────────
print("Running RandomizedSearchCV...")
print("Trying 30 random combinations × 5 folds = 150 model fits")
print("This takes about 2-3 minutes...\n")

search = RandomizedSearchCV(
    estimator  = pipeline,
    param_distributions = param_distributions,
    n_iter     = 30,        # try 30 random combinations
    cv         = 5,         # 5-fold cross-validation
    scoring    = 'accuracy',
    n_jobs     = -1,        # use all CPU cores — speeds it up
    random_state = 42,
    verbose    = 1
)

search.fit(X, y)

# ─── Results ─────────────────────────────────────────────────
best_params = search.best_params_
best_cv     = round(search.best_score_, 4)

print(f"\n{'='*45}")
print(f"  Best CV accuracy: {best_cv}")
print(f"{'='*45}")
for k, v in best_params.items():
    clean_key = k.replace('model__', '')
    print(f"  {clean_key}: {round(v, 4) if isinstance(v, float) else v}")

# ─── Log to MLflow ───────────────────────────────────────────
mlflow.set_experiment("titanic-survival")

with mlflow.start_run(run_name="RandomizedSearch-best"):

    for k, v in best_params.items():
        mlflow.log_param(k.replace('model__',''), v)

    mlflow.log_metric("cv_accuracy", best_cv)
    mlflow.sklearn.log_model(search.best_estimator_, "model")

    print(f"\nBest model logged to MLflow.")

# ─── Compare with your baseline ──────────────────────────────
baseline_cv = 0.8204
improvement = round((best_cv - baseline_cv) * 100, 2)

print(f"\n{'='*45}")
print(f"  COMPARISON")
print(f"{'='*45}")
print(f"  Baseline XGBoost cv:  {baseline_cv}")
print(f"  Tuned XGBoost cv:     {best_cv}")
print(f"  Improvement:          +{improvement}%")

# ─── Save if better ──────────────────────────────────────────
if best_cv > baseline_cv:
    joblib.dump(search.best_estimator_, "model.pkl")
    print(f"\n  New model.pkl saved — tuned version is better.")
else:
    print(f"\n  Keeping existing model.pkl — tuning didn't improve CV score.")