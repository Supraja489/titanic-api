import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import numpy as np

# ─── Load data ───────────────────────────────────────────────
df = pd.read_csv('train.csv')

# ─── Base cleaning ───────────────────────────────────────────
df['Age']      = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare']     = df['Fare'].fillna(df['Fare'].median())

# ─── Feature engineering ─────────────────────────────────────

# FamilySize — total people travelling together
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# IsAlone — direct binary signal
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# FareBand — bin raw fare into quartiles
df['FareBand'] = pd.qcut(
    df['Fare'], q=4,
    labels=['Low', 'Medium', 'High', 'VeryHigh']
)

# Deck — extract from Cabin, Unknown if missing
df['Deck'] = df['Cabin'].apply(
    lambda x: x[0] if pd.notna(x) else 'Unknown'
)

# AgeGroup — bin age into life stages
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 2, 18, 60, 100],
    labels=['Infant', 'Young', 'Adult', 'Old']
)

# ─── Features ────────────────────────────────────────────────
FEATURES    = ['Pclass', 'Sex', 'FamilySize', 'IsAlone',
               'Embarked', 'AgeGroup', 'FareBand', 'Deck']

categorical = ['Sex', 'Embarked', 'AgeGroup', 'FareBand', 'Deck']
numerical   = ['Pclass', 'FamilySize', 'IsAlone']

X = df[FEATURES]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── Shared preprocessor ─────────────────────────────────────
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numerical)
])

# ─── Helper: train, evaluate, log one run ────────────────────
def run_experiment(model, run_name, params):
    mlflow.set_experiment("titanic-survival")

    with mlflow.start_run(run_name=run_name):

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model',         model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred    = pipeline.predict(X_test)
        cv_scores = cross_val_score(
            pipeline, X, y, cv=5, scoring='accuracy'
        )

        acc    = round(accuracy_score(y_test,  y_pred), 4)
        f1     = round(f1_score(y_test,        y_pred), 4)
        prec   = round(precision_score(y_test, y_pred), 4)
        rec    = round(recall_score(y_test,    y_pred), 4)
        cv     = round(cv_scores.mean(),                4)
        cv_std = round(cv_scores.std(),                 4)

        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("features", FEATURES)

        mlflow.log_metric("accuracy",        acc)
        mlflow.log_metric("f1",              f1)
        mlflow.log_metric("precision",       prec)
        mlflow.log_metric("recall",          rec)
        mlflow.log_metric("cv_accuracy",     cv)
        mlflow.log_metric("cv_accuracy_std", cv_std)

        mlflow.sklearn.log_model(pipeline, "model")

        print(f"\n{'='*40}")
        print(f"  {run_name}")
        print(f"{'='*40}")
        print(f"  Accuracy:    {acc}")
        print(f"  F1:          {f1}")
        print(f"  Precision:   {prec}")
        print(f"  Recall:      {rec}")
        print(f"  CV (5-fold): {cv} ± {cv_std}")
        print(f"  CV folds:    {[round(s,4) for s in cv_scores]}")
        print(f"  Logged to MLflow.")

        return acc, cv

# ─── Run 1: Random Forest ────────────────────────────────────
rf_acc, rf_cv = run_experiment(
    RandomForestClassifier(n_estimators=100, random_state=42),
    run_name="RandomForest-clean",
    params={
        "model":             "RandomForest",
        "n_estimators":      100,
        "max_depth":         "None",
        "min_samples_split": 2
    }
)

# ─── Run 2: XGBoost ──────────────────────────────────────────
xgb_acc, xgb_cv = run_experiment(
    XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),
    run_name="XGBoost-clean",
    params={
        "model":         "XGBoost",
        "n_estimators":  100,
        "learning_rate": 0.1,
        "max_depth":     4
    }
)

# ─── Summary ─────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"  SUMMARY")
print(f"{'='*40}")
print(f"  RandomForest  cv: {rf_cv}")
print(f"  XGBoost       cv: {xgb_cv}")
winner = "XGBoost" if xgb_cv >= rf_cv else "RandomForest"
print(f"  Winner: {winner}")

# ─── Save production model ───────────────────────────────────
# Retrain winner on FULL dataset — no holdout needed for production
print(f"\nRetraining {winner} on full dataset...")

final_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numerical)
    ])),
    ('model', XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ))
])

final_pipeline.fit(X, y)
joblib.dump(final_pipeline, "model.pkl")
print("model.pkl saved — XGBoost on full dataset, no Title feature.")
print("Ready for API and Docker deployment.")