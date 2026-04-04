import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Load your actual Titanic CSV ---
df = pd.read_csv('/Users/suprajakommuri/titanic-api/train.csv')

# --- Preprocessing (same as your notebook) ---
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])

# Recreate Title feature
name_split = df['Name'].str.split(r'[,.\s]+', expand=True)
df['Title'] = name_split[1]

def simplify_title(title):
    if title in ['Mr']:
        return 'Mr'
    elif title in ['Miss', 'Ms', 'Mlle']:
        return 'Miss'
    elif title in ['Mrs', 'Mme']:
        return 'Mrs'
    elif title in ['Master']:
        return 'Master'
    else:
        return 'Rare'

df['Title'] = df['Title'].apply(simplify_title)

# Recreate AgeGroup feature
def age_group(age):
    if age <= 2:
        return 'Infant'
    elif age <= 18:
        return 'Young'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Old'

df['AgeGroup'] = df['Age'].apply(age_group)

# --- Select final features ---
FEATURES = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'AgeGroup']
X = df[FEATURES]
y = df['Survived']

# --- Build Pipeline with encoding inside it ---
categorical = ['Sex', 'Embarked', 'Title', 'AgeGroup']
numerical   = ['Pclass', 'SibSp', 'Parch']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numerical)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)
print(f"Model trained. Features: {FEATURES}")

joblib.dump(pipeline, 'model.pkl')
print("Saved to model.pkl")