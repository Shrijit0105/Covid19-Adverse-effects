
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier



df = pd.read_csv("UPDATED_FINAL_DATASET.csv")

# FEATURE ENGINEERING FROM ALL_SYMPTOM

df["COVID_FLAG"] = df["ALL_SYMPTOMS"].str.contains(
    "covid|corona|sars",
    case=False,
    na=False
).astype(int)

df["FEVER_FLAG"] = df["ALL_SYMPTOMS"].str.contains(
    "fever|temperature",
    case=False,
    na=False
).astype(int)

df["PAIN_FLAG"] = df["ALL_SYMPTOMS"].str.contains(
    "pain|ache|body pain|chest pain|headache",
    case=False,
    na=False
).astype(int)

df["BREATHING_FLAG"] = df["ALL_SYMPTOMS"].str.contains(
    "breath|lung|respiratory|asthma",
    case=False,
    na=False
).astype(int)

df["NEURO_FLAG"] = df["ALL_SYMPTOMS"].str.contains(
    "dizziness|seizure|neuro|brain|stroke",
    case=False,
    na=False
).astype(int)

# REMOVE RAW TEXT COLUMN


df.drop(columns=["ALL_SYMPTOMS"], inplace=True, errors="ignore")


label_encoder = LabelEncoder()

df["TARGET"] = label_encoder.fit_transform(df["Risk_Category"])



X = df.drop(columns=["Risk_Category", "TARGET"], errors="ignore")
y = df["TARGET"]



X = pd.get_dummies(X, drop_first=True)



feature_columns = X.columns

joblib.dump(feature_columns, "feature_columns.pkl")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)


xgb = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    random_state=42
)

xgb.fit(X_train, y_train)


# EVALUATION


rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

print("\nRANDOM FOREST\n")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

print("\nXGBOOST\n")
print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))



joblib.dump(rf, "rf_model.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n MODELS SAVED SUCCESSFULLY")