import pandas as pd
import joblib

# LOAD FILES
rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# SAMPLE INPUT
sample = {
    "AGE_YRS": 20,
    "SEX": "F",
    "NUMDAYS": 1,
    "HOSPITAL": 0,
    "DIED": 0,
    "L_THREAT": 0,
    "DISABLE": 0,
    "ER_VISIT": 0,
    "ER_ED_VISIT": 0,
    "RECOVD": 1,
    "HOSPDAYS": 0,
    "Age_Group": "Adult",
    "Severity_Score": 1,
    "YEAR": 2021,
    "RECOVERY_COMPLEXITY": "Simple",
    "SYMPTOM_CATEGORY": "Mild",
    "COVID_FLAG": 0,
    "FEVER_FLAG": 0,
    "PAIN_FLAG": 0,
    "BREATHING_FLAG": 0,
    "NEURO_FLAG": 0
}

# CONVERT TO DATAFRAME
input_df = pd.DataFrame([sample])

# ALIGN FEATURES
input_df = pd.get_dummies(input_df)

input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# PREDICT
rf_pred = rf.predict(input_df)[0]
xgb_pred = xgb.predict(input_df)[0]

# DECODE
rf_result = label_encoder.inverse_transform([rf_pred])[0]
xgb_result = label_encoder.inverse_transform([xgb_pred])[0]

print("RF Prediction:", rf_result)
print("XGB Prediction:", xgb_result)