import streamlit as st
import pandas as pd
import joblib


model = joblib.load("xgb_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")


st.set_page_config(
    page_title="Risk Prediction System",
    layout="wide"
)

st.title(" Vaccine Risk Prediction System")

st.markdown("Upload a CSV file OR manually enter patient details for prediction.")


# FEATURE ENGINEERING FUNCTION


def engineer_features(df):

    if "ALL_SYMPTOMS" in df.columns:

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

        df.drop(columns=["ALL_SYMPTOMS"], inplace=True, errors="ignore")

    return df


st.header(" Manual Prediction")

with st.form("manual_form"):

    AGE_YRS = st.number_input("Age", min_value=1, max_value=120)

    SEX = st.selectbox("Gender", ["M", "F"])

    NUMDAYS = st.number_input("Number of Days", min_value=0)

    HOSPITAL = st.selectbox("Hospitalized", [0, 1])

    L_THREAT = st.selectbox("Life Threat", [0, 1])

    DISABLE = st.selectbox("Disability", [0, 1])

    ER_VISIT = st.selectbox("ER Visit", [0, 1])

    ER_ED_VISIT = st.selectbox("ER Emergency Visit", [0, 1])

    RECOVD = st.selectbox("Recovered", [0, 1])

    HOSPDAYS = st.number_input("Hospital Days", min_value=0)

    Severity_Score = st.slider("Severity Score", 1, 10)

    YEAR = st.number_input("Year", min_value=2000, max_value=2035)

    Age_Group = st.selectbox(
        "Age Group",
        ["Child", "Teen", "Adult", "Senior"]
    )

    RECOVERY_COMPLEXITY = st.selectbox(
        "Recovery Complexity",
        ["Simple", "Moderate", "Complex"]
    )

    SYMPTOM_CATEGORY = st.selectbox(
        "Symptom Category",
        ["Mild", "Moderate", "Severe", "Critical"]
    )

    ALL_SYMPTOMS = st.text_area("Symptoms")

    submit = st.form_submit_button("Predict")


if submit:

    sample = pd.DataFrame([{
        "AGE_YRS": AGE_YRS,
        "SEX": SEX,
        "NUMDAYS": NUMDAYS,
        "HOSPITAL": HOSPITAL,
        "L_THREAT": L_THREAT,
        "DISABLE": DISABLE,
        "ER_VISIT": ER_VISIT,
        "ER_ED_VISIT": ER_ED_VISIT,
        "RECOVD": RECOVD,
        "HOSPDAYS": HOSPDAYS,
        "Severity_Score": Severity_Score,
        "YEAR": YEAR,
        "Age_Group": Age_Group,
        "RECOVERY_COMPLEXITY": RECOVERY_COMPLEXITY,
        "SYMPTOM_CATEGORY": SYMPTOM_CATEGORY,
        "ALL_SYMPTOMS": ALL_SYMPTOMS
    }])

    sample = engineer_features(sample)

    sample = pd.get_dummies(sample)

    sample = sample.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(sample)[0]

    result = label_encoder.inverse_transform([prediction])[0]

    st.success(f" Predicted Risk Category: {result}")


st.header(" CSV Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    csv_df = pd.read_csv(uploaded_file)

    original_df = csv_df.copy()

    csv_df = engineer_features(csv_df)

    csv_df = pd.get_dummies(csv_df)

    csv_df = csv_df.reindex(columns=feature_columns, fill_value=0)

    predictions = model.predict(csv_df)

    decoded_predictions = label_encoder.inverse_transform(predictions)

    original_df["Predicted_Risk"] = decoded_predictions

    st.success(" Predictions Generated Successfully")

    st.dataframe(original_df)

    csv = original_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Predictions CSV",
        csv,
        "predicted_results.csv",
        "text/csv"
    )