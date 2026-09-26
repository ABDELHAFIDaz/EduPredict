import streamlit as st
import pandas as pd
import joblib
import json
from edupredict.preprocessing import encode_features

model = joblib.load("models/final_pipeline.joblib")

with open("models/expected_columns.json") as f:
    expected_columns = json.load(f)

PASS_MARK = 60

st.title("EduPredict — Exam Score Predictor")
st.write("Enter a student's profile to estimate their exam score.")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider("Hours Studied (weekly)", 1, 44, 20)
    attendance = st.slider("Attendance (%)", 60, 100, 80)
    sleep_hours = st.slider("Sleep Hours", 4, 10, 7)
    previous_scores = st.slider("Previous Scores", 50, 100, 75)
    tutoring_sessions = st.slider("Tutoring Sessions (monthly)", 0, 8, 1)
    physical_activity = st.slider("Physical Activity (weekly hours)", 0, 6, 3)

    gender = st.selectbox("Gender", ["Male", "Female"])
    school_type = st.selectbox("School Type", ["Public", "Private"])
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])

with col2:
    learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
    peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])

    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
    motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
    parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
    distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

if st.button("Predict"):
    input_row = pd.DataFrame([{
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,
        "Gender": gender,
        "School_Type": school_type,
        "Extracurricular_Activities": extracurricular,
        "Internet_Access": internet_access,
        "Learning_Disabilities": learning_disabilities,
        "Peer_Influence": peer_influence,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Motivation_Level": motivation_level,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance_from_home,
    }])

    try:
        encoded_row = encode_features(input_row)
        encoded_row = encoded_row.reindex(columns=expected_columns, fill_value=0)

        prediction = model.predict(encoded_row)[0]

        st.subheader(f"Estimated Exam Score: {prediction:.1f}")

        if prediction < PASS_MARK:
            st.error(f"This student is at risk (predicted score below {PASS_MARK}).")
        else:
            st.success("This student is not flagged as at risk.")

    except ValueError as e:
        st.error(f"Invalid input: {e}")