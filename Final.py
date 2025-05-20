import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load datasets and models
df = pd.read_csv(r"C:\Users\usre\Downloads\Employee-Attrition.csv")

# Load models and scalers
with open(r'C:\Users\usre\Randomforestfinal.pkl', 'rb') as file:
    attrition_model = pickle.load(file)

with open(r'C:\Users\usre\scaler1.pkl', 'rb') as f:
    attrition_scaler = pickle.load(f)

with open(r'C:\Users\usre\RM.pkl','rb') as f:
    perf_model = pickle.load(f)

with open(r'C:\Users\usre\sc.pkl', 'rb') as f:
    perf_scaler = pickle.load(f)

# Sidebar for navigation
st.sidebar.title("Prediction Options")
app_mode = st.sidebar.selectbox("Choose Prediction", ["Attrition", "Performance Rating"])

# --- Attrition Prediction ---
if app_mode == "Attrition":
    st.title("Employee Attrition Prediction")
    st.subheader("Enter Employee Details")

    Age = st.number_input("Age", min_value=18, max_value=80, value=30)
    Gender = st.selectbox("Gender", options=["Male", "Female"])
    OverTime = st.selectbox("OverTime", options=["No", "Yes"])
    TotalWorkingYears = st.number_input("Total Working Years", min_value=0, max_value=100, value=10)
    JobLevel = st.number_input("Job Level", min_value=1, max_value=5, value=2)
    JobSatisfaction = st.number_input("Job Satisfaction", min_value=1, max_value=5, value=3)
    YearsWithCurrManager = st.number_input("Years With Current Manager", min_value=0, max_value=40, value=5)

    # Mapping
    gender_map = {"Male": 1, "Female": 0}
    overtime_map = {"Yes": 1, "No": 0}
    Gender = gender_map[Gender]
    OverTime = overtime_map[OverTime]

    attrition_input = np.array([[Age, Gender, OverTime, TotalWorkingYears, JobLevel, JobSatisfaction, YearsWithCurrManager]])
    scaled_attrition_input = attrition_scaler.transform(attrition_input)

    if st.button("Predict Attrition"):
        attrition_prediction = attrition_model.predict(scaled_attrition_input)
        if attrition_prediction[0] == 1:
            st.error("⚠️ The employee is likely to leave the company.")
        else:
            st.success("✅ The employee is likely to stay with the company.")

# --- Performance Rating Prediction ---
elif app_mode == "Performance Rating":
    st.title("Employee Performance Rating Prediction")
    st.subheader("Enter Employee Details")

    Education = st.number_input("Education", min_value=1, max_value=5, value=1)
    JobInvolvement = st.number_input("JobInvolvement", min_value=1, max_value=4, value=2)
    JobLevel = st.number_input("JobLevel", min_value=1, max_value=5, value=2)
    MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, value=1000)
    YearsAtCompany = st.number_input("YearsAtCompany", min_value=0, value=10)
    YearsInCurrentRole = st.number_input("YearsInCurrentRole", min_value=0, value=10)

    perf_input = np.array([[Education, JobInvolvement, JobLevel, MonthlyIncome, YearsAtCompany, YearsInCurrentRole]])
    scaled_perf_input = perf_scaler.transform(perf_input)

    if st.button("Predict Performance Rating"):
        perf_prediction = perf_model.predict(scaled_perf_input)[0]
        st.success(f"🎯 The employee's predicted Performance Rating is: {perf_prediction}")
