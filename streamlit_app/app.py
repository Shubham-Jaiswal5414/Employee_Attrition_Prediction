import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("Employee Attrition Predictor")

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "model" / "model.pkl")
    preprocessor = joblib.load(BASE / "model" / "preprocessor.pkl")
    mappings = joblib.load(BASE / "model" / "binary_mappings.pkl")
    return model, preprocessor, mappings

model, preprocessor, mappings = load_artifacts()
raw_names = preprocessor.get_feature_names_out()
feature_names = [n.split("__", 1)[-1].replace("_", " ") for n in raw_names]

st.sidebar.header("Employee Details")

age = st.sidebar.slider("Age", 18, 70, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
dependents = st.sidebar.slider("Number of Dependents", 0, 6, 1)
distance = st.sidebar.slider("Distance from Home (km)", 0, 100, 15)

job_role = st.sidebar.selectbox("Job Role", ["Education", "Healthcare", "Media", "Finance", "Technology"])
job_level = st.sidebar.selectbox("Job Level", ["Entry", "Mid", "Senior"])
monthly_income = st.sidebar.slider("Monthly Income ($)", 2000, 20000, 7000, 100)
years_at_company = st.sidebar.slider("Years at Company", 0, 50, 5)
promotions = st.sidebar.slider("Number of Promotions", 0, 5, 1)
company_tenure = st.sidebar.slider("Company Tenure (months)", 0, 200, 40)
company_size = st.sidebar.selectbox("Company Size", ["Small", "Medium", "Large"])

work_life = st.sidebar.selectbox("Work-Life Balance", ["Poor", "Fair", "Good", "Excellent"])
job_satisfaction = st.sidebar.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
performance = st.sidebar.selectbox("Performance Rating", ["Below Average", "Average", "High", "Low"])
company_reputation = st.sidebar.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"])
recognition = st.sidebar.selectbox("Employee Recognition", ["Low", "Medium", "High", "Very High"])
education = st.sidebar.selectbox("Education Level", ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"])

overtime = st.sidebar.radio("Overtime", ["No", "Yes"], horizontal=True)
remote = st.sidebar.radio("Remote Work", ["No", "Yes"], horizontal=True)
leadership = st.sidebar.radio("Leadership Opportunities", ["No", "Yes"], horizontal=True)
innovation = st.sidebar.radio("Innovation Opportunities", ["No", "Yes"], horizontal=True)

threshold = st.sidebar.slider("Prediction Threshold", 0.1, 0.9, 0.5, 0.05,
    help="Probability above this = predict 'Will Leave'")

if st.sidebar.button("Predict", type="primary"):
    row = {
        "Age": age,
        "Gender": gender,
        "Years at Company": years_at_company,
        "Job Role": job_role,
        "Monthly Income": monthly_income,
        "Work-Life Balance": work_life,
        "Job Satisfaction": job_satisfaction,
        "Performance Rating": performance,
        "Number of Promotions": promotions,
        "Overtime": overtime,
        "Distance from Home": distance,
        "Education Level": education,
        "Marital Status": marital_status,
        "Number of Dependents": dependents,
        "Job Level": job_level,
        "Company Size": company_size,
        "Company Tenure": company_tenure,
        "Remote Work": remote,
        "Leadership Opportunities": leadership,
        "Innovation Opportunities": innovation,
        "Company Reputation": company_reputation,
        "Employee Recognition": recognition,
    }

    df = pd.DataFrame([row])

    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0, 1]
    pred = 1 if prob >= threshold else 0

    if pred == 1:
        st.error(f"### Prediction: WILL LEAVE")
    else:
        st.success(f"### Prediction: WILL STAY")

    st.write(f"**Probability of leaving:** {prob:.1%}")
    st.caption(f"Threshold: {threshold:.2f} → {'Leave' if prob >= threshold else 'Stay'}")

    st.subheader("Feature Impact")
    coef_df = pd.DataFrame({"feature": feature_names, "coef": model.coef_[0]})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top = coef_df.nlargest(10, "abs_coef")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in top["coef"]]
    ax.barh(top["feature"], top["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Impact on Attrition Probability")
    ax.set_title("Top 10 Factors (red = increases risk, green = decreases)")
    st.pyplot(fig)
