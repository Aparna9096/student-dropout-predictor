import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page config
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    with open(os.path.join(base_path, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Header ──
st.title("🎓 Student Dropout Risk Predictor")
st.markdown("#### Predict whether a student is at risk of withdrawing from their course")
st.markdown("---")

# ── Sidebar info ──
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("ROC-AUC Score", "93.78%")
    st.metric("Accuracy", "86%")
    st.metric("Recall (Dropout)", "82.47%")
    st.metric("Students Trained On", "32,593")
    st.markdown("---")
    st.markdown("**Model:** XGBoost Classifier")
    st.markdown("**Features:** 20 engineered features")
    st.markdown("**Dataset:** OULAD EdTech Dataset")

# ── Input form ──
st.subheader("📋 Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["M", "F"])
    age_band = st.selectbox(
        "Age Band", ["0-35", "35-55", "55<="])
    disability = st.selectbox("Disability", ["N", "Y"])
    region = st.selectbox("Region", [
        "London Region", "North Western Region",
        "South East Region", "South Region",
        "North Region", "East Anglian Region",
        "West Midlands Region", "East Midlands Region",
        "South West Region", "Yorkshire Region",
        "Scotland", "Ireland", "Wales"
    ])

with col2:
    st.markdown("**Academic Background**")
    highest_education = st.selectbox("Highest Education", [
        "No Formal quals",
        "Lower Than A Level",
        "A Level or Equivalent",
        "HE Qualification",
        "Post Graduate Qualification"
    ])
    imd_band = st.selectbox("IMD Band (Deprivation)", [
        "0-10%", "10-20%", "20-30%", "30-40%",
        "40-50%", "50-60%", "60-70%", "70-80%",
        "80-90%", "90-100%"
    ])
    num_of_prev_attempts = st.slider(
        "Previous Attempts", 0, 5, 0)
    studied_credits = st.slider(
        "Credits Studied", 60, 600, 120, step=60)

with col3:
    st.markdown("**Course Activity**")
    avg_score = st.slider("Average Assessment Score", 0, 100, 60)
    total_assessments = st.slider("Total Assessments Submitted", 0, 20, 5)
    submissions_on_time = st.slider("Submissions On Time", 0, 20, 4)
    total_clicks = st.number_input(
        "Total VLE Clicks", min_value=0, max_value=50000, value=1000, step=100)
    active_days = st.slider("Active Days on Platform", 0, 200, 50)

st.markdown("---")

# ── Predict button ──
if st.button("🔍 Predict Dropout Risk", use_container_width=True):

    # Encode inputs to match training
    gender_map = {"M": 1, "F": 0}
    disability_map = {"N": 0, "Y": 1}
    age_map = {"0-35": 0, "35-55": 1, "55<=": 2}
    edu_map = {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 4
    }
    imd_map = {
        "0-10%": 0, "10-20%": 1, "20-30%": 2, "30-40%": 3,
        "40-50%": 4, "50-60%": 5, "60-70%": 6,
        "70-80%": 7, "80-90%": 8, "90-100%": 9
    }
    region_map = {
        "London Region": 0, "North Western Region": 1,
        "South East Region": 2, "South Region": 3,
        "North Region": 4, "East Anglian Region": 5,
        "West Midlands Region": 6, "East Midlands Region": 7,
        "South West Region": 8, "Yorkshire Region": 9,
        "Scotland": 10, "Ireland": 11, "Wales": 12
    }

    # Build feature vector — must match feature_cols order from Phase 2
    features = np.array([[
        gender_map[gender],           # gender_encoded
        region_map[region],           # region_encoded
        edu_map[highest_education],   # highest_education_encoded
        imd_map[imd_band],            # imd_band_encoded
        age_map[age_band],            # age_band_encoded
        disability_map[disability],   # disability_encoded
        0,                            # code_module_encoded (default)
        0,                            # code_presentation_encoded (default)
        num_of_prev_attempts,         # num_of_prev_attempts
        studied_credits,              # studied_credits
        avg_score,                    # avg_score
        avg_score,                    # max_score
        avg_score * 0.7,              # min_score
        total_assessments,            # total_assessments
        submissions_on_time,          # submissions_on_time
        total_clicks,                 # total_clicks
        active_days,                  # active_days
        total_clicks / max(active_days, 1),  # avg_clicks_per_day
        -30,                          # date_registration (default)
        1                             # registered_early (default)
    ]])

    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.markdown("---")
    st.subheader("📈 Prediction Result")

    col_result, col_gauge = st.columns([1, 2])

    with col_result:
        if prediction == 1:
            st.error("⚠️ HIGH DROPOUT RISK")
            st.markdown(f"### Dropout Probability: `{probability*100:.1f}%`")
            st.markdown("""
            **Recommended Actions:**
            - 📞 Immediate counsellor outreach
            - 📚 Provide additional study resources
            - 🎯 Set up weekly check-ins
            - 💡 Offer flexible assessment deadlines
            """)
        else:
            st.success("✅ LOW DROPOUT RISK")
            st.markdown(f"### Dropout Probability: `{probability*100:.1f}%`")
            st.markdown("""
            **Student Status:**
            - 🟢 Engagement level is healthy
            - 📈 On track to complete the course
            - 🌟 Consider advanced challenge modules
            """)

    with col_gauge:
        # Risk breakdown bar
        st.markdown("**Risk Probability Breakdown**")
        st.progress(float(probability))
        
        risk_level = (
            "🔴 Critical" if probability > 0.75
            else "🟠 High" if probability > 0.5
            else "🟡 Moderate" if probability > 0.25
            else "🟢 Low"
        )
        st.markdown(f"**Risk Level:** {risk_level}")
        st.markdown(f"**Confidence:** {max(probability, 1-probability)*100:.1f}%")

# ── Footer ──
st.markdown("---")
st.markdown(
    "Built with XGBoost + SMOTE | "
    "Dataset: OULAD (Open University Learning Analytics) | "
    "Model ROC-AUC: 93.78%"
)