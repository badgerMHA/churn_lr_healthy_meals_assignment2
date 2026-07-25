import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# -----------------------------------------------------------------------------
# Load model and encoder
# -----------------------------------------------------------------------------

from pathlib import Path

@st.cache_resource
def load_artifacts():

    base_dir = Path(__file__).parent

    with open(base_dir / "churn_lr_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)

    with open(base_dir / "churn_encoder_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder

# -----------------------------------------------------------------------------
# PAGE TITLE
# -----------------------------------------------------------------------------

st.title("Customer Renewal Probability Predictor")

st.write(
    "Enter customer attributes to predict the likelihood of subscription renewal."
)

# -----------------------------------------------------------------------------
# INPUTS
# -----------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    total_sessions = st.number_input(
        "Total Sessions (2022)",
        min_value=0,
        value=0
    )

    total_session_duration = st.number_input(
        "Total Session Duration (2022)",
        min_value=0,
        value=0
    )

    active_days = st.number_input(
        "Active Days (2022)",
        min_value=0,
        value=0
    )

    count_active_quarters = st.slider(
        "Count Active Quarters (2022)",
        min_value=0,
        max_value=4,
        value=0
    )

    avg_sessions_per_quarter = st.number_input(
        "Avg Sessions Per Quarter",
        min_value=0.0,
        value=0.0
    )

    days_since_last_activity = st.number_input(
        "Days Since Last Activity",
        min_value=0,
        value=0
    )

with col2:

    age = st.number_input(
        "Age",
        min_value=0,
        value=30
    )

    tech_comfort_score = st.slider(
        "Tech Comfort Score",
        min_value=1,
        max_value=5,
        value=3
    )

# Hard-coded categories from your notebook output

income_level = st.selectbox(
    "Income Level",
    ["High", "Low", "Medium", "Very High"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "High School", "Other", "Post-Graduate"]
)

device_type = st.selectbox(
    "Device Type",
    ["Desktop-only", "Mobile-only", "Multi-device"]
)


# -----------------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------------

if st.button("Predict Renewal Probability"):

    numeric_df_raw = pd.DataFrame({
        "TOTAL_SESSIONS": [total_sessions],
        "TOTAL_SESSION_DURATION": [total_session_duration],
        "ACTIVE_DAYS": [active_days],
        "COUNT_ACTIVE_QUARTERS": [count_active_quarters],
        "AVG_SESSIONS_PER_QUARTER": [avg_sessions_per_quarter],
        "DAYS_SINCE_LAST_ACTIVITY": [days_since_last_activity],
        "AGE": [age],
        "TECH_COMFORT_SCORE": [tech_comfort_score]
    })[encoder["numeric_cols"]]

    categorical_df_raw = pd.DataFrame({
        "INCOME_LEVEL": [income_level],
        "EDUCATION": [education],
        "DEVICE_TYPE": [device_type]
    })[encoder["categorical_cols"]]

    num_transformed = encoder["numeric_imputer"].transform(
        numeric_df_raw
    )

    num_transformed = encoder["scaler"].transform(
        num_transformed
    )

    cat_transformed = encoder["categorical_imputer"].transform(
        categorical_df_raw
    )

    cat_transformed = encoder["encoder"].transform(
        cat_transformed
    )

    input_final = np.hstack([
        num_transformed,
        cat_transformed
    ])

    prob = model.predict_proba(input_final)[0, 1]

    risk = (
        "Low"
        if prob >= 0.60
        else "Medium"
        if prob >= 0.40
        else "High"
    )

    st.subheader("Prediction")
    st.metric(
        "Renewal Probability",
        f"{prob:.2%}"
    )

    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")