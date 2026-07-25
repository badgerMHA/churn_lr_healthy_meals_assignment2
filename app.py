import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load the trained model and preprocessing artifacts ──────────────────────
# These two files must sit in the same directory as this script in the repo:
#   model.pkl          — the fitted LogisticRegression model
#   model_encoder.pkl   — dict bundling numeric_imputer, scaler,
#                         categorical_imputer, encoder, numeric_cols,
#                         categorical_cols (saved together in the training
#                         notebook so preprocessing at inference matches
#                         training exactly)

@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

st.title("Healthy Meals Churn Prediction Model")
st.write("Enter customer features to predict renewal probability.")

# ── Collect inputs ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    total_sessions = st.number_input("Total Sessions (2022)", min_value=0, value=0)
    total_session_duration = st.number_input("Total Session Duration (2022)", min_value=0, value=0)
    active_days = st.number_input("Active Days (2022)", min_value=0, value=0)
    count_active_quarters = st.slider("Count Active Quarters (2022)", min_value=0, max_value=4, value=0)
    avg_sessions_per_quarter = st.number_input("Avg Sessions Per Quarter", min_value=0.0, value=0.0)
    days_since_last_activity = st.number_input("Days Since Last Activity", min_value=0, value=0)

with col2:
    age = st.number_input("Age", min_value=0, value=30)
    tech_comfort_score = st.slider("Tech Comfort Score", min_value=1, max_value=5, value=3)
    income_level = st.selectbox("Income Level", encoder['encoder'].categories_[0].tolist())
    education = st.selectbox("Education", encoder['encoder'].categories_[1].tolist())
    device_type = st.selectbox("Device Type", encoder['encoder'].categories_[2].tolist())

# ── Prediction ────────────────────────────────────────────────────────────
if st.button("Predict Renewal Probability"):

    # Build raw numeric input, enforcing the same column order used in training
    numeric_df_raw = pd.DataFrame({
        'TOTAL_SESSIONS': [total_sessions],
        'TOTAL_SESSION_DURATION': [total_session_duration],
        'ACTIVE_DAYS': [active_days],
        'COUNT_ACTIVE_QUARTERS': [count_active_quarters],
        'AVG_SESSIONS_PER_QUARTER': [avg_sessions_per_quarter],
        'DAYS_SINCE_LAST_ACTIVITY': [days_since_last_activity],
        'AGE': [age],
        'TECH_COMFORT_SCORE': [tech_comfort_score],
    })[encoder['numeric_cols']]

    # Build raw categorical input, same order as training
    categorical_df_raw = pd.DataFrame({
        'INCOME_LEVEL': [income_level],
        'EDUCATION':    [education],
        'DEVICE_TYPE':  [device_type],
    })[encoder['categorical_cols']]

    # Apply the saved numeric imputer + scaler
    num_transformed = encoder['numeric_imputer'].transform(numeric_df_raw)
    num_transformed = encoder['scaler'].transform(num_transformed)

    # Apply the saved categorical imputer + one-hot encoder
    cat_transformed = encoder['categorical_imputer'].transform(categorical_df_raw)
    cat_transformed = encoder['encoder'].transform(cat_transformed)

    # Combine in the same order used during training: numeric block, then categorical block
    input_final = np.hstack([num_transformed, cat_transformed])

    prob = model.predict_proba(input_final)[0, 1]
    risk = "Low" if prob >= 0.6 else "Medium" if prob >= 0.4 else "High"

    st.subheader("Prediction")
    st.metric("Renewal Probability", f"{prob:.2%}")
    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")
