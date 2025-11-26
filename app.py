# app.py — Health Recommendation Web App

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Health Recommendation App", page_icon="💉", layout="wide")

# -------------------------------
# 1️⃣ Load model
# -------------------------------
model = joblib.load("model.pkl")
preprocessor = model.named_steps['pre'] if 'pre' in model.named_steps else None
ml_model = model.named_steps['model'] if 'model' in model.named_steps else model

# -------------------------------
# 2️⃣ UI Header
# -------------------------------
st.title("💉 Health Recommendation System using Machine Learning")
st.markdown("Enter your details below to check your health category and get personalized advice.")

st.divider()

# -------------------------------
# 3️⃣ Input fields
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    recency = st.number_input("🩸 Recency (days since last donation/check-up)", min_value=0.0, value=180.0)
    frequency = st.number_input("🔁 Frequency (number of donations)", min_value=0.0, value=5.0)
with col2:
    monetary = st.number_input("💰 Monetary (total blood volume or similar)", min_value=0.0, value=2500.0)
    time_days = st.number_input("⏱️ Time (days since first donation/check-up)", min_value=0.0, value=500.0)

# -------------------------------
# 4️⃣ Prediction button
# -------------------------------
if st.button("🔍 Predict My Health Status"):
    
    # Prepare input dataframe
    user_df = pd.DataFrame([{
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'days_since': time_days
    }])

    # Feature engineering (same as training)
    for col in ['Frequency','Monetary']:
        user_df[f'{col}_log'] = np.log1p(user_df[col])

    # Add dummy RFM scores (or based on dataset quantiles)
    try:
        df_quantiles = pd.DataFrame({
            'Recency': [recency],
            'Frequency': [frequency],
            'Monetary': [monetary]
        })
        user_df['R_score'] = pd.cut(df_quantiles['Recency'], 
                                    bins=np.quantile(df_quantiles['Recency'], q=[0,0.25,0.5,0.75,1]),
                                    labels=[4,3,2,1]).astype(int)
        user_df['F_score'] = pd.cut(df_quantiles['Frequency'], 
                                    bins=np.quantile(df_quantiles['Frequency'], q=[0,0.25,0.5,0.75,1]),
                                    labels=[1,2,3,4]).astype(int)
        user_df['M_score'] = pd.cut(df_quantiles['Monetary'], 
                                    bins=np.quantile(df_quantiles['Monetary'], q=[0,0.25,0.5,0.75,1]),
                                    labels=[1,2,3,4]).astype(int)
    except:
        user_df['R_score'] = user_df['F_score'] = user_df['M_score'] = 2

    user_df['RFM_sum'] = user_df['R_score'] + user_df['F_score'] + user_df['M_score']

    feature_cols = ['Recency','Frequency','Monetary','days_since','R_score','F_score','M_score']
    X_user = user_df[feature_cols]

    # Preprocess
    if preprocessor:
        X_user_transformed = preprocessor.transform(X_user)
    else:
        X_user_transformed = X_user.values

    # Predict
    prediction = ml_model.predict(X_user_transformed)[0]
    prob = ml_model.predict_proba(X_user_transformed)[0][1] if hasattr(ml_model, 'predict_proba') else None

    # -------------------------------
    # 5️⃣ Display result
    # -------------------------------
    st.subheader("🔮 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ You are predicted as **At Risk** (Class 1)")
    else:
        st.success(f"✅ You are predicted as **Healthy / Low Risk** (Class 0)")
    
    if prob is not None:
        st.metric(label="Probability of being At-Risk", value=f"{prob*100:.2f}%")

    # -------------------------------
    # 6️⃣ Recommendation logic
    # -------------------------------
    st.markdown("### 💬 Personalized Health Recommendation:")
    if prediction == 1 or (prob and prob > 0.5):
        st.warning("""
        - You might be in a higher-risk group.  
        - Consider scheduling a health check-up soon.  
        - Maintain consistent donation and medical review habits.
        """)
    else:
        st.info("""
        - You seem to be in a healthy category.  
        - Continue regular monitoring and healthy lifestyle habits.  
        - Keep donating or checking in at regular intervals.
        """)
