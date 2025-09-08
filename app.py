import streamlit as st
import pickle
import numpy as np

# ---------- Load Model ----------
with open("model/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------- Page Config ----------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ---------- Heading ----------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌟 Customer Churn Prediction App 🌟</h1>", unsafe_allow_html=True)
st.write("Fill out the details below to predict if a customer will churn.")

# ---------- Layout (Two Columns) ----------
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 850, 600, key="credit")
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"], key="geo")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    age = st.slider("Age", 18, 100, 30, key="age")
    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3, key="tenure")

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=0.0, step=100.0, key="balance")
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], key="products")
    has_credit_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="card")
    is_active_member = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="active")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0, key="salary")

# ---------- Predict Button ----------
if st.button("🚀 Predict Churn", key="predict_button"):
    # Encoding for gender and geography as in training
    geography_map = {"France": 0, "Spain": 1, "Germany": 2}
    gender_map = {"Male": 1, "Female": 0}

    input_data = np.array([[credit_score,
                            geography_map[geography],
                            gender_map[gender],
                            age,
                            tenure,
                            balance,
                            num_products,
                            has_credit_card,
                            is_active_member,
                            estimated_salary]])

    prediction = model.predict(input_data)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("🔴 The customer is **likely to churn**.")
    else:
        st.success("🟢 The customer is **not likely to churn**.")
