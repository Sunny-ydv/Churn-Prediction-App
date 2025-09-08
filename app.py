import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go

# Load the model
with open('model/churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('model/scaler.pkl', 'rb') as file:    # <-- Added scaler loading
    scaler = pickle.load(file)

# Page Config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ---------- Styling ----------
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    h1 {color: #4CAF50; text-align: center;}
    .stButton>button {
        background-color: #6c63ff;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        height: 3em;
        width: 100%;
        margin-top: 10px;
        cursor: pointer;
    }
    .segment-buttons button {
        background-color: #7A52FF;
        color: white;
        border-radius: 10px;
        margin: 10px 10px 10px 0;
        padding: 10px 20px;
        font-weight: bold;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("<h1>🌟 Customer Churn Prediction 🌟</h1>", unsafe_allow_html=True)
st.markdown("Use the input fields below to predict the likelihood of customer churn.")

# ---------- Layout ----------
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 850, 600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure", 0, 10, 3)

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=0.0, step=100.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_credit_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

status = None
prediction_prob = 0

# ---------- Prediction ----------
if st.button("🚀 Predict Churn"):

    # Encode categorical
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

    # Scale the input data before prediction
    input_data_scaled = scaler.transform(input_data)  # <-- New line to scale input

    prediction = model.predict(input_data_scaled)[0]  # Use scaled input here
    prediction_prob = model.predict_proba(input_data_scaled)[0][1]  # Probability of class 1

    # Show message
    if prediction == 1:
        st.error("🔴 This customer is **likely to churn**.")
        status = "High Risk"
    else:
        st.success("🟢 This customer is **not likely to churn**.")
        status = "Low Risk"

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prediction_prob * 100, 2),
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        title={'text': "Prediction Quality (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#6C63FF"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction_prob * 100}
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# ---------- Action Buttons ----------
st.markdown("<div class='segment-buttons'>", unsafe_allow_html=True)

if st.button("Create Segment"):
    st.success("Segment created successfully!")
    # Yahan segment creation ka logic daal sakte hain

if st.button("Create Campaign"):
    st.success("Campaign created successfully!")
    # Yahan campaign creation ka logic daal sakte hain

st.markdown("</div>", unsafe_allow_html=True)
