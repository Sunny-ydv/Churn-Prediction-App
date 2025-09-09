import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go

# ---------- Load Model, Scaler, and Encoder ----------
with open('model/churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ⚠️ Encoder ko training ke waqt save kiya tha to use bhi load karo
try:
    with open('model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    use_encoder = True
except:
    encoder = None
    use_encoder = False

# ---------- Page Config and Styling ----------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

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
    balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=0.0, step=100.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_credit_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)

# ---------- Optional Warnings ----------
if balance > 200000 or estimated_salary > 150000:
    st.warning("⚠️ These input values are higher than typical training data. Prediction may be less accurate.")

# ---------- Prediction ----------
if st.button("🚀 Predict Churn"):

    # Prepare input data
    if use_encoder:
        categorical = [[geography, gender]]
        categorical_encoded = encoder.transform(categorical).toarray()

        numerical = np.array([[credit_score, age, tenure, balance,
                               num_products, has_credit_card, is_active_member, estimated_salary]])

        final_input = np.hstack((numerical, categorical_encoded))

    else:
        geography_map = {"France": 0, "Spain": 1, "Germany": 2}
        gender_map = {"Male": 1, "Female": 0}

        final_input = np.array([[credit_score,
                                 geography_map[geography],
                                 gender_map[gender],
                                 age,
                                 tenure,
                                 balance,
                                 num_products,
                                 has_credit_card,
                                 is_active_member,
                                 estimated_salary]])

    # Scale the input data
    input_scaled = scaler.transform(final_input)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    prediction_prob = model.predict_proba(input_scaled)[0][1]  # Probability of churn (class 1)

    # Debugging info
    with st.expander("🔍 See raw prediction data"):
        st.write("Final Input (before scaling):", final_input)
        st.write("Scaled Input:", input_scaled)
        st.write("Prediction Probability:", prediction_prob)

    # Show result
    if prediction == 1:
        st.error("🔴 This customer is **likely to churn**.")
        status = "High Risk"
    else:
        st.success("🟢 This customer is **not likely to churn**.")
        status = "Low Risk"

    # ---------- Animated Analog Gauge (Needle) ----------
    final_value = round(prediction_prob * 100, 2)

    fig = go.Figure()

    # Background arcs (gauge colors)
    fig.add_trace(go.Pie(
        values=[30, 40, 30],
        hole=0.7,
        rotation=180,
        text=["Low", "Medium", "High"],
        textinfo="text",
        textposition="inside",
        marker_colors=["lightgreen", "orange", "red"],
        direction="clockwise",
        showlegend=False
    ))

    # Function to create needle at given value
    def gauge_needle(value, radius=0.6):
        theta = (1 - value / 100) * 180
        x = radius * np.cos(np.radians(theta))
        y = radius * np.sin(np.radians(theta))
        return go.Scatter(
            x=[0, x], y=[0, y],
            mode="lines+markers",
            line=dict(color="black", width=4),
            marker=dict(size=[0, 12], color="black"),
            showlegend=False
        )

    # Frames for animation
    steps = np.linspace(0, final_value, 40)
    frames = []
    for step in steps:
        frames.append(go.Frame(data=[gauge_needle(step)], name=str(step)))

    # Initial needle at 0
    fig.add_trace(gauge_needle(0))
    fig.frames = frames

    # Layout
    fig.update_layout(
        title="Prediction Probability: {}%".format(final_value),
        xaxis=dict(range=[-1, 1], showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 1], showticklabels=False, zeroline=False),
        margin=dict(l=20, r=20, t=60, b=20),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}}]
            }]
        }]
    )

    st.plotly_chart(fig, use_container_width=True)
