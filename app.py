import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('model/churn_model.pkl', 'rb'))

# Streamlit app title
st.title('Customer Churn Prediction')
import streamlit as st
import pickle
import numpy as np

# Load your model once at the start
with open('model/churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 850, 600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)

with col2:
    tenure = st.slider("Tenure", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, 250000.0, 0.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_credit_card = st.radio("Has Credit Card", [0, 1])
    is_active_member = st.radio("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict Churn"):
    # Preprocess inputs same as training
    # Encode categorical variables to numeric if needed
    geography_dict = {"France":0, "Spain":1, "Germany":2}
    gender_dict = {"Male":1, "Female":0}
    
    input_data = np.array([[credit_score,
                            geography_dict[geography],
                            gender_dict[gender],
                            age,
                            tenure,
                            balance,
                            num_products,
                            has_credit_card,
                            is_active_member,
                            estimated_salary]])

    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is not likely to churn.")


# User inputs
CreditScore = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
Geography = st.selectbox('Geography', ('France', 'Germany', 'Spain'))
Gender = st.selectbox('Gender', ('Male', 'Female'))
Age = st.number_input('Age', min_value=18, max_value=100, value=30)
Tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
Balance = st.number_input('Balance', min_value=0.0, value=0.0)
NumOfProducts = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox('Has Credit Card', (0, 1))
IsActiveMember = st.selectbox('Is Active Member', (0, 1))
EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

# Convert categorical variables to numeric (encoding)
geography_dict = {'France': 0, 'Germany': 1, 'Spain': 2}
gender_dict = {'Male': 1, 'Female': 0}

# Prepare input array for prediction
input_data = np.array([[CreditScore,
                        geography_dict[Geography],
                        gender_dict[Gender],
                        Age,
                        Tenure,
                        Balance,
                        NumOfProducts,
                        HasCrCard,
                        IsActiveMember,
                        EstimatedSalary]])

# Prediction button
if st.button('Predict Churn'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error('The customer is likely to leave (churn).')
    else:
        st.success('The customer is likely to stay.')

