import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('model/churn_model.pkl', 'rb'))

# Streamlit app title
st.title('Customer Churn Prediction')

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

