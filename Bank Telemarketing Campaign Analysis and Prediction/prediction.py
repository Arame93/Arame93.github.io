import pandas as pd
import joblib
import streamlit as st

model = joblib.load('bank_telemarketing_model.pkl')

st.title('Random Forest Classifier')
st.write('This app predicts whether a customer will accept or deny the bank term deposit offer.')

# User inputs
age = st.number_input('Age', min_value=0, max_value=100, value=30)
salary = st.number_input('Salary', min_value=0, value=50000)
balance = st.number_input('Balance', min_value=-200, value=100000)         
marital = st.selectbox('Marital Status', options=['married','single','divorced'])
housing = st.selectbox('Housing', options=['yes', 'no'])
loan = st.selectbox('Loan', options=['yes', 'no'])
poutcome= st.selectbox('Previous Outcome', options=['unknown', 'other', 'failure', 'success'])
job= st.selectbox('Job', options=['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
education= st.selectbox('Education', options=['primary', 'secondary', 'tertiary', 'unknown'])

# Prediction
input_data = pd.DataFrame([[age, salary, balance, marital, housing, loan, poutcome, job, education]], columns=['age', 'salary', 'balance',
                          'marital', "housing", "loan", "poutcome", "job", "education"])
prediction = model.predict(input_data)

label_mapping = {0: 'no', 1: 'yes'}
prediction_label = label_mapping[prediction[0]]

# Display prediction
if st.button('Predict'):
    st.write("The prediction is:" ,prediction_label)
