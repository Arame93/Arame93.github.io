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
targeted = st.selectbox('Targeted', options=['yes', 'no'])
default = st.selectbox('Default', options=['yes', 'no'])
housing = st.selectbox('Housing', options=['yes', 'no'])
loan = st.selectbox('Loan', options=['yes', 'no'])
day = st.number_input('Day', min_value=1, value=10)       
month = st.selectbox('Month', options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
duration = st.number_input('Duration', min_value=1, value=10)  
campaign = st.number_input('Campaign', min_value=1, value=10) 
previous= st.number_input('Previous', min_value=1, value=10) 
poutcome= st.selectbox('Previous Outcome', options=['unknown', 'other', 'failure', 'success'])
job= st.selectbox('Previous Outcome', options=['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
education= st.selectbox('Previous Outcome', options=['primary', 'secondary', 'tertiary', 'unknown'])

# Prediction
input_data = pd.DataFrame([[age, salary, balance, marital, targeted, default, housing, loan, day, month, duration,
                           campaign, previous, poutcome, job, education]], columns=['age', 'salary', 'balance',
                          'marital', "targeted", "default", "housing", "loan", "day", "month", "duration",
                          "campaign", "previous", "poutcome", "job", "education"])
prediction = model.predict(input_data)

# Display prediction
if st.button('Predict'):
    st.write("The prediction is:" ,{prediction[0]})