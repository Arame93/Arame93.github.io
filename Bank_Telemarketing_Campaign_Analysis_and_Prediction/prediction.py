import pandas as pd
import joblib
import streamlit as st

model = joblib.load('Bank_Telemarketing_Campaign_Analysis_and_Prediction/bank_telemarketing_classifier.pkl')

st.title('Bank Telemarketing Campaign Prediction Model')
st.write('This app predicts whether a customer will accept or deny the bank term deposit offer.')

def map_prediction(prediction):
    label_mapping = {0: 'no', 1: 'yes'}
    return [label_mapping[p] for p in prediction]

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

if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_label = map_prediction(prediction)
    st.write("The prediction is:", prediction_label[0])
  
#label_mapping = {0: 'no', 1: 'yes'}
#prediction_label = label_mapping[prediction[0]]

# Display prediction
#if st.button('Predict'):
#    st.write("The prediction is:" ,prediction_label)
  
# File upload for batch prediction
st.header('Batch Prediction')
uploaded_file = st.file_uploader("Upload your dataset in CSV format", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.dataframe(data.head())
    
    # Predict using the model
    batch_prediction = model.predict(data)
    prediction_labels = map_prediction(batch_prediction)
    
    # Add predictions to the dataframe and display
    data['prediction'] = prediction_labels
    st.write("Predictions:")
    st.dataframe(data)

