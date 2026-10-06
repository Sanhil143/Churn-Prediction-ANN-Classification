import pandas as pd
import tensorflow as tf
import streamlit as st
import pickle

## Load the trained model
model = tf.keras.models.load_model('ann_model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('onehot_encoder_geography.pkl','rb') as file:
    onehot_encoder_geography = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    
## streamlit app
st.title("Customer Churn Prediction")

## User input
geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=350)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])


## prepare the inout data for prediction
input_data = pd.DataFrame({
  'CreditScore': [credit_score],
  'Geography': [geography],
  'Gender': [gender],
  'Age': [age],
  'Tenure': [tenure],
  'Balance': [balance],
  'NumOfProducts': [num_of_products],
  'HasCrCard': [has_cr_card],
  'IsActiveMember': [is_active_member],
  'EstimatedSalary': [estimated_salary]
})

## Label encode the "Gender" column
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

## one hot encode the "Geography" column
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

## Combine onehot encoding with the original dataframe
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

## scale the input data
input_data_scaled = scaler.transform(input_data)

## Predict churn
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write(f"Prediction Probability of Churn: {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")