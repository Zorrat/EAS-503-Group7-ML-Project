import streamlit as st
import json
import requests


st.title('Video Game Sales Prediction Model')

# add sub title
st.write('This app predicts the global sales of a video game based on the user inputs')

st.header("Group 07")
st.subheader("Members: Rohan Thorat, Prathamesh Varhadpande, Nikhil Gokhale")
with open('sample_input.json') as f:
    sample_input = json.load(f)

with open('categorical_values_distinct.json') as f:
    categorical_values = json.load(f)

# Sidebar
st.sidebar.title('User Input Features')

# load inputs from the keys of the sample input

# User Inputs
# Set drop downs for categorical columns Genre, Publisher, Platform, Rating. Loading the unique values from the categorical_values_distinct.json file



user_input = {}
for key in sample_input.keys():
    if key in categorical_values:
        user_input[key] = st.sidebar.selectbox(key, categorical_values[key])
    else:
        user_input[key] = st.sidebar.text_input(key, sample_input[key])

# When the user clicks the predict button, the model will be called

if st.sidebar.button('Predict'):
    response = requests.post('http://138.197.12.80:8080/inference', json=user_input)
    prediction = response.json()
    
    # Neatly display the prediction and model name to the user from keys model_name and prediction in bold

    st.write(f"Model Name: **{prediction['model_name']}**")
    global_sales = round(float(prediction['prediction']) * 1000000 ,2)
    st.write(f"The predicted Global sales should be: **{global_sales}** million units")
    
