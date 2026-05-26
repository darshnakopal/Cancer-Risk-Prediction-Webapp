import streamlit as st
import pandas as pd
import numpy as np
from library import *

st.set_page_config(page_title="Predictor")
st.header("ML based Prediction", divider="orange")
st.write("Answer the below questions")

#storing the user input in different variables
feature_1 = st.number_input("Enter your age", min_value=0, max_value=120, value=None)
feature_7 = st.selectbox("Select your gender", options=["Male", "Female", "Other"], index=None)
feature_2 = st.number_input("Being truthful to yourself what do you think is your obesity level in the range of 0 to 10?", min_value=0, max_value=10, step=1, value=None, placeholder="0 being not obese at all and 10 being extremely obese")
feature_3 = st.number_input("Based on your family history what do you think is your genetic risk of getting cancer in the range of 0 to 10?", min_value=0, max_value=10, step=1, value=None, placeholder="0 being no family history of cancer and 10 being a strong family history of cancer")
feature_4 = st.number_input("Being truthful to yourself what do you think is your alcohol intake level in the range of 0 to 10?", min_value=0, max_value=10, step=1, value=None, placeholder="0 being no alcohol intake and 10 being extremely high")
feature_5 = st.number_input("How will you rank air pollution around your home in the range of 0 to 10?", min_value=0, max_value=10, step=1, value=None, placeholder="0 being no pollution and 10 being extremely polluted")
feature_6 = st.number_input("Being truthful to yourself what do you think is your smoking frequency level in the range of 0 to 10?", min_value=0, max_value=10, step=1, value=None, placeholder="0 being never and 10 being daily")

gender_male = 1 if feature_7 == "Male" else 0
gender_female = 1 if feature_7 == "Female" else 0
gender_other = 1 if feature_7 == "Other" else 0 

if st.button("Predict"):
    #error handling for missing input
    if feature_1 is None or feature_2 is None or feature_3 is None or feature_4 is None or feature_5 is None or feature_6 is None or feature_7 is None:
        st.error("Please fill in all the fields")
    else:
        #create a dataframe for the input features
        
        user_data = pd.DataFrame({
            "Obesity_Level": [feature_2],
            "Age": [feature_1],
            "Genetic_Risk": [feature_3],
            "Air_Pollution": [feature_5],
            "Alcohol_Use": [feature_4],
            "Smoking": [feature_6],
            "Gender_Female": [gender_female],
            "Gender_Male": [gender_male]
        })

        #call predict function from mlCode.py
        prediction = predict_linreg_z(user_data, model_2["beta"], model_2["means"], model_2["stds"])
        st.success(f"Your predicted cancer risk is: {prediction.item():.2f}")

        #custom message based on the predicted risk
        if prediction <= 3:
            st.info("Your predicted cancer risk is low. Keep up the healthy lifestyle!")
        elif prediction < 7:
            st.warning("Your predicted cancer risk is moderate. Consider making some lifestyle changes to reduce your risk.")
        else:
            st.error("Your predicted cancer risk is high. It is recommended to consult with a healthcare professional for further evaluation and guidance.")