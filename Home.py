import streamlit as st

st.set_page_config(page_title="Home", layout="centered")

# Create three structural columns. 
# The middle column (col2) will hold your content, acting as a centered layout anchor.
col1, col2, col3 = st.columns([0.01, 5, 0.01])

with col2:
    st.header("Welcome to Cancer Prediction Web-app!", divider="blue")

    st.markdown("""
    This app uses a multiple linear regression model to predict **Cancer Severity Score** based on user input features. Follow the steps below to get started:

    ### How to Use the App:
    1. **Go to the Predictor tab** -> This is where you'll enter the required input values.
    2. **Enter values for each feature** -> Fill in the required information for each feature.
    3. **Click the Predict button** to see your predicted Cancer Severity Score.
    4. **View the result** -> The app will display the predicted output.
    """)

    # This spacing pushes the warning down slightly so it breathes better
    st.write("") 

    # Placing the warning inside col2 limits its maximum width, making it much more compact!


    st.warning("""
    ⚠️ **Disclaimer:** This tool uses past data for educational purposes only. It does **not** provide clinical suggestions, 
               medical advice, or professional diagnoses. Always consult a qualified healthcare professional 
               for medical concerns.
    """)