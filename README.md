# Cancer-Risk-Prediction-Webapp


A multi-page data science web application built using **Streamlit**. The app employs a custom-built Multiple Linear Regression model to predict a user's Cancer Severity Score based on demographic and lifestyle inputs.


---

## Features

* **Interactive Risk Calculator:** Users can input age, gender, obesity levels, genetic history, alcohol consumption, air pollution exposure, and smoking habits to receive an instant prediction score.
* **Custom ML Engine:** Built entirely from scratch without relying on high-level ML libraries like Scikit-Learn for model execution.
* **Multi-Page Navigation:** Seamlessly browse between the App Home, Scientific Background, Predictor Engine, and Frequently Asked Questions.

---

## Repository Structure

```text
├── pages/
│   ├── 1_Background.py       # Scientific details on cancer metrics
│   ├── 2_Predictor.py        # Streamlit UI for the prediction model
│   ├── 3_FAQs.py             # Term project origin story and project FAQs
├── global_cancer_patients_2015_2024.csv  # Core research dataset
├── Home.py                   # Main entry point and user guide
├── library.py                # Mathematical engine (Gradient Descent, Z-Normalization)
└── README.md                 # Project documentation