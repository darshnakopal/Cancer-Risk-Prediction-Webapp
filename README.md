# Cancer-Risk-Prediction-Webapp


A multi-page data science web application built using **Streamlit**. The app employs a custom-built Multiple Linear Regression model optimized via Gradient Descent to predict a user's Cancer Severity Score based on demographic and lifestyle inputs.

---

## Project Origin Story

This application was developed as a mandatory team project for **Term 3** at the **Singapore University of Technology and Design (SUTD)**. While it started as a strict requirement to pass the term, it allowed us to apply foundational machine learning principles, data scaling, and matrix mathematics to a real-world medical dataset.

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
│   └── 4_Team_Members.py     # Group member profiles
├── global_cancer_patients_2015_2024.csv  # Core research dataset
├── Home.py                   # Main entry point and user guide
├── library.py                # Mathematical engine (Gradient Descent, Z-Normalization)
└── README.md                 # Project documentation