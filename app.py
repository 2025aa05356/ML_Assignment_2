# app.py - Wine Quality Prediction Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# URL for Wine Quality Dataset (Red Wine)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

st.title("Wine Quality Prediction ML Models")

# ----------------------
# Step 1: Load Dataset
# ----------------------
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, sep=';')
    return df

data = load_data(DATA_URL)
st.subheader("Dataset Preview")
st.dataframe(data.head())

# ----------------------
# Step 2: Model Selection
# ----------------------
model_options = ["logistic_regression_model", "decision_tree_model", "knn_model", "naive_bayes_model", "random_forest_model", "XGBoost_model"]
selected_model = st.selectbox("Select ML Model", model_options)

# ----------------------
# Step 3: Load Model
# ----------------------
model_folder = "home/cloud/ML_assignment_02/model/"
model_file = os.path.join(model_folder, selected_model + ".pkl")
st.write ("model_files:", model_file)

if not os.path.exists(model_file):
    st.warning(f"Model file '{model_file}' not found. Please make sure you've trained and saved it in 'model/' folder.")
else:
    model = joblib.load(model_file)

    # ----------------------
    # Step 4: Prepare Data
    # ----------------------
    if "quality" in data.columns:
        X_input = data.drop("quality", axis=1)
        y_true = data["quality"]
    else:
        X_input = data.copy()
        y_true = None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_input)

    # ----------------------
    # Step 5: Make Predictions
    # ----------------------
    predictions = model.predict(X_scaled)
    st.subheader("Predictions")
    st.write(predictions)

    # ----------------------
    # Step 6: Show Evaluation Metrics
    # ----------------------
    if y_true is not None:
        st.subheader("Classification Report")
        report = classification_report(y_true, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # ----------------------
        # Step 7: Confusion Matrix
        # ----------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)