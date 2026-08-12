# ============================================================
# SVM - Telco Customer Churn Prediction
# Target = Churn
# Kernel = RBF
# Streamlit Deployment
# ============================================================


# ============================================================
# 1) Import Libraries
# ============================================================
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="SVM Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# Title
# ============================================================

st.title("📊 SVM - Telco Customer Churn Prediction")

st.write(
    "Customer Churn Prediction using Support Vector Machine (SVM)"
)

st.write("**Kernel:** RBF")


# ============================================================
# 2) Load Dataset
# ============================================================

# Get the folder where session14.py is located
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_FILE = (
    BASE_DIR
    / "data"
    / "telco_data.csv"
)

# Model path
MODEL_FILE = (
    BASE_DIR
    / "Telco_Model.pkl"
)

print("Dataset path:", DATA_FILE)
print("Dataset exists:", DATA_FILE.exists())

df = pd.read_csv(DATA_FILE)

st.subheader("Dataset")

st.write("Dataset Shape:")
st.write(df.shape)

st.write("First 5 Rows:")
st.dataframe(df.head())


# ============================================================
# 3) Basic Information
# ============================================================

st.subheader("Dataset Information")

st.write("Dataset Information:")
st.write(df.info())

st.write("Missing Values:")
st.dataframe(df.isnull().sum().to_frame("Missing Values"))


# ============================================================
# 4) Convert TotalCharges
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# ============================================================
# 5) Remove Customer ID
# ============================================================

df = df.drop(columns=["customerID"])


# ============================================================
# 6) Convert Target Churn
# ============================================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# Target Distribution
# ============================================================

st.subheader("Target Distribution")

churn_counts = df["Churn"].value_counts().sort_index()

plt.figure(figsize=(6, 5))

plt.bar(
    ["No Churn", "Churn"],
    churn_counts.values
)

plt.title("Target Distribution")

plt.xlabel("Churn")

plt.ylabel("Count")

plt.tight_layout()

st.pyplot(plt)

plt.close()


# ============================================================
# Missing Values Plot
# ============================================================

st.subheader("Missing Values")

missing = df.isnull().sum()

missing = missing[
    missing > 0
].sort_values(
    ascending=False
)

if len(missing) > 0:

    plt.figure(figsize=(8, 5))

    plt.bar(
        missing.index,
        missing.values
    )

    plt.title("Missing Values")

    plt.xlabel("Columns")

    plt.ylabel("Number of Missing Values")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(plt)

    plt.close()

else:

    st.success("No Missing Values Found")


# ============================================================
# 7) Separate X and y
# ============================================================

X = df.drop(columns=["Churn"])

y = df["Churn"]


# ============================================================
# 8) Identify Columns
# ============================================================

numeric_cols = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = X.select_dtypes(
    include="object"
).columns.tolist()


st.subheader("Features")

st.write("Numerical Columns:")
st.write(numeric_cols)

st.write("Categorical Columns:")
st.write(categorical_cols)


# ============================================================
# 9) Preprocessing
# ============================================================

# Numerical preprocessing:
#
# Missing Values
# ↓
# Median
# ↓
# StandardScaler

numeric_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    )

])


# Categorical preprocessing:
#
# Missing Values
# ↓
# Most Frequent
# ↓
# OneHotEncoder

categorical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )

])


# Combine both pipelines

preprocessor = ColumnTransformer([

    (
        "num",
        numeric_pipeline,
        numeric_cols
    ),

    (
        "cat",
        categorical_pipeline,
        categorical_cols
    )

])


# ============================================================
# 10) Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y

)


st.subheader("Train / Test Split")

st.write("Training Samples:")
st.write(len(X_train))

st.write("Testing Samples:")
st.write(len(X_test))


# ============================================================
# 11) Build SVM Model
# ============================================================

svm_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "svm",
        SVC(

            kernel="rbf",

            C=1.0,

            gamma="scale",

            random_state=42

        )
    )

])


# ============================================================
# 12) Train Model
# ============================================================

svm_model.fit(
    X_train,
    y_train
)


# ============================================================
# 13) Make Predictions
# ============================================================

y_pred = svm_model.predict(
    X_test
)


# ============================================================
# 14) Calculate Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


# ============================================================
# 15) Print Results
# ============================================================

st.subheader("SVM RBF Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

col2.metric(
    "Precision",
    f"{precision:.4f}"
)

col3.metric(
    "Recall",
    f"{recall:.4f}"
)

col4.metric(
    "F1-Score",
    f"{f1:.4f}"
)


# ============================================================
# 16) Classification Report
# ============================================================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "No Churn",
        "Churn"
    ]
)

st.text(report)


# ============================================================
# 17) Confusion Matrix
# ============================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(figsize=(6, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
)

disp.plot(
    values_format="d",
    ax=ax
)

plt.title(
    "SVM Confusion Matrix"
)

plt.tight_layout()

st.pyplot(fig)

plt.close()


# ============================================================
# Streamlit Deployment - Customer Prediction
# ============================================================

st.divider()

st.header("🔮 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict whether "
    "the customer will churn or not."
)


# ============================================================
# Create Input Form
# ============================================================

with st.form("customer_form"):

    input_data = {}

    st.subheader("Customer Information")


    # --------------------------------------------------------
    # Numerical Inputs
    # --------------------------------------------------------

    for col in numeric_cols:

        # Get median value from original data
        default_value = float(
            X[col].median()
        )

        input_data[col] = st.number_input(
            col,
            value=default_value
        )


    # --------------------------------------------------------
    # Categorical Inputs
    # --------------------------------------------------------

    for col in categorical_cols:

        options = sorted(
            X[col].dropna().unique().tolist()
        )

        input_data[col] = st.selectbox(
            col,
            options
        )


    # --------------------------------------------------------
    # Prediction Button
    # --------------------------------------------------------

    submit_button = st.form_submit_button(
        "🔍 Predict Churn"
    )


# ============================================================
# Make Customer Prediction
# ============================================================

if submit_button:

    # Convert input dictionary to DataFrame

    new_customer = pd.DataFrame(
        [input_data]
    )


    # Make prediction

    prediction = svm_model.predict(
        new_customer
    )[0]


    # ========================================================
    # Display Prediction
    # ========================================================

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to CHURN"
        )

        st.write(
            "Prediction: **Churn**"
        )

    else:

        st.success(
            "✅ Customer is likely to STAY"
        )

        st.write(
            "Prediction: **No Churn**"
        )


# ============================================================
# END
# ============================================================

st.divider()

st.success(
    "SVM MODEL FINISHED SUCCESSFULLY"
)

