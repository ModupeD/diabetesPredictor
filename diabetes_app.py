import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="Diabetes Prediction Demo", layout="wide")

st.markdown(
    """
    <style>
    .main-header {font-size: 2.5rem; font-weight: 800;}
    .sub-header  {font-size: 1.3rem; margin-top: 1rem; font-weight: 600;}
    .metric-box  {background-color: #fafafa; padding: 1rem; border-radius: 8px; text-align: center; color: #111;}
    .metric-box h3 {margin-bottom: 0.5rem; color: #000;}
    .metric-box p  {margin: 0; font-size: 1.5rem; color: #000;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper functions (reuse from the script) ---------------------------------------

def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    return pd.read_csv(url)


def preprocess_data(df: pd.DataFrame):
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y


def split_and_scale(X, y):
    """Split, scale, and return the fitted scaler for future inference."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Return scaler so it can be reused when predicting new samples
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    param_grid = {"n_neighbors": range(3, 20, 2)}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    best_knn = grid.best_estimator_
    return log_reg, best_knn


def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    return cm, fpr, tpr, auc(fpr, tpr)

# --- Sidebar -------------------------------------------------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "📖 Overview",
        "🔍 Dataset",
        "⚙️ Pre-processing",
        "🏋️‍♂️ Train & Evaluate",
        "🩺 Predict",
    ],
)

# --- MAIN ----------------------------------------------------------------------------
if section == "📖 Overview":
    st.markdown("<div class='main-header'>Diabetes Prediction</div>", unsafe_allow_html=True)
    st.write(
        "This interactive demo shows how we can use **machine learning** to predict whether a patient has diabetes.\n"
        "We'll walk through loading the data, cleaning it, training two different algorithms, and evaluating how well they perform."
    )
    st.image(
        "https://images.pexels.com/photos/7088537/pexels-photo-7088537.jpeg?auto=compress&cs=tinysrgb&w=600",
        caption="Photo by Nataliya Vaitkevich",
    )

elif section == "🔍 Dataset":
    st.markdown("<div class='sub-header'>Pima Indians Diabetes Dataset</div>", unsafe_allow_html=True)
    df = load_data()
    st.write("The dataset contains **768** rows and **9** columns (8 features + 1 label).")
    st.dataframe(df.head())

    st.write("**Feature descriptions:**")
    st.markdown(
        """• **Pregnancies** – Number of times pregnant  
        • **Glucose** – Plasma glucose concentration (2 hours in an oral glucose tolerance test)  
        • **BloodPressure** – Diastolic blood pressure (mm Hg)  
        • **SkinThickness** – Triceps skin fold thickness (mm)  
        • **Insulin** – 2-Hour serum insulin (mu U/ml)  
        • **BMI** – Body mass index (kg/m²)  
        • **DiabetesPedigreeFunction** – Diabetes pedigree function  
        • **Age** – Age (years)  
        • **Outcome** – 1 = diabetic, 0 = non-diabetic"""
    )

elif section == "⚙️ Pre-processing":
    df = load_data()
    st.markdown("<div class='sub-header'>Handling Missing Values</div>", unsafe_allow_html=True)

    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    zero_counts = (df[cols_with_zero] == 0).sum()
    st.write("Number of zeros (treated as missing values):")
    st.bar_chart(zero_counts)

    X, _ = preprocess_data(df)
    st.success("Zeros replaced by NaNs and median imputation applied ✅")

    st.markdown("<div class='sub-header'>Feature Scaling</div>", unsafe_allow_html=True)
    st.write("After splitting, every numeric column is rescaled so they are on the same scale (mean 0 / std-dev 1). This helps algorithms like KNN.")

elif section == "🏋️‍♂️ Train & Evaluate":
    st.markdown("<div class='sub-header'>Model Training & Evaluation</div>", unsafe_allow_html=True)
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    with st.spinner("Training models…"):
        log_reg, knn = train_models(X_train, y_train)
    st.success("Training complete!")

    model_choice = st.selectbox("Select model to inspect", ["Logistic Regression", "K-Nearest Neighbors"])
    model = log_reg if model_choice == "Logistic Regression" else knn

    cm, fpr, tpr, roc_auc = eval_model(model, X_test, y_test)

    # Display metrics
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-box'><h3>Accuracy</h3><p>{acc:.2f}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'><h3>AUC</h3><p>{roc_auc:.2f}</p></div>", unsafe_allow_html=True)
    col3.markdown(
        f"<div class='metric-box'><h3>Best k (if KNN)</h3><p>{getattr(model, 'n_neighbors', '—')}</p></div>",
        unsafe_allow_html=True,
    )

    # Confusion Matrix heatmap
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    st.write("### ROC Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

    st.info("A higher AUC and a darker diagonal in the confusion matrix indicates better performance.")

# ---------------- PREDICTION PAGE ---------------------------------------------------
elif section == "🩺 Predict":
    st.markdown("<div class='sub-header'>Predict Diabetes for a New Patient</div>", unsafe_allow_html=True)

    # Load & train (cached to avoid re-running every interaction)
    @st.cache_resource(show_spinner=False)
    def get_model_and_scaler():
        data = load_data()
        X, y = preprocess_data(data)
        X_train_scaled, _, y_train, _, fitted_scaler = split_and_scale(X, y)
        log_reg, _ = train_models(X_train_scaled, y_train)
        return log_reg, fitted_scaler

    log_reg_model, fitted_scaler = get_model_and_scaler()

    with st.form("patient_form"):
        st.write("### Enter patient details:")

        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
            glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
            blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=140, value=70)
            skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
        with col2:
            insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=79)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.47, step=0.01)
            age = st.number_input("Age", min_value=0, max_value=120, value=33)

        submitted = st.form_submit_button("Predict")

    if submitted:
        user_array = np.array([
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
        ]).reshape(1, -1)

        user_array_scaled = fitted_scaler.transform(user_array)
        prob = log_reg_model.predict_proba(user_array_scaled)[0][1]
        prediction = "LIKELY" if prob >= 0.5 else "UNLIKELY"

        st.write("## Result")
        st.success(f"The patient is **{prediction}** to have diabetes.")
        st.metric(label="Predicted Probability of Diabetes", value=f"{prob:.2%}")
        st.caption("Note: This prediction is for educational purposes only and should not be used as medical advice.") 