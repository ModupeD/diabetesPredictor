# diabetes_prediction.py
"""
Diabetes Prediction Using Logistic Regression and k-Nearest Neighbors (KNN)

This script demonstrates a complete ML workflow:
1. Data loading
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Train-test split
5. Model training (Logistic Regression & KNN)
6. Model evaluation (Confusion Matrix, ROC Curve, AUC)
7. Comparison of models

Dataset: Pima Indians Diabetes Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
)


def load_data():
    """Load the Pima Indians Diabetes dataset as a pandas DataFrame."""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df


def preprocess_data(df: pd.DataFrame):
    """Clean and split the DataFrame into features (X) and target (y)."""
    cols_with_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

    # Impute missing values using the median of each column
    df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y


def split_and_scale(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split the data into train and test sets, apply standard scaling, and return the scaler for future use."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return the scaler so it can be reused for new user input
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    """Train a KNN classifier with hyperparameter tuning (number of neighbors)."""
    param_grid = {"n_neighbors": range(3, 20, 2)}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_knn = grid.best_estimator_
    return best_knn


def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """Print evaluation metrics and plot ROC curve for the given model."""
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix - {model_name}:\n{cm}")
    print(
        f"\nClassification Report - {model_name}:\n{classification_report(y_test, y_pred)}"
    )

    # Probability scores for ROC curve
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise AttributeError(
            "The provided model does not have probability scores for ROC computation."
        )

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name).plot()
    plt.show()

    print(f"AUC for {model_name}: {roc_auc:.3f}")

    return roc_auc


def predict_user_input(model, scaler):
    """Prompt the user for feature values and predict diabetes likelihood."""

    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    print("\nEnter patient data to predict diabetes (press Enter to use 0):")
    user_values = []
    for feat in features:
        while True:
            try:
                val = input(f"  {feat}: ").strip()
                val = float(val) if val else 0.0
                user_values.append(val)
                break
            except ValueError:
                print("    Invalid input. Please enter a numeric value.")

    user_array = np.array(user_values).reshape(1, -1)
    user_array_scaled = scaler.transform(user_array)

    prediction = model.predict(user_array_scaled)[0]
    prob = (
        model.predict_proba(user_array_scaled)[0][1]
        if hasattr(model, "predict_proba")
        else None
    )

    if prediction == 1:
        print("\nThe model predicts that the patient is LIKELY to have diabetes.")
    else:
        print("\nThe model predicts that the patient is UNLIKELY to have diabetes.")

    if prob is not None:
        print(f"Probability of diabetes: {prob:.2%}")


def main():
    print("Loading data...")
    df = load_data()
    print("First 5 rows of the dataset:\n", df.head())

    print("\nPreprocessing data...")
    X, y = preprocess_data(df)

    print("\nSplitting and scaling data...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)

    print("\nTraining Logistic Regression model...")
    log_reg = train_logistic_regression(X_train_scaled, y_train)

    print("\nTraining K-Nearest Neighbors model with hyperparameter tuning...")
    knn = train_knn(X_train_scaled, y_train)
    print(f"Best K found for KNN: {knn.n_neighbors}")

    print("\nEvaluating Logistic Regression...")
    lr_auc = evaluate_model(
        log_reg, X_test_scaled, y_test, model_name="Logistic Regression"
    )

    print("\nEvaluating K-Nearest Neighbors...")
    knn_auc = evaluate_model(
        knn, X_test_scaled, y_test, model_name="K-Nearest Neighbors"
    )

    print("\nAUC Summary:")
    print(f"  Logistic Regression AUC: {lr_auc:.3f}")
    print(f"  KNN AUC: {knn_auc:.3f}")

    print("\n--- Predict Diabetes for a New Patient ---")
    predict_user_input(log_reg, scaler)


if __name__ == "__main__":
    main() 