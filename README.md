# 🩺 Diabetes Prediction with Machine Learning

Ever wondered if a computer could predict whether someone might have **diabetes**—just by looking at a few health stats?  
This mini-project explores how **machine learning** can help make that guess!

---

## 💡 What is Diabetes?

**Diabetes** is a condition where the body struggles to regulate blood sugar levels. Traditionally, doctors diagnose it using lab tests.  
In this project, we build a **machine learning model** that analyzes health data (like age, blood pressure, glucose levels, etc.) to *predict* whether someone is likely to have diabetes.

> ⚠️ **Note:** This is an educational demo—not a real diagnostic tool! Always consult a medical professional for health concerns.

---

## 🤖 Machine Learning, Simplified

Imagine you studied thousands of math quizzes with the answers already filled in. Eventually, you’d start noticing patterns and could predict answers to new questions.  
That’s essentially what machine learning does: it **learns from past data** to make predictions about new cases.

We use two beginner-friendly models in this project:

| Model                  | What's the idea?                                                                 |
|------------------------|-----------------------------------------------------------------------------------|
| **Logistic Regression**| Tries to draw a straight boundary that separates people with and without diabetes.|
| **k-Nearest Neighbors**| Looks at the closest *k* similar people and goes with the majority vote.          |

---

## 🧰 What’s in This Project?

| File                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `diabetes_prediction.py` | CLI script: loads data, trains models, shows accuracy and ROC curves in the terminal. |
| `diabetes_app.py`         | Interactive **Streamlit** web app with visuals and explanations.      |
| `requirements.txt`        | List of required Python packages to run the project.                  |

---

## 🚀 How to Run It (In 5 Minutes)

```bash
# 1. Open a terminal in the project folder

# 2. (Optional) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run the terminal version
python3 diabetes_prediction.py

# OR

# 4b. Launch the Streamlit web app
streamlit run diabetes_app.py


