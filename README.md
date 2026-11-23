# 🎓 Student Performance Dashboard & AI Predictor

An interactive student performance analytics and machine learning application built using Python, Scikit-learn, Pandas and Streamlit.

## 🚀 Live Application

**Try the live app here:**

https://student-performance-ai-nvaprhkzhxy7zusfqtuztw.streamlit.app/

## 📌 Project Overview

The Student Performance Dashboard (SPD) analyzes student academic data and provides an interactive interface to explore performance patterns and estimate final grades using a trained machine learning model.

The project uses the UCI Student Performance Mathematics dataset.

## ✨ Features

- **Student Performance Dashboard:** Overview of the dataset and academic performance.
- **Performance Analytics:** Visualizations of grade distributions, study time, absences and other factors.
- **AI Grade Predictor:** Predicts estimated final grades using student information.
- **Model Comparison:** Displays machine learning model comparison results.
- **Feature Insights:** Explores feature importance.
- **Interactive Interface:** Built using Streamlit.

## 🧠 Machine Learning

The project explores three regression models:

- Dummy Regressor
- Linear Regression
- Random Forest Regressor

The preprocessing pipeline uses:
- ColumnTransformer
- OneHotEncoder
- Numerical feature passthrough

The final prediction model is a trained Random Forest Regressor.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 📂 Project Structure

```text
student-performance-ai/
│
├── app.py/
│   └── app.py
│
├── data/
│   └── student-mat.csv
│
├── models/
│   └── student_performance_model.pkl
│
├── outputs/
│   ├── grade_distribution.png
│   ├── studytime_vs_grades.png
│   ├── absences_vs_grades.png
│   ├── gender_vs_grades.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
└── requirements.txt
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/meenalsingh5925-creator/student-performance-ai.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py/app.py
```

## 📊 Dataset

UCI Student Performance Dataset – Mathematics.

## 🎯 Project Goals

- Explore student academic performance through data analytics.
- Understand relationships between student characteristics and final grades.
- Apply machine learning regression techniques.
- Build and deploy an interactive predictive application.

## ⚠️ Disclaimer

Predicted grades are experimental estimates and should not be treated as actual academic results or used as the sole basis for educational decisions.

## 👩‍💻 Author

**Meenal Singh**

BS Data Science and Applications  
Indian Institute of Technology Madras

[GitHub](https://github.com/meenalsingh5925-creator)

[Portfolio](https://sites.google.com/view/meenalsingh/home)
