
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "student_performance_model.pkl"
)

st.set_page_config(
    page_title="SPD | Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🎓 Student Performance Dashboard")
st.caption("Data Analytics | Machine Learning | Student Insights")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard",
        "Performance Analytics",
        "AI Grade Predictor",
        "Model Comparison",
        "Feature Insights",
        "About Project"
    ]
)

if page == "Dashboard":

    st.header("Student Performance Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset", "395 Students")
    c2.metric("Features", "30")
    c3.metric("Target", "Final Grade G3")

    st.write(
        "Explore student performance patterns and "
        "machine learning predictions."
    )

    image_path = os.path.join(
        BASE_DIR,
        "outputs",
        "grade_distribution.png"
    )

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

elif page == "Performance Analytics":

    st.header("Performance Analytics")

    charts = [
        ("Grade Distribution", "grade_distribution.png"),
        ("Study Time vs Grades", "studytime_vs_grades.png"),
        ("Absences vs Grades", "absences_vs_grades.png"),
        ("Gender Analysis", "gender_vs_grades.png"),
        ("Correlation Heatmap", "correlation_heatmap.png")
    ]

    for title, filename in charts:

        path = os.path.join(
            BASE_DIR,
            "outputs",
            filename
        )

        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)

elif page == "AI Grade Predictor":

    st.header("AI Student Grade Predictor")

    st.info(
        "Enter student information to estimate a final grade. "
        "This is an experimental estimate, not an actual grade."
    )

    @st.cache_data
    def load_reference_data():
        return pd.read_csv(
            os.path.join(BASE_DIR, "data", "student-mat.csv"),
            sep=";"
        )

    reference_df = load_reference_data()

    input_data = {}

    st.subheader("Student Information")

    for column in reference_df.drop(
        columns=["G1", "G2", "G3"]
    ).columns:

        if reference_df[column].dtype == "object":

            options = sorted(
                reference_df[column].dropna().unique().tolist()
            )

            input_data[column] = st.selectbox(
                column,
                options
            )

        else:
            numeric_values = pd.to_numeric(
            reference_df[column], errors="coerce"
        ).dropna()

        if numeric_values.empty:
            input_data[column] = st.number_input(
                column,
                value=0
            )
        else:
            minimum = int(numeric_values.min())
            maximum = int(numeric_values.max())
            median_value = int(numeric_values.median())

            if minimum == maximum:
                input_data[column] = minimum
            else:
                input_data[column] = st.number_input(
                    column,
                    min_value=minimum,
                    max_value=maximum,
                    value=max(minimum, min(maximum, median_value))
                )


    if st.button("Predict Final Grade"):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        prediction = max(0, min(20, prediction))

        st.success(
            f"Estimated Final Grade: {prediction:.2f} / 20"
        )

        st.caption(
            "Predictions may be inaccurate and should not "
            "be used to make educational decisions."
        )

elif page == "Model Comparison":

    st.header("Machine Learning Model Comparison")

    results_path = os.path.join(
        BASE_DIR,
        "outputs",
        "model_comparison.csv"
    )

    results = pd.read_csv(results_path)

    st.dataframe(
        results,
        use_container_width=True
    )

    st.bar_chart(
        results.set_index("Model")["MAE"]
    )

elif page == "Feature Insights":

    st.header("Feature Importance")

    feature_path = os.path.join(
        BASE_DIR,
        "outputs",
        "feature_importance.csv"
    )

    features = pd.read_csv(feature_path)

    st.dataframe(
        features.head(15),
        use_container_width=True
    )

    st.bar_chart(
        features.head(15).set_index("Feature")["Importance"]
    )

elif page == "About Project":

    st.header("About SPD")

    st.write("""
    SPD is a student performance analysis and prediction project.

    **Dataset:** UCI Student Performance - Mathematics dataset.

    **Technologies:**
    - Python
    - Pandas and NumPy
    - Matplotlib and Seaborn
    - Scikit-learn
    - Streamlit

    **Machine Learning Models:**
    - Dummy Regressor
    - Linear Regression
    - Random Forest Regressor

    **Prediction Target:** Final student grade (G3).

    This project is for educational and analytical purposes.
    """)
