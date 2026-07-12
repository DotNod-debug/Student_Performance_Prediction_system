import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Academic Performance Predictor")
st.write(
    "Predict a student's overall academic score using a few academic indicators."
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📋 Student Details")

age = st.sidebar.number_input(
    "Age",
    min_value=15,
    max_value=30,
    value=20
)

study_hours = st.sidebar.number_input(
    "Study Hours Per Day",
    min_value=0.0,
    max_value=12.0,
    value=4.0,
    step=0.5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    0,
    100,
    80
)

average_marks = st.sidebar.slider(
    "Average Subject Marks",
    0,
    100,
    75
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

extra = st.sidebar.selectbox(
    "Extra Activities",
    ["Yes", "No"]
)

# -----------------------------
# Encoding
# -----------------------------
gender_male = 1 if gender == "Male" else 0
extra_activities_yes = 1 if extra == "Yes" else 0

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Performance"):

    student = pd.DataFrame({
        "age": [age],
        "study_hours": [study_hours],
        "attendance_percentage": [attendance],
        "AverageMarks": [average_marks],
        "gender_male": [gender_male],
        "extra_activities_yes": [extra_activities_yes]
    })

    # Scale the input
    student_scaled = scaler.transform(student)

    # Predict
    prediction = model.predict(student_scaled)[0]

    # Clamp value between 0 and 100 for display
    prediction = max(0, min(prediction, 100))

    st.divider()

    st.subheader("📊 Prediction Result")

    st.metric(
        label="Predicted Overall Score",
        value=f"{prediction:.2f}"
    )

    st.progress(int(prediction))

    # Performance Message
    if prediction >= 85:
        st.success("🌟 Excellent Performance")
    elif prediction >= 70:
        st.info("👍 Good Performance")
    elif prediction >= 50:
        st.warning("📚 Average Performance")
    else:
        st.error("⚠️ Needs Improvement")

    st.divider()

    st.subheader("Student Input Summary")

    st.write(f"**Age:** {age}")
    st.write(f"**Study Hours:** {study_hours} hrs/day")
    st.write(f"**Attendance:** {attendance}%")
    st.write(f"**Average Marks:** {average_marks}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Extra Activities:** {extra}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")