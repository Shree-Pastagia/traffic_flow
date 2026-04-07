import streamlit as st
import pandas as pd
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.predict import predict_congestion

# Load dataset for dropdown options
df = pd.read_csv("data/raw/traffic_data.csv")

# Extract dropdown values
areas = sorted(df["Area Name"].unique())

weather_types = sorted(
    df["Weather Conditions"].unique()
)

road_types = ["Highway", "Main Road", "Street"]

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

# Title
st.title("🚦 Traffic Congestion Prediction System")

st.markdown(
"### Enter Traffic Conditions"
)

# UI Inputs

area = st.selectbox(
    "Select Area",
    areas
)

hour = st.slider(
    "Select Hour",
    0,
    23,
    8
)

day = st.selectbox(
    "Select Day",
    days
)

month = st.slider(
    "Select Month",
    1,
    12,
    6
)

is_weekend = st.selectbox(
    "Is Weekend",
    [0, 1]
)

weather = st.selectbox(
    "Weather",
    weather_types
)

temperature = st.number_input(
    "Temperature",
    value=30
)

humidity = st.number_input(
    "Humidity",
    value=60
)

visibility = st.number_input(
    "Visibility",
    value=5
)

precipitation = st.number_input(
    "Precipitation",
    value=0
)

holiday = st.selectbox(
    "Holiday",
    ["Yes", "No"]
)

road_type = st.selectbox(
    "Road Type",
    road_types
)

roadwork = st.selectbox(
    "Roadwork Activity",
    ["Yes", "No"]
)

# Predict Button

if st.button("Predict Traffic"):

    input_data = {

        "Area Name": area,
        "Hour": hour,
        "Day_of_week": day,
        "Month": month,
        "Is_Weekend": is_weekend,
        "Weather Conditions": weather,
        "Temperature": temperature,
        "Humidity": humidity,
        "Visibility": visibility,
        "Precipitation": precipitation,
        "Holiday": holiday,
        "Road_Type": road_type,
        "Roadwork and Construction Activity":
        roadwork

    }

    prediction, confidence = predict_congestion(
        input_data
    )

    # Suggestion logic

    if prediction == "High":

        warning = "⚠ Heavy congestion expected."
        suggestion = (
            "Consider alternate routes "
            "or avoid peak hours."
        )

    elif prediction == "Medium":

        warning = "⚠ Moderate traffic expected."
        suggestion = (
            "Plan slight delays."
        )

    else:

        warning = "✅ Traffic is smooth."
        suggestion = (
            "Safe to travel."
        )

    # Output display

    st.success(
        f"🚦 Traffic Level: {prediction}"
    )

    st.info(
        f"📊 Confidence: {round(confidence*100,2)}%"
    )

    st.warning(warning)

    st.write(
        f"💡 Suggested Action: {suggestion}"
    )

    # Peak hour warning

    if hour in [8,9,10,17,18,19]:

        st.warning(
            "⚠ Peak hour detected."
        )