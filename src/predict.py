import joblib
import pandas as pd

# Load model
model = joblib.load("models/traffic_model.pkl")

# Load encoders
encoders = joblib.load("models/encoders.pkl")

# Load dataset
data = pd.read_csv(
    "data/raw/traffic_data.csv"
)

def predict_congestion(input_data):

    # Convert input to dataframe
    df_input = pd.DataFrame([input_data])

    # Get matching rows by Area
    area_rows = data[
        data["Area Name"]
        == input_data["Area Name"]
    ]

    # If area not found → use full dataset
    if area_rows.empty:

        area_rows = data

    # Compute average hidden values
    df_input["Traffic Volume"] = (
        area_rows["Traffic Volume"].mean()
    )

    df_input["Average Speed"] = (
        area_rows["Average Speed"].mean()
    )

    df_input["Public Transport Usage"] = (
        area_rows[
            "Public Transport Usage"
        ].mean()
    )

    df_input["Parking Usage"] = (
        area_rows["Parking Usage"].mean()
    )

    df_input[
        "Pedestrian and Cyclist Count"
    ] = (
        area_rows[
            "Pedestrian and Cyclist Count"
        ].mean()
    )

    df_input["Incident Reports"] = (
        area_rows["Incident Reports"].mean()
    )

    # Encode categorical columns
    for column in df_input.columns:

        if column in encoders:

            df_input[column] = encoders[
                column
            ].transform(
                df_input[column]
            )

    # Match training feature order
    df_input = df_input[
        model.feature_names_in_
    ]

    # Prediction
    prediction = model.predict(
        df_input
    )[0]

    # Confidence
    confidence = max(
        model.predict_proba(df_input)[0]
    )

    return prediction, confidence