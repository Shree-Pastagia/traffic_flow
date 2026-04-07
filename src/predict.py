import joblib
import pandas as pd

# Load trained model
model = joblib.load(
    "models/traffic_model.pkl"
)

# Load encoders
encoders = joblib.load(
    "models/encoders.pkl"
)

def predict_congestion(input_data):

    df = pd.DataFrame([input_data])

    # Apply encoders to text columns
    for column in df.columns:

        if column in encoders:

            le = encoders[column]

            try:
                df[column] = le.transform(
                    df[column]
                )

            except ValueError:

                print(
                    f"Invalid value for {column}"
                )

                return "Invalid Input"

    prediction = model.predict(df)

    return prediction[0]