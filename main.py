import joblib

from src.data_preprocessing import (
    load_data,
    preprocess_data
)

from src.feature_engineering import (
    encode_features
)

from src.model_training import (
    train_model
)

from src.model_evaluation import (
    evaluate_model
)

# Load dataset
df = load_data(
    "data/raw/traffic_data.csv"
)

# Preprocess
df = preprocess_data(df)

# Select features

features = [
    "Area Name",
    "Traffic Volume",
    "Average Speed",
    "Incident Reports",
    "Public Transport Usage",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    "Weather Conditions",
    "Roadwork and Construction Activity",
    "Day_of_week",
    "Month",
    "Is_Weekend"
]

X = df[features]

y = df["Congestion_Category"]

# Encode text
X = encode_features(X)

# Train model
model, X_test, y_test = train_model(X, y)

# Evaluate model
accuracy = evaluate_model(
    model,
    X_test,
    y_test
)

# Save model
joblib.dump(
    model,
    "models/traffic_model.pkl"
)

print("Model saved successfully.")