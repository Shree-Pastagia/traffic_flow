import joblib

from src.model_evaluation import (
    evaluate_model,
    plot_model_comparison,
    plot_feature_importance
)

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

    "Hour",
    "Day_of_week",
    "Month",
    "Is_Weekend",
    "Weather Conditions",
    "Temperature",
    "Humidity",
    "Visibility",
    "Precipitation",
    "Holiday",
    "Road_Type",

    # Internal features
    "Traffic Volume",
    "Average Speed",
    "Public Transport Usage",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    "Incident Reports",
    "Roadwork and Construction Activity"

]
X = df[features]

y = df["Congestion_Category"]

# Encode text
X = encode_features(X)

# Train model
model, X_test, y_test,results = train_model(X, y)

# Evaluate model
accuracy = evaluate_model(
    model,
    X_test,
    y_test
)

plot_feature_importance(model, X)
plot_model_comparison(results)


# Save model
joblib.dump(
    model,
    "models/traffic_model.pkl"
)

print("Model saved successfully.")
print("\nModel Comparison Results:")

for model_name, score in results.items():

    print(
        f"{model_name}: {score}"
    )