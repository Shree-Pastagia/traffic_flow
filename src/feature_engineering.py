from sklearn.preprocessing import LabelEncoder
import joblib

def encode_features(X):

    # Make safe copy
    X = X.copy()

    encoders = {}

    for column in X.columns:

        if X[column].dtype == "object":

            le = LabelEncoder()

            X[column] = le.fit_transform(
                X[column]
            )

            encoders[column] = le

    # Save encoders
    joblib.dump(
        encoders,
        "models/encoders.pkl"
    )

    print("Encoders saved successfully.")

    return X