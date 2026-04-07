from sklearn.preprocessing import LabelEncoder

def encode_features(X):

    # Make a copy to avoid warnings
    X = X.copy()

    le = LabelEncoder()

    for column in X.columns:
        if X[column].dtype == "object":
            X.loc[:, column] = le.fit_transform(X[column])

    return X