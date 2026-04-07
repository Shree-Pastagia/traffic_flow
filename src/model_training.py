from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):

    # Create model
    model = RandomForestClassifier()

    # Cross Validation
    scores = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    print(
        "Cross Validation Accuracy:",
        scores.mean()
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    return model, X_test, y_test