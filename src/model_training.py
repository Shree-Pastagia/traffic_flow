from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y):

    # Define models
    models = {

        "Decision Tree":
        DecisionTreeClassifier(),

        "Logistic Regression":
        LogisticRegression(
            max_iter=10000,
            solver="saga"),

        "Random Forest":
        RandomForestClassifier()

    }

    results = {}

    best_model = None
    best_score = 0

    # Train each model
    for name, model in models.items():

        scores = cross_val_score(
            model,
            X,
            y,
            cv=5
        )

        avg_score = scores.mean()

        print(
            f"{name} CV Accuracy:",
            avg_score
        )

        results[name] = avg_score

        if avg_score > best_score:

            best_score = avg_score
            best_model = model

    print(
        "\nBest Model Selected:",
        best_model
    )

    # Final train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train best model
    best_model.fit(
        X_train,
        y_train
    )

    return best_model, X_test, y_test, results