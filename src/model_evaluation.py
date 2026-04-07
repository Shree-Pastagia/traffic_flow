from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Model Accuracy:", accuracy)

    # Create confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    # Plot confusion matrix
    plt.figure(figsize=(6,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save plot
    plt.savefig(
        "outputs/plots/confusion_matrix.png"
    )

    plt.show()

    return accuracy

def plot_feature_importance(model, X):

    import pandas as pd

    importances = model.feature_importances_

    feature_names = X.columns

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_df
    )

    plt.title("Feature Importance")

    plt.savefig(
        "outputs/plots/feature_importance.png"
    )

    plt.show()