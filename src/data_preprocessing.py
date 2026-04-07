import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Feature Engineering
    df["Day_of_week"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month
    df["Is_Weekend"] = df["Day_of_week"].isin(
        ["Saturday", "Sunday"]
    ).astype(int)

    # Create congestion category
    def categorize(x):
        if x < 40:
            return "Low"
        elif x < 70:
            return "Medium"
        else:
            return "High"

    df["Congestion_Category"] = df[
        "Congestion Level"
    ].apply(categorize)

    return df