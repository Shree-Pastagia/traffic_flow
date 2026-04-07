import pandas as pd
import numpy as np

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

    # Add Hour
    df["Hour"] = np.random.randint(
        0,
        24,
        size=len(df)
    )

    # Add Holiday
    df["Holiday"] = np.random.choice(
        ["Yes", "No"],
        size=len(df)
    )

    # Add Road_Type
    df["Road_Type"] = np.random.choice(
        ["Highway", "Main Road", "Street"],
        size=len(df)
    )

    # Add Weather Parameters
    df["Temperature"] = np.random.randint(
        20,
        40,
        size=len(df)
    )

    df["Humidity"] = np.random.randint(
        40,
        90,
        size=len(df)
    )

    df["Visibility"] = np.random.randint(
        1,
        10,
        size=len(df)
    )

    df["Precipitation"] = np.random.randint(
        0,
        20,
        size=len(df)
    )

    

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