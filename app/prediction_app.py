import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.predict import predict_congestion

# Load dataset to extract options
df = pd.read_csv("data/raw/traffic_data.csv")

# Extract unique options
areas = sorted(df["Area Name"].unique())
weather_types = sorted(df["Weather Conditions"].unique())

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


def choose_from_list(options, title):

    print(f"\nAvailable {title}:")

    for i, option in enumerate(options):
        print(f"{i}: {option}")

    index = int(input(f"\nSelect {title} number: "))

    return options[index]


def get_user_input():

    # Choose area
    area = choose_from_list(
        areas,
        "Area Name"
    )

    # Choose weather
    weather = choose_from_list(
        weather_types,
        "Weather"
    )

    # Choose day
    day = choose_from_list(
        days,
        "Day"
    )

    month = int(input("\nEnter Month (1-12): "))

    is_weekend = int(
        input("Is Weekend (1/0): ")
    )

    traffic_volume = float(
        input("Enter Traffic Volume: ")
    )

    avg_speed = float(
        input("Enter Average Speed: ")
    )

    incidents = int(
        input("Enter Incident Reports: ")
    )

    public_transport = float(
        input("Enter Public Transport Usage: ")
    )

    parking = float(
        input("Enter Parking Usage: ")
    )

    pedestrian = int(
        input("Enter Pedestrian Count: ")
    )

    roadwork = input(
        "Roadwork (Yes/No): "
    )

    data = {

        "Area Name": area,

        "Traffic Volume": traffic_volume,

        "Average Speed": avg_speed,

        "Incident Reports": incidents,

        "Public Transport Usage": public_transport,

        "Parking Usage": parking,

        "Pedestrian and Cyclist Count": pedestrian,

        "Weather Conditions": weather,

        "Roadwork and Construction Activity": roadwork,

        "Day_of_week": day,

        "Month": month,

        "Is_Weekend": is_weekend

    }

    return data


if __name__ == "__main__":

    user_data = get_user_input()

    prediction = predict_congestion(
        user_data
    )

    print("\n🚦 Predicted Traffic Level:", prediction)