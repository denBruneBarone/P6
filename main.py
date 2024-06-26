import os
import data_processing.data_cleansing
from machine_learning.main import train
from pathfinding.main import find_and_show_optimal_path

if os.path.exists("data/datasets/rodrigues/flights_processed.csv") and os.path.exists(
        "data/datasets/rodrigues/flights.csv"):
    print("Correct file found. Very nice!")
else:
    print("Correct files not found: Generating...")
    data_processing.data_cleansing.main()

if __name__ == "__main__":
    train()
    find_and_show_optimal_path()
