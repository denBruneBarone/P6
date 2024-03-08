import os
import data_processing.data_cleansing
from machine_learning.main import train


if not os.path.exists("data/datasets/rodrigues/flights_processed.csv"):
    print("flights_processed.csv not found: Generating...")
    data_processing.data_cleansing.main()
else:
    print("flights_processed.csv found. Very nice!")


if __name__ == "__main__":
    train()
