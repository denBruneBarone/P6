import os

import pandas as pd
import requests
from machine_learning.pre_processing import load_data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

input_file = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights.csv")
output_file = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")
model_url = "https://kilthub.cmu.edu/ndownloader/files/26385151"


def main():
    try:
        if os.path.exists("data/datasets/rodrigues/flights.csv"):
            print("Found flights.csv")
            filter_flights()
        else:
            print('''
            --------------------------------------------
            File 'flights.csv' not found.
            Downloading the file 'flights.csv' from:
            https://doi.org/10.1184/R1/12683453.v1
            --------------------------------------------''')
            download_model()
            filter_flights()
    except FileNotFoundError:
        raise Exception('An error occured')


def download_model():
    response = requests.get(model_url, stream=True)
    print("Downloading flights.csv file")

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(input_file, "wb") as model_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    model_file.write(chunk)

        # Check if the file exists after writing
        if os.path.exists(input_file):
            print(f"Download successful: {input_file}")
            # Number of rows in flights.csv file:
            df = pd.read_csv(input_file, low_memory=False)
            num_rows = len(df)
            print("Number of rows in flights.csv:", num_rows)
            return True
        else:
            print("Error: File not found after download.")
    else:
        print(f"Error: Failed to download file. Status code: {response.status_code}")

    return False


def filter_flights():
    print("Generating flights_processed")
    data = load_data(input_file)
    # Checks if "Route" in file starts with R, and goes from 1 through 7
    data = data[data['route'].apply(lambda x: any(x.startswith(f'R{i}') for i in range(1, 8)))]
    data_processed = data
    data_processed.to_csv(output_file, index=False)

    # Number of rows in flights_processed.csv file:
    df = pd.read_csv(output_file, low_memory=False)
    num_rows = len(df)
    print("Number of rows in flights_processed.csv:", num_rows)


if __name__ == '__main__':
    main()
