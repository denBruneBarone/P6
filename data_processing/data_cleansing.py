import pandas as pd
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


def main():
    input_file = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights.csv")
    output_file = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")

    try:
        data = pd.read_csv(input_file, low_memory=False)
        data = data[((data.route == 'R1') | (data.route == 'R2') | (data.route == 'R3') | (data.route == 'R4') |
                     (data.route == 'R5') | (data.route == 'R6') | (data.route == 'R7'))]
        data_processed = data

        data_processed.to_csv(output_file, index=False)
    except FileNotFoundError:
        print('''
        --------------------------------------------
        Error: File 'flights.csv' not found.
        Please download the file 'flights.csv' from:
        https://doi.org/10.1184/R1/12683453.v1
        --------------------------------------------''')


if __name__ == '__main__':
    main()
