"""
This module processes the data avaliable at https://doi.org/10.1184/R1/12683453.v1
"""

import pandas as pd


def main():
    try:
        data = pd.read_csv('../data/datasets/rodrigues/flights.csv', low_memory=False)
        data = data[((data.route == 'R1') | (data.route == 'R2') | (data.route == 'R3') | (data.route == 'R4') |
                     (data.route == 'R5') | (data.route == 'R6') | (data.route == 'R7'))]
        data_processed = data
        data_processed.to_csv("../data/datasets/rodrigues/flights_processed.csv", index=False)
    except FileNotFoundError:
        print('''
        --------------------------------------------
        Error: File 'flights.csv' not found.
        Please download the file 'flights.csv' from:
        https://doi.org/10.1184/R1/12683453.v1
        --------------------------------------------''')



if __name__ == '__main__':
    main()
