import pandas as pd
import matplotlib.pyplot as plt

from data_processing import datapoints_summing
from data_processing import trapeziod_integration


def read_csv():
    url = '../data/datasets/rodrigues/flights.csv'
    data = pd.read_csv(url, sep=',')
    return data


def plot_connected_graph(data_x, data_y, label, color, linestyle='-'):
    plt.plot(data_x, data_y, linestyle, color=color, marker='o', label=label)

def plot_stacked_chart(data_summing, data_integrated):
    plt.figure(figsize=(50, 30))

    # Plot connected graph for data_summing
    plt.plot(list(data_summing.keys()), list(data_summing.values()), color='blue', marker='o', label='Data Summing')

    # Plot connected graph for data_integrated
    plt.plot(list(data_integrated.keys()), list(data_integrated.values()), color='orange', marker='o', label='Data Integrated')

    # Stacked chart
    for flight in data_summing.keys():
        plt.fill_between([flight], data_summing[flight], data_integrated[flight], color='gray', alpha=0.5)

    plt.xlabel('Flight')
    plt.ylabel('Energy Consumption (Wh)')
    plt.title('Total Energy Consumption for Each Flight')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


data_summing = datapoints_summing.read_multiple_dataset()
# data_summing = datapoints_summing.read_single_dataset()

data_integrated = trapeziod_integration.integrate_multiple_flight_data(read_csv())
# data_integrated = trapeziod_integration.integrate_single_flight_data(read_csv())


plot_stacked_chart(data_summing, data_integrated)
