import pandas as pd
import matplotlib.pyplot as plt

from data_processing.energy_consumption import datapoints_summation, trapeziod_integration
from data_processing.energy_consumption.trapeziod_integration import trapezoidal_integration


def read_csv():
    url = '../../data/datasets/rodrigues/flights.csv'
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


# data_summing = datapoints_summation.calculate_energy_by_summing(read_csv())
# data_integrated = trapeziod_integration.integrate_flight_data(read_csv())
data_integrated = trapeziod_integration.integrate_each_row_specific_flight_data(read_csv())
data_integrated = trapeziod_integration.add_cumulative_column(data_integrated)

print(f"CALCULATION ENERGY CONSUMPTION FOR ALL FLIGHTS")
print("")

# # Summation
# print("-----------------------------------------------")
# print("Calculate using summation")
# print("-----------------------------------------------")
# for key, value in data_summing.items():
#     print(f"Flight {key}: ", '%.2f' % value, 'joules', sep='')  # Using string concatenation
# print("-----------------------------------------------")
# print("")

# Integration
print("-----------------------------------------------")
print("Calculate using integration")
print("-----------------------------------------------")
for key, value in data_integrated.items():
    print(f"Flight {key}: ", '%.2f' % value, ' joules', sep='')  # Using string concatenation
print("-----------------------------------------------")
print("")
# plot_stacked_chart(data_summing, data_integrated)
