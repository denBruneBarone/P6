import pandas as pd
import os
from machine_learning.pre_processing import *

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

input_file = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights.csv")

dataframe = load_data(input_file)

flight_1_data = dataframe[dataframe['flight'] == 1]

# Extract relevant columns (position x, y, z)
positions = flight_1_data[['position_x', 'position_y', 'position_z']]

print("Length of positions DataFrame:", len(positions))

# Subtract the first entry from all entries to zero out the location and get movement differences
zeroed_positions = positions - positions.iloc[0]

print("Length of zeroed_positions DataFrame:", len(zeroed_positions))

# Calculate vectors
vectors = zeroed_positions.diff()
print("Length of vectors DataFrame:", len(vectors))

# Drop the first row since it contains NaN values due to diff()
vectors = vectors.dropna()
print("Length of vectors DataFrame after dropping NaN values:", len(vectors))

# Convert the vectors DataFrame to a list of lists
vector_list = vectors.values.tolist()

print("Length of vector_list:", len(vector_list))
