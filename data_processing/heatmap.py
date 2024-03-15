import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from machine_learning.prepare_for_training import format_data
from machine_learning.pre_processing import *

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
file_path = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")

# Extract csv data

df = load_data(file_path)
# Remove unwanted columns
list_df = format_data(df)

# Compute the correlation matrix
corr_matrix = list_df.corr()

# Generate a heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

