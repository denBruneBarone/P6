import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from machine_learning.prepare_for_training import format_data
from machine_learning.pre_processing import *

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
file_path = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from machine_learning.prepare_for_training import format_data
from machine_learning.pre_processing import *


def generate_correlation_heatmaps(file_path, cutoff=0.09):
    # Extract csv data
    df = load_data(file_path)

    # Remove unwanted columns
    list_df = format_data(df)

    # Compute the correlation matrix
    corr_matrix = list_df.corr()

    # Generate the full heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Full Correlation Heatmap')
    plt.show()

    # Generate the heatmap for strong relationships
    plt.figure(figsize=(20, 16))
    strong_corr_matrix = corr_matrix.copy()
    strong_corr_matrix[abs(corr_matrix) <= cutoff] = None
    sns.heatmap(strong_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap for Strong Relationships (cutoff={})'.format(cutoff))
    plt.show()


generate_correlation_heatmaps(file_path)


