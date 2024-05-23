import os
import seaborn as sns
import matplotlib.pyplot as plt
from machine_learning.prepare_for_training import format_data
from machine_learning.pre_processing import *

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
file_path = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")


def adjust_font_size(label, max_length, default_size, reduced_size):
    """Adjust font size based on label length."""
    if len(label.get_text()) > max_length:
        label.set_size(reduced_size)
    else:
        label.set_size(default_size)


def generate_correlation_heatmaps(file_path):
    # Extract csv data
    df = load_data(file_path)

    # Remove unwanted columns
    list_df = format_data(df)

    # Compute the correlation matrix
    corr_matrix = list_df.corr()

    # Set font size
    plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})

    # Generate the full heatmap
    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                     annot_kws={"fontsize": 10, "fontweight": "normal"})  # Adjust font size here
    # Adjust layout
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    max_length_x = 12
    max_length_y = 18
    default_size = 16
    reduced_size_x = 10
    reduced_size_y = 14

    # Adjust font size in x and y axis labels based on their length
    for label in ax.get_xticklabels():
        adjust_font_size(label, max_length_x, default_size, reduced_size_x)

    for label in ax.get_yticklabels():
        adjust_font_size(label, max_length_y, default_size, reduced_size_y)

    plt.title('Full Correlation Heatmap', fontsize=24, fontweight="bold")
    plt.show()


def generate_strong_correlation_heatmaps(file_path, cutoff=0.09):
    # Extract csv data
    df = load_data(file_path)

    # Remove unwanted columns
    list_df = format_data(df)

    # Compute the correlation matrix
    corr_matrix = list_df.corr()

    # Set font size
    plt.rcParams.update({'font.size': 18, 'font.weight': 'light'})

    # Generate the full heatmap
    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                     annot_kws={"fontsize": 10, "fontweight": "normal"})  # Adjust font size here
    # Adjust layout
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    max_length_x = 14
    max_length_y = 18
    default_size = 16
    reduced_size_x = 10
    reduced_size_y = 14

    # Adjust font size in x and y axis labels based on their length
    for label in ax.get_xticklabels():
        adjust_font_size(label, max_length_x, default_size, reduced_size_x)

    for label in ax.get_yticklabels():
        adjust_font_size(label, max_length_y, default_size, reduced_size_y)

    # Generate the heatmap for strong relationships
    plt.figure(figsize=(20, 16))
    strong_corr_matrix = corr_matrix.copy()
    strong_corr_matrix[abs(corr_matrix) <= cutoff] = None
    sns.heatmap(strong_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap for Strong Relationships (cutoff={})'.format(cutoff))
    plt.show()


generate_correlation_heatmaps(file_path)
