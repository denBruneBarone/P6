import os
import pickle
from machine_learning.pre_processing import pre_process_and_split_data
from machine_learning.prepare_for_training import format_data
from machine_learning.training import train_model, evaluate_model
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
flights_processed = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")

# Define the path for saving/loading the model
MODEL_FILE_PATH = os.path.join(PROJECT_ROOT, "machine_learning/model_file/best_model.pkl")
GRAPH_FILE_PATH = os.path.join(PROJECT_ROOT, "machine_learning/model_file/decision_tree.png")


def train():
    print("Pre-processing data...")
    input_file = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")
    print("Splitting data...")
    train_data, test_data = pre_process_and_split_data(input_file)
    print("Formatting data...")
    train_data = format_data(train_data)
    test_data = format_data(test_data)

    # Check if the model file exists
    if os.path.exists(MODEL_FILE_PATH):
        print("Loading pre-trained model...")
        with open(MODEL_FILE_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        evaluate_model(model, test_data, save_predictions_in_excel=False)
        dot_data = StringIO()
        export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        Image(filename=GRAPH_FILE_PATH)
        graph.write_png(GRAPH_FILE_PATH)

        def count_wind_speed_ranges(csv_file):
            """
            This function reads a CSV file, categorizes wind speed into ranges,
            and counts the number of rows in each category.

            Args:
                csv_file: Path to the CSV file.

            Returns:
                A Series containing wind speed category counts.
            """

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file, low_memory=False)

            # Define ranges for wind speed
            bins = [0, 5, 10, 15, 20]  # Adjust the upper limit of the last bin if needed

            # Create labels for bins
            labels = ['0-5', '5-10', '10-15', '>15']

            # Use cut function to categorize wind speed into bins
            wind_speed_category = pd.cut(df['wind_speed'], bins=bins, labels=labels)

            # Count the number of rows in each category
            wind_speed_counts = wind_speed_category.value_counts()

            return wind_speed_counts

        # Example usage
        csv_file = flights_processed  # Replace with your actual CSV file path
        wind_speed_counts = count_wind_speed_ranges(csv_file)

        print(wind_speed_counts)



    else:
        model = train_model(train_data, test_data, use_grid_search=True)

        print("Saving trained model...")
        with open(MODEL_FILE_PATH, 'wb') as model_file:
            pickle.dump(model, model_file)
        print("Model saved in " + MODEL_FILE_PATH)






if __name__ == "__main__":
    train()
