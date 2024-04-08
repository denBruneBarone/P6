import os
import pickle
from machine_learning.pre_processing import pre_process_and_split_data
from machine_learning.prepare_for_training import format_data
from machine_learning.training import train_model, evaluate_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
flights_processed = os.path.join(PROJECT_ROOT, "data/datasets/rodrigues/flights_processed.csv")

# Define the path for saving/loading the model
MODEL_FILE_PATH = os.path.join(PROJECT_ROOT, "machine_learning/model_file/trained_model.pkl")


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
    else:
        model = train_model(train_data, test_data, use_grid_search=True)

        print("Saving trained model...")
        with open(MODEL_FILE_PATH, 'wb') as model_file:
            pickle.dump(model, model_file)
        print("Model saved in " + MODEL_FILE_PATH)


if __name__ == "__main__":
    train()
