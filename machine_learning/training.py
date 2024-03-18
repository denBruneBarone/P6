import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from machine_learning.config import HPConfig
from machine_learning.prepare_for_training import TrainingDataset


def train_model(train_data):
    training_dataset = TrainingDataset(train_data)

    # Instantiate the decision tree model with specified hyperparameters
    model = DecisionTreeRegressor(criterion=HPConfig.criterion, max_depth=HPConfig.max_depth,
                                  max_features=HPConfig.max_features, max_leaf_nodes=HPConfig.max_leaf_nodes,
                                  random_state=42)

    # Extract features and targets from the training dataset
    train_features = []
    train_targets = []
    for index in range(len(training_dataset)):
        input_array, target_array = training_dataset[index]
        train_features.append(input_array)
        train_targets.append(target_array)

    # Concatenate the lists along the appropriate axis
    train_features_np = np.concatenate(train_features, axis=0)
    train_targets_np = np.concatenate(train_targets, axis=0)

    # Fit the decision tree model
    model.fit(train_features_np, train_targets_np)

    return model


def evaluate_model(model, test_data):
    test_dataset = TrainingDataset(test_data)

    print("Evaluating...")
    # Extract features and targets from the test dataset
    test_features = []
    test_targets = []
    for index in range(len(test_dataset)):
        test_input_array, test_target_array = test_dataset[index]
        test_features.append(test_input_array)
        test_targets.append(test_target_array)

    # Concatenate the lists along the appropriate axis
    test_features_np = np.concatenate(test_features, axis=0)
    test_targets_np = np.concatenate(test_targets, axis=0)

    # Predict on the test set
    test_predictions = model.predict(test_features_np)

    # Calculate RMSE for the two output parameters
    test_rmse = np.sqrt(mean_squared_error(test_targets_np, test_predictions))
    print(f"Test Root Mean Squared Error (RMSE) for Voltage and Current: {test_rmse}")

    # Calculate power consumption predictions by multiplying voltage and current predictions
    power_consumption_predictions = test_predictions[:, 0] * test_predictions[:, 1]

    # Calculate the time difference between each timestamp
    time_diff = np.diff(test_features_np[:, 0], prepend=0)

    # Adjust power consumption predictions by multiplying with the time difference
    adjusted_power_consumption = power_consumption_predictions * time_diff

    # Calculate cumulative power consumption for each flight
    cumulative_power = np.cumsum(adjusted_power_consumption)

    # Calculate RMSE on the original cumulative power for the test set
    test_targets_cumulative_power = np.cumsum(test_targets_np[:, 0] * test_targets_np[:, 1] * time_diff)
    original_test_rmse = np.sqrt(mean_squared_error(test_targets_cumulative_power, cumulative_power))

    print(f"Original Test Root Mean Squared Error (RMSE) for Cumulative Power: {original_test_rmse}")

    # Calculate RMSE on the adjusted cumulative power for the test set
    test_rmse = np.sqrt(
        mean_squared_error(test_targets_np[:, 0] * test_targets_np[:, 1], power_consumption_predictions))
    print(f"Adjusted Test Root Mean Squared Error (RMSE) for Cumulative Power: {test_rmse}")

    print("Evaluation finished!")

    return model
