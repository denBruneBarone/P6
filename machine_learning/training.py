import numpy as np
from sklearn.tree import DecisionTreeRegressor
from machine_learning.prepare_for_training import TrainingDataset
from sklearn.metrics import mean_squared_error
from machine_learning.config import HPConfig


def training_and_evaluating(train_data, test_data):
    training_dataset = TrainingDataset(train_data)
    validation_dataset = TrainingDataset(test_data)

    # Instantiate the decision tree model with specified hyperparameters
    model = DecisionTreeRegressor(criterion=HPConfig.criterion, max_depth=HPConfig.max_depth,
                                  max_features=HPConfig.max_features, max_leaf_nodes=HPConfig.max_leaf_nodes, random_state=42)

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

    print("Evaluating...")
    # Evaluate on the validation set
    val_features = []
    val_targets = []
    for index in range(len(validation_dataset)):
        val_input_array, val_target_array = validation_dataset[index]
        val_features.append(val_input_array)
        val_targets.append(val_target_array)

    # Concatenate the lists along the appropriate axis
    val_features_np = np.concatenate(val_features, axis=0)
    val_targets_np = np.concatenate(val_targets, axis=0)

    # Predict on the validation set
    val_predictions = model.predict(val_features_np)

    # Calculate RMSE for the two output parameters
    val_rmse = np.sqrt(mean_squared_error(val_targets_np, val_predictions))
    print(f"Validation Root Mean Squared Error (RMSE) for Voltage and Current: {val_rmse}")

    # Calculate power consumption predictions by multiplying voltage and current predictions
    power_consumption_predictions = val_predictions[:, 0] * val_predictions[:, 1]

    # Calculate the time difference between each timestamp
    time_diff = np.diff(val_features_np[:, 0], prepend=0)

    # Adjust power consumption predictions by multiplying with the time difference
    adjusted_power_consumption = power_consumption_predictions * time_diff

    # Calculate cumulative power consumption for each flight
    cumulative_power = np.cumsum(adjusted_power_consumption)

    # Calculate RMSE on the original cumulative power for the validation set
    val_targets_cumulative_power = np.cumsum(val_targets_np[:, 0] * val_targets_np[:, 1] * time_diff)
    original_val_rmse = np.sqrt(mean_squared_error(val_targets_cumulative_power, cumulative_power))
    # udtryk for forskellen fra cumulative power på ground truth og cumulative power på vores predictions. Jo tættere på nul, jo strammere hul
    print(f"Original Validation Root Mean Squared Error (RMSE) for Cumulative Power: {original_val_rmse}")

    # Calculate RMSE on the adjusted cumulative power for the validation set
    val_rmse = np.sqrt(mean_squared_error(val_targets_np[:, 0] * val_targets_np[:, 1], power_consumption_predictions))
    # Udtryk for hvor præcis vores cumulative power på vores predictions er i forhold til sig selv. Store udsving er dårlige.
    print(f"Adjusted Validation Root Mean Squared Error (RMSE) for Cumulative Power: {val_rmse}")

    print("Training finished somehow!")

    # TODO: Caasper her regnet jeg også bare cumulative power consumption på den simple måde. Her skal den også være integralet i stedet. Du kan bruge dem samme funktion som du laver i trapezoid_integration.