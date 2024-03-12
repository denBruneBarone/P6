import numpy as np
from sklearn.tree import DecisionTreeRegressor
from machine_learning.config import TrainingConfig
from machine_learning.prepare_for_training import TrainingDataset
from sklearn.metrics import mean_squared_error

def training(dataset, validation_dataset):
    training_dataset = TrainingDataset(dataset)
    validation_dataset = TrainingDataset(validation_dataset)

    # Specify hyperparameters
    criterion = 'friedman_mse'  # You can choose from 'mse', 'friedman_mse', 'mae'
    max_depth = 5
    max_features = None  # You can choose from 'auto', 'sqrt', 'log2'
    max_leaf_nodes = 10

    # Instantiate the decision tree model with specified hyperparameters
    model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth,
                                  max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=42)

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

    # Calculate RMSE on the validation set
    val_rmse = np.sqrt(mean_squared_error(val_targets_np, val_predictions))
    print(f"Validation Root Mean Squared Error (RMSE): {val_rmse}")

    print("Training finished somehow!")
