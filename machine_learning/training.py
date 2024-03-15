import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from machine_learning.prepare_for_training import TrainingDataset
from sklearn.metrics import mean_squared_error
from machine_learning.config import HPConfig, GridSearchConfig
from sklearn.metrics import make_scorer

def rmse_cum_power(true_labels, predicted_labels):
    time_diff = np.diff(true_labels[:, 0], prepend=0)
    true_cumulative_power = np.cumsum(predicted_labels[:, 0] * predicted_labels[:, 1] * time_diff)
    test_targets_cumulative_power = np.cumsum(true_labels[:, 0] * true_labels[:, 1] * time_diff)

    rmse = np.sqrt(mean_squared_error(test_targets_cumulative_power, true_cumulative_power))
    return rmse


def custom_scoring(true_labels, predicted_labels):
    return rmse_cum_power(true_labels, predicted_labels)


custom_scoring = make_scorer(custom_scoring)


def training_and_evaluating(train_data, test_data, grid_search_cv=True):
    if grid_search_cv:
        training_dataset = TrainingDataset(train_data)
        test_dataset = TrainingDataset(test_data)


        # Instantiate the decision tree model with specified hyperparameters
        model = DecisionTreeRegressor()
        # splits the train-test data into n_splits number of subsets for cross validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=GridSearchConfig.param_grid,
                                   cv=cv, scoring=custom_scoring,verbose=2)
        train_features = []  # input til model
        train_targets = []  # target til model
        for index in range(len(training_dataset)):
            input_array, target_array = training_dataset[index]
            train_features.append(input_array)
            train_targets.append(target_array)



        # concatenate over axis 0, hvilket svarer til rows i et numpy array.
        # Det svarer til at lave et stort array ud ad alle input arrays, hvor hver entry er inputs til een flight.
        train_features_np = np.concatenate(train_features, axis=0)
        train_targets_np = np.concatenate(train_targets, axis=0)

        # Fit the decision tree model
        grid_search.fit(train_features_np, train_targets_np)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print("Best Params: ", best_params)
        print("Best score: ", best_score)

        best_model = grid_search.best_estimator_
        best_model.fit(train_features_np, train_targets_np)
        # TODO: Tilføj print detaljer
    else:

        training_dataset = TrainingDataset(train_data)
        test_dataset = TrainingDataset(test_data)

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

        model.fit(train_features_np, train_targets_np)

        print("Evaluating...")
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
        # TODO: find rmse på begge targets individuelt
        test_rmse = np.sqrt(mean_squared_error(test_targets_np, test_predictions))
        print(f"Test Root Mean Squared Error (RMSE) for Voltage and Current: {test_rmse}")

        # udtryk for forskellen fra cumulative power på ground truth og cumulative power på vores predictions.
        # Jo tættere på nul, jo strammere hul
        print(f"Original Test Root Mean Squared Error (RMSE) for Cumulative Power: "
              f"{rmse_cum_power(test_targets, test_predictions)}")

        print("Training finished somehow!")

        # TODO: Caasper her regnet jeg også bare cumulative power consumption på den simple måde.
        #  Her skal den også være integralet i stedet. Du kan bruge dem samme funktion som du laver i trapezoid_integration.




