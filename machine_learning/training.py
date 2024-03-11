import torch
from torch.utils.data import DataLoader
from machine_learning.Model import ModelClass
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from machine_learning.config import ModelConfig, TrainingConfig, GridSearchConfig
from machine_learning.prepare_for_training import TrainingDataset  # Import your actual TrainingDataset class

def train_data(dataset):
    # Instantiate your dataset and dataloader using TrainingDataset (replace this with your actual dataset and dataloader)
    training_dataset = TrainingDataset(dataset)
    train_loader = DataLoader(training_dataset, batch_size=TrainingConfig.batch_size, shuffle=True)

    # Define hyperparameters and grid search parameters
    input_size = ModelConfig.input_size
    hidden_size = ModelConfig.hidden_size
    criterion = ModelConfig.criterion

    perform_grid_search = False

    param_grid = GridSearchConfig.param_grid

    # Instantiate the model
    model = ModelClass(input_size, hidden_size)

    if perform_grid_search:
        base_model = GradientBoostingRegressor()
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(training_dataset.X, training_dataset.y)

        # Use the best hyperparameters from grid search
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        # Update the model with the best hyperparameters
        model.boosted_model = GradientBoostingRegressor(**best_params)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.learning_rate)
    criterion = torch.nn.MSELoss()

    num_epochs = TrainingConfig.num_epochs
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            x, target = batch_data

            # Forward pass
            output = model(x)

            # Compute regression loss
            loss_regression = criterion(output, target)

            # Backpropagation
            optimizer.zero_grad()
            loss_regression.backward()
            optimizer.step()

            # Update the gradient boosting model if necessary
            # This depends on how you intend to use the gradient boosting model in your training process
            # For example, you might update it here based on the predictions of the neural network

