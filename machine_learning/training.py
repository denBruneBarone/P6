import torch
from torch.utils.data import DataLoader
from machine_learning.Model import ModelClass
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from machine_learning.config import ModelConfig, TrainingConfig, GridSearchConfig
from machine_learning.prepare_for_training import TrainingDataset


#custom collate. Fordi vi arbejder med tre tensorer: input, sequentiel of target tensor.
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    sequential_data = [item[1] for item in batch]
    targets = [item[2] for item in batch]

    return inputs, sequential_data, targets


def training(dataset):
    # Instantiate your dataset and dataloader using TrainingDataset (replace this with your actual dataset and dataloader)
    training_dataset = TrainingDataset(dataset)
    train_loader = DataLoader(training_dataset, batch_size=TrainingConfig.batch_size, shuffle=True, collate_fn=collate_fn)

    # Define hyperparameters
    input_size = ModelConfig.input_size
    hidden_size = ModelConfig.hidden_size

    # Instantiate the model
    model = ModelClass(input_size, hidden_size)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.learning_rate)
    criterion = torch.nn.MSELoss()

    num_epochs = TrainingConfig.num_epochs
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            x, target = batch_data

            # Forward pass
            output = model(x)

            # Ensure the output is reshaped if needed
            # output = output.view(-1)  # Uncomment if necessary

            # Compute regression loss
            loss_regression = criterion(output, target)

            # Backpropagation
            optimizer.zero_grad()
            loss_regression.backward()
            optimizer.step()

            # Update the gradient boosting model if necessary
            # This depends on how you intend to use the gradient boosting model in your training process
            # For example, you might update it here based on the predictions of the neural network