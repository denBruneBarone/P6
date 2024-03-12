import torch
from torch.utils.data import DataLoader
from machine_learning.Model import ModelClass
from machine_learning.config import ModelConfig, TrainingConfig
from machine_learning.prepare_for_training import TrainingDataset


def training(dataset):
    training_dataset = TrainingDataset(dataset, max_seq_length=5000)
    train_loader = DataLoader(training_dataset, batch_size=TrainingConfig.batch_size, shuffle=True)

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
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        running_loss = 0.0
        for batch_idx, batch_data in enumerate(train_loader, 1):
            inputs, sequential_data, targets, original_length = batch_data

            # Forward pass
            output = model(inputs, sequential_data, original_length)

            # Ensure the output is reshaped if needed
            # output = output.view(-1)  # Uncomment if necessary

            # Ensure the target tensor has the same size as the output tensor
            # For example, assuming targets have size [batch_size], reshape it to [batch_size, 1]
            targets = targets.view(-1, 1)

            # Compute regression loss
            loss_regression = criterion(output, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss_regression.backward()
            optimizer.step()

            running_loss += loss_regression.item()

            # Print progress every 'print_freq' batches
            print_freq = 1
            if batch_idx % print_freq == 0:
                avg_loss = running_loss / print_freq
                print(f"    Batch [{batch_idx}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Optionally, you can print or log additional information after each epoch

    print("Training finished somehow!")
