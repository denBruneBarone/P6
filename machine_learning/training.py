import torch
from torch.utils.data import DataLoader
from machine_learning.Model import ModelClass
from machine_learning.config import ModelConfig, TrainingConfig
from machine_learning.prepare_for_training import TrainingDataset


def training_and_validation(train_dataset, val_dataset):
    training_dataset = TrainingDataset(train_dataset, max_seq_length=5000)
    validation_dataset = TrainingDataset(val_dataset, max_seq_length=5000) # Usikker på denne!! OBS

    train_loader = DataLoader(training_dataset, batch_size=TrainingConfig.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=TrainingConfig.batch_size, shuffle=False)

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

        model.train() # Set model to training mode
        train_running_loss = 0.0
        for batch_idx, train_batch_data in enumerate(train_loader, 1):
            inputs, sequential_data, targets, original_length = train_batch_data

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

            train_running_loss += loss_regression.item()

            # Print progress every 'print_freq' batches
            print_freq = 1
            if batch_idx % print_freq == 0:
                avg_loss = train_running_loss / print_freq
                print(f"    Batch [{batch_idx}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, val_batch_data in enumerate(val_loader, 1):
                val_inputs, val_sequential_data, val_targets, val_original_length = val_batch_data

                val_output = model(val_inputs, val_sequential_data, val_original_length)

                val_targets = val_targets.view(-1, 1)
                val_loss = criterion(val_output, val_targets)

                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Optionally, you can print or log additional information after each epoch

    print("Training finished somehow!")
