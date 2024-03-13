class ModelConfig:
    input_size = 21  # Dimensionen på input for hvert tidsinterval (hver eneste datapoint).

    # TODO: hidden_size skal findes med grid search CV, men kan ikke se noget om den i paperet?
    hidden_size = 16
    # only when not using grid search cv
    criterion = "friedman_mse"


class TrainingConfig:
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.1


class GridSearchConfig:
    param_grid = {
        'criterion': ['mse', 'friedman_mse', 'mae'],
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

