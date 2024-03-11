import torch
import torch.nn as nn

class ModelClass(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelClass, self).__init__()

        self.embedding_layer = nn.Embedding(input_size, hidden_size)
        self.rnn_layer = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        embedded = self.embedding_layer(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.rnn_layer(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.fc_layer(output[:, -1, :])
        return output
