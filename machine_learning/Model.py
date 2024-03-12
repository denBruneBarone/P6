import torch
import torch.nn as nn


class ModelClass(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelClass, self).__init__()

        self.rnn_layer = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_layer = nn.Linear(hidden_size * 2, 1)  # Adjusted input size for concatenation


    def forward(self, inputs, sequential_data, original_length_tensor):
        # Pack the embedded_seq for variable-length sequences
        packed_seq = nn.utils.rnn.pack_padded_sequence(sequential_data, original_length_tensor, batch_first=True, enforce_sorted=False)

        # Pass the packed_seq through the LSTM layers
        output, _ = self.rnn_layer(packed_seq)

        max_sequence_length = max(original_length_tensor)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=int(max_sequence_length))


        tensor1 = torch.unsqueeze(inputs, 2)#torch.Size([32, 1, 1])
        tensor2 = output #torch.Size([32, x, hidden_size])

        tensor1_expanded = tensor1.expand(32, output.size(1), output.size(2)) ##torch.Size([32, x, hidden_size])
        # tensor1_expanded = tensor1.expand(32, output.size(1), 1) ##torch.Size([32, x, 1])

        combined_input = torch.cat((output, tensor1_expanded), dim=2) #torch.Size([32, x, hidden_size*2]) eller #torch.Size([32, x, hidden_size+1])
        #TODO: test bedste tensor configuration. [32, x hidden_size*2] vs [32, x hidden_size+1]


        # Pass the combined input through the final linear layer
        final_output = self.fc_layer(combined_input[:, -1, :])

        return final_output