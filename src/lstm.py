import torch
import torch.nn as nn
class LSTMModel(nn.Module):
    def _init_(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self)._init_()
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,  # Input dimension at each time step
            hidden_size=hidden_size,  # Number of hidden units
            num_layers=num_layers,  # Number of LSTM layers
            batch_first=True  # Input format as (batch_size, seq_len, input_size)
        )
        # Define the output layer (fully connected layer)
        self.fc = nn.Linear(hidden_size, output_size)  # Mapping from hidden state to output
    def forward(self, x):
        # Pass input sequence through LSTM layers
        lstm_out, (hn, cn) = self.lstm(x)
        # Use the last hidden state as the output and pass it through the fully connected layer
        out = self.fc(lstm_out[:, -1, :])
        return out