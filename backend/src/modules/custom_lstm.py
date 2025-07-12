import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, cell_activation="tanh", hidden_activation="tanh"):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_g = nn.Linear(input_size + hidden_size, hidden_size)

        # Activation functions
        self.cell_activation = self._get_activation(cell_activation)
        self.hidden_activation = self._get_activation(hidden_activation)

    def _get_activation(self, name):
        if name == "tanh":
            return torch.tanh
        elif name == "elu":
            return F.elu
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)

        i = torch.sigmoid(self.W_i(combined))
        f = torch.sigmoid(self.W_f(combined))
        o = torch.sigmoid(self.W_o(combined))
        g = self.cell_activation(self.W_g(combined))

        c = f * c_prev + i * g
        h = o * self.hidden_activation(c)

        return h, c


class LSTMModel(nn.Module):
    """
    3-layer LSTM model with 50 hidden units per layer, dropout of 0.2,
    and a final dense output layer. The first two LSTM layers return sequences,
    and the third returns only the final output.
    """

    def __init__(self, input_size, hidden_size=50, dropout=0.2, activation_fn="tanh"):
        super(LSTMModel, self).__init__()
        
        # Use the first value from hidden_size if it's a list
        hidden_size = hidden_size[0] if isinstance(hidden_size, list) else hidden_size
        
        # First LSTM layer (returns sequences)
        self.lstm1 = CustomLSTMCell(input_size, hidden_size, activation_fn, activation_fn)
        self.dropout1 = nn.Dropout(dropout)

        # Second LSTM layer (returns sequences)
        self.lstm2 = CustomLSTMCell(hidden_size, hidden_size, activation_fn, activation_fn)
        self.dropout2 = nn.Dropout(dropout)

        # Third LSTM layer (does NOT return sequences)
        self.lstm3 = CustomLSTMCell(hidden_size, hidden_size, activation_fn, activation_fn)
        self.dropout3 = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Input shape: [batch, seq_len, input_size] → [seq_len, batch, input_size]
        x = x.permute(1, 0, 2)
        batch_size = x.size(1)

        # Initialize hidden/cell states for all layers
        h1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=x.device)
        c1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=x.device)
        h2 = torch.zeros(batch_size, self.lstm2.hidden_size, device=x.device)
        c2 = torch.zeros(batch_size, self.lstm2.hidden_size, device=x.device)
        h3 = torch.zeros(batch_size, self.lstm3.hidden_size, device=x.device)
        c3 = torch.zeros(batch_size, self.lstm3.hidden_size, device=x.device)

        # First LSTM (returns full sequence)
        seq_out1 = []
        for t in range(x.size(0)):
            h1, c1 = self.lstm1(x[t], (h1, c1))
            seq_out1.append(h1)
        seq_out1 = torch.stack(seq_out1, dim=0)
        seq_out1 = self.dropout1(seq_out1)

        # Second LSTM (returns full sequence)
        seq_out2 = []
        for t in range(seq_out1.size(0)):
            h2, c2 = self.lstm2(seq_out1[t], (h2, c2))
            seq_out2.append(h2)
        seq_out2 = torch.stack(seq_out2, dim=0)
        seq_out2 = self.dropout2(seq_out2)

        # Third LSTM (returns only last timestep)
        for t in range(seq_out2.size(0)):
            h3, c3 = self.lstm3(seq_out2[t], (h3, c3))
        h3 = self.dropout3(h3)

        return self.fc(h3)
