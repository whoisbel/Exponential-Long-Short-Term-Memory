import torch
import torch.nn as nn


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



class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 cell_activation=torch.tanh,
                 hidden_activation=torch.tanh):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Allow per-layer activations (can be a list or same for all layers)
        if not isinstance(cell_activation, list):
            cell_activation = [cell_activation] * num_layers
        if not isinstance(hidden_activation, list):
            hidden_activation = [hidden_activation] * num_layers

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.cells.append(
                CustomLSTMCell(in_size, hidden_size,
                               cell_activation[layer],
                               hidden_activation[layer])
            )

    def forward(self, x, hidden=None):
        """
        x: (seq_len, batch, input_size)
        hidden: tuple of (h_0, c_0), each (num_layers, batch, hidden_size)
        """
        seq_len, batch_size, _ = x.size()

        if hidden is None:
            h_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            h_t, c_t = list(hidden[0]), list(hidden[1])

        outputs = []
        for t in range(seq_len):
            input_t = x[t]
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.cells[layer](input_t, (h_t[layer], c_t[layer]))
                input_t = h_t[layer]  # feed output to next layer
            outputs.append(h_t[-1].unsqueeze(0))  # only last layer output

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_size)
        h_n = torch.stack(h_t)  # (num_layers, batch, hidden_size)
        c_n = torch.stack(c_t)  # (num_layers, batch, hidden_size)

        return outputs, (h_n, c_n)
