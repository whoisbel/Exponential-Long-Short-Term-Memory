import torch
import torch.nn as nn

class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, custom_activation):
        super(CustomLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.custom_activation = custom_activation

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim, hidden_dim)

        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_c = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, inputs, states):
        prev_output, prev_state = states

        i = torch.sigmoid(self.W_i(inputs) + self.U_i(prev_output) + self.b_i)
        f = torch.sigmoid(self.W_f(inputs) + self.U_f(prev_output) + self.b_f)
        o = torch.sigmoid(self.W_o(inputs) + self.U_o(prev_output) + self.b_o)
        c = self.custom_activation(
            torch.clamp(self.W_c(inputs) + self.U_c(prev_output) + self.b_c, -10, 10)
        )

        new_state = f * prev_state + i * c
        output = o * self.custom_activation(new_state)

        return output, (output, new_state)

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, custom_activation, dropout=0.3):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.custom_activation = custom_activation
        
        self.lstm_cell = CustomLSTMCell(input_size, hidden_size, custom_activation)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        for t in range(seq_len):
            h, (h, c) = self.lstm_cell(x[:, t, :], (h, c))
            h = self.dropout(h)
        
        out = self.fc(h)
        return out
