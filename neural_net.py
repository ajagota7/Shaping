import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CustomizableFeatureNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob=0.2, dtype=torch.float32):
        super(CustomizableFeatureNet, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # Create the hidden layers based on the provided sizes
        for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims):
            self.hidden_layers.append(nn.Linear(in_dim, out_dim).to(dtype))

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim).to(dtype)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
