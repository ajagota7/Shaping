import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# class CustomizableFeatureNet(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob=0.2, dtype=torch.float32):
#         super(CustomizableFeatureNet, self).__init__()
#         self.hidden_layers = nn.ModuleList()

#         # Create the hidden layers based on the provided sizes
#         for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims):
#             self.hidden_layers.append(nn.Linear(in_dim, out_dim).to(dtype))

#         self.output_layer = nn.Linear(hidden_dims[-1], output_dim).to(dtype)
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, x):
#         for layer in self.hidden_layers:
#             x = F.relu(layer(x))
#             x = self.dropout(x)
#         x = self.output_layer(x)
#         return x



class CustomizableFeatureNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob=0.2, l2_lambda=0.01, dtype=torch.float32):
        super(CustomizableFeatureNet, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # Create the hidden layers based on the provided sizes
        for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims):
            layer = nn.Linear(in_dim, out_dim).to(dtype)
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim).to(dtype)
        self.dropout = nn.Dropout(dropout_prob)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

    def l2_regularization(self):
        l2_reg = torch.tensor(0., device=self.output_layer.weight.device)
        for layer in self.hidden_layers:
            l2_reg += torch.norm(layer.weight)
        l2_reg += torch.norm(self.output_layer.weight)
        return self.l2_lambda * l2_reg


class NN_l1_l2_reg(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob=0.2, l1_lambda=0.01, l2_lambda=0.01, dtype=torch.float32):
        super(NN_l1_l2_reg, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # Create the hidden layers based on the provided sizes
        for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims):
            layer = nn.Linear(in_dim, out_dim).to(dtype)
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim).to(dtype)
        self.dropout = nn.Dropout(dropout_prob)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

    def l1_regularization(self):
        l1_reg = torch.tensor(0., device=self.output_layer.weight.device)
        for layer in self.hidden_layers:
            l1_reg += torch.norm(layer.weight, p=1)
        l1_reg += torch.norm(self.output_layer.weight, p=1)
        return self.l1_lambda * l1_reg

    def l2_regularization(self):
        l2_reg = torch.tensor(0., device=self.output_layer.weight.device)
        for layer in self.hidden_layers:
            l2_reg += torch.norm(layer.weight, p=2)
        l2_reg += torch.norm(self.output_layer.weight, p=2)
        return self.l2_lambda * l2_reg

    def total_regularization(self):
        return self.l1_regularization() + self.l2_regularization()