import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

import prep_variance
from prep_variance import calc_variance_t

def mse(feature_net, state_tensor_og, psi_reshaped):

  state_og_arrays = np.empty((len(state_tensor_og),), dtype=object)
  # Stack the tensors in each sublist along a new dimension (assuming each sublist has the same number of tensors)
  stacked_og = [torch.stack(sublist, dim=0) for sublist in state_tensor_og]

  for i, tensor in enumerate(stacked_og):
      output_og = feature_net(tensor)  # Process each tensor separately

      # Store the output tensor directly into state_og_arrays
      state_og_arrays[i] = output_og
  # Initialize an empty array to store the outputs

  # Calculate the mean squared error (MSE) loss
  mse_loss = nn.MSELoss()

  # Calculate the loss for each pair of tensors and then take the mean
  loss = torch.mean(torch.stack([mse_loss(output, target) for output, target in zip(state_og_arrays, psi_reshaped)]))

  mse_loss = loss #/len(np.concatenate(psi_reshaped))#len(behavior_policies)

  return mse_loss


def train_mse_var(net, num_epochs, learning_rate, mse_ratio, var_ratio, IS, state_tensors, w_diff, f,sample_last_tensors, phi_set, st_og, psi_res):
  net.train()
  # Define the optimizer (Adam optimizer)
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)

  # Training loop
  for epoch in range(num_epochs):
    total_loss = 0
    variance_loss = calc_variance_t(IS, state_tensors, w_diff, f,sample_last_tensors, net, phi_set)
    mse_loss = mse(net, st_og, psi_res)/len(st_og)
    print(f"Epoch {epoch+1}")
    print("Var loss: ", variance_loss)
    print("Feature loss (MSE): ", mse_loss)
    tot = mse_ratio*mse_loss + var_ratio*variance_loss
    # tot = variance_loss
    # tot = mse_loss


    # Backpropagation and optimization for the trajectory
    optimizer.zero_grad()
    tot.backward()
    optimizer.step()

    # print("Total Loss: ", tot)

    total_loss += tot.item()

    print(f"Total Loss: {total_loss}")
    print("-" * 40)

  # Print the weights of the neural network
  for name, param in net.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}")
        print(f"Weights: {param.data}")
  # results = SCOPEnet(scope_set, 30000, eval_policy, behav_policy, net)
  # print("scope_results: ", results)

  return net

