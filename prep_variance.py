import numpy as np 
from IS import calculate_importance_weights
import torch

def variance_terms_tens(eval_policy, behav_policy, behavior_policies):
  # Initialize lists to store axis data for each policy
  t = []
  s = []
  s_last = []
  a = []
  r = []
  g_last = []
  w_last = []
  w = calculate_importance_weights(eval_policy, behav_policy, behavior_policies)
  psi = []

  for index, policy in enumerate(behavior_policies):
      policy_array = np.array(policy)
      t.append(policy_array[:, 4].astype(int))
      # s.append(policy_array[:, 0])

      # last timestep for gamma
      g_last.append(len(policy))
      # last importance weight
      w_last.append(w[index][-1])


      s.append(policy_array[:, 0][1:])
      psi.append(policy_array[:,5][1:])
      s_last.append(policy_array[:,0][-1])
      a.append(policy_array[:, 1])
      r.append(policy_array[:, 2].astype(float))


  s_w_diff = []
  for index, weight in enumerate(w):
    # diff = np.array(w[index][:-1]) - np.array(w[index][1:])
    diff = np.array(weight[:-1]) - np.array(weight[1:])
    s_w_diff.append(diff)

  gtrw = np.power(0.9,t)*r*np.array(w)
  gw_l = np.power(0.9, g_last)*w_last
  # Number of bootstrap iterations
  num_iterations = 1000

  np.random.seed(0)
  # Get bootstrap indices
  bootstrap_indices = np.random.choice(len(behavior_policies), size=(num_iterations, len(behavior_policies)), replace=True)

  gtrw = np.power(0.9,t)*r*np.array(w)
  samples_IS = np.take(gtrw, bootstrap_indices, axis = 0)
  samples_s = np.take(s, bootstrap_indices, axis = 0)
  samples_w_diff = np.take(s_w_diff, bootstrap_indices, axis = 0)
  samples_last = np.take(gw_l, bootstrap_indices, axis = 0)
  samples_last_states = np.take(s_last, bootstrap_indices, axis = 0)

  IS_all = np.array([np.sum(np.concatenate(arr), axis=0)/len(behavior_policies) for arr in samples_IS])
  F_all = np.array([np.sum((arr), axis=0)/len(behavior_policies) for arr in samples_last])

  ft = torch.tensor(F_all).reshape(-1,1)
  f_res = [tensor for tensor in ft]
  last_first_tens_arr = np.array(f_res)

  It = torch.tensor(IS_all).reshape(-1,1)
  IS_res = [tensor for tensor in It]
  IS_tens_arr = np.array(IS_res)

  ss = [np.concatenate(p_set) for p_set in samples_s]
  state_tensors = [
    [torch.tensor(sample, dtype=torch.float32) for sample in sublist]
    for sublist in ss]

  # sample_weights = [torch.tensor(np.concatenate(p_set)).reshape(-1,1) for p_set in samples_w_diff]

  # Initialize an empty array to store the tensors
  w_diff_tensors = np.empty((len(samples_w_diff),), dtype=object)

  for i, p_set in enumerate(samples_w_diff):
      # Concatenate the NumPy arrays and reshape the resulting tensor
      ssw_tensor = torch.tensor(np.concatenate(p_set)).reshape(-1, 1)

      # Store the tensor directly into w_diff_tensors
      w_diff_tensors[i] = ssw_tensor


  state_tensor_og = [[torch.tensor(state_t, dtype = torch.float32) for state_t in traj_t] for traj_t in s]

  psi_og = [[torch.tensor(state, dtype = torch.float32) for state in traj] for traj in psi]
  psi_arrays = np.empty((len(psi_og),), dtype=object)
  # Stack the tensors in each sublist along a new dimension (assuming each sublist has the same number of tensors)
  stacked_psi = [torch.stack(sublist, dim=0) for sublist in psi_og]

  for i, tensor in enumerate(stacked_psi):
      # psi = net(tensor)  # Process each tensor separately
      # Store the psi tensor directly into psi_arrays
      psi_arrays[i] = tensor
  reshaped_psi = [tensor.unsqueeze(1) for tensor in psi_arrays]


  sample_last_tensors = [
  [torch.tensor(sample, dtype=torch.float32) for sample in sublist]
  for sublist in samples_last_states]

  return IS_tens_arr, state_tensors , w_diff_tensors, last_first_tens_arr, state_tensor_og, reshaped_psi, sample_last_tensors

  # return IS_all, samples_s, samples_w_diff, F_all





def calc_variance_t(IS, state_tensors, w_diff, f, sample_last_tensors, feature_net, behavior_policies):

  output_arrays = np.empty((len(state_tensors),), dtype=object)
  # Stack the tensors in each sublist along a new dimension (assuming each sublist has the same number of tensors)
  stacked_samples = [torch.stack(sublist, dim=0) for sublist in state_tensors]

  for i, tensor in enumerate(stacked_samples):
      output = feature_net(tensor)  # Process each tensor separately

      # Store the output tensor directly into output_arrays
      output_arrays[i] = output

  phi_w_diff_arrays = output_arrays*w_diff
  # Assuming phi_w_diff_arrays is a list of PyTorch tensors
  phi_diff_sums_array = np.empty(len(phi_w_diff_arrays), dtype=object)

  for i, p_set in enumerate(phi_w_diff_arrays):
      phi_diff_sum = torch.sum(p_set, dim=0) / len(behavior_policies)
      phi_diff_sums_array[i] = phi_diff_sum


  output_arrays_last = np.empty((len(sample_last_tensors),), dtype=object)
  # Stack the tensors in each sublist along a new dimension (assuming each sublist has the same number of tensors)
  stacked_samples_last = [torch.stack(sublist, dim=0) for sublist in sample_last_tensors]

  for i, tensor in enumerate(stacked_samples_last):
      output = feature_net(tensor)  # Process each tensor separately

      # Store the output tensor directly into output_arrays_last
      output_arrays_last[i] = output

  l_f = output_arrays_last*f

  # FIX
  state_0 = feature_net(torch.tensor((0,0), dtype = torch.float32))
  # state_end = feature_net(torch.tensor((4,4), dtype = torch.float32))
  f_all = phi_diff_sums_array + (l_f - state_0.item())


  IS_phi_terms = IS*f_all
  IS_sq = torch.mean(torch.stack(list(IS**2)), dim = 0)
  IS_sq_all = torch.mean(torch.stack(list(IS)), dim = 0)**2
  IS_phi_l_f = torch.mean(torch.stack(list(IS_phi_terms)), dim = 0)
  IS_and_phi_l_f = torch.mean(torch.stack(list(IS)), dim = 0)*torch.mean(torch.stack(list(f_all)), dim = 0)
  phi_w_sq = torch.mean(torch.stack(list(phi_diff_sums_array**2)), dim = 0)
  phi_w_sq_all = torch.mean(torch.stack(list(phi_diff_sums_array)), dim = 0)**2

  var_IS = IS_sq - IS_sq_all
  var_scope = IS_sq + 2*IS_phi_l_f + phi_w_sq - IS_sq_all - 2*IS_and_phi_l_f - phi_w_sq_all

  # return mse_loss, var_scope
  return var_scope
  # return var_IS, var_scope

