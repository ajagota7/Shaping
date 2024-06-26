# import numpy as np 
# from IS import calculate_importance_weights

# import torch 

# class SCOPE_variance(object):
#     def __init__(self, model, gamma, num_bootstraps, pi_b, P_pi_b, P_pi_e, dtype):
#         self.model = model
#         self.gamma = gamma
#         self.num_bootstraps = num_bootstraps
#         self.pi_b = pi_b
#         self.P_pi_b = P_pi_b
#         self.P_pi_e = P_pi_e
#         self.dtype = dtype


#     def prep_policies(self):
#         # Initialize lists to store axis data for each policy
#         timesteps = []
#         states = []
#         state_first = []
#         state_last = []
#         actions = []
#         rewards = []
#         gamma_last = []
#         weight_last = []
#         weight_first = []
#         # all_weights_temp, weights = calculate_importance_weights(self.P_pi_e, self.P_pi_b, self.pi_b)
#         weights = calculate_importance_weights(self.P_pi_e, self.P_pi_b, self.pi_b)
#         psi = []

#         for index, policy in enumerate(self.pi_b):
#             policy_array = np.array(policy)
#             timesteps.append(policy_array['timestep'].astype(int))
#             # s.append(policy_array[:, 0])

#             # last timestep for gamma
#             gamma_last.append(len(policy))
#             # last importance weight
#             weight_last.append(weights[index][-1])
#             weight_first.append(weights[index][0])


#             states.append(policy_array['state'][1:])
#             psi.append(policy_array['psi'][1:])
#             state_first.append(policy_array['state'][0])
#             state_last.append(policy_array['state'][-1])
#             actions.append(policy_array['action'])
#             rewards.append(policy_array['reward'].astype(float))

#         weights_difference = []
#         for index, weight in enumerate(weights):
#             # diff = np.array(w[index][:-1]) - np.array(w[index][1:])
#             diff = np.array(weight[:-1]) - np.array(weight[1:])
#             weights_difference.append(diff)

#         return timesteps, states, state_first, state_last, actions, rewards, gamma_last, weight_last, weights, weights_difference, weight_first, psi

#     def padding_IS_terms(self,timesteps, actions, rewards, weights):
#         # Find the maximum length among all lists
#         max_length = max(len(traj) for traj in timesteps)

#         # Define the padding values
#         zero_padding = 0

#         # Pad each list to match the maximum length
#         padded_timesteps = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in timesteps]
#         padded_rewards = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in rewards]
#         padded_actions = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in actions]
#         padded_weights = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in weights]

#         return padded_timesteps, padded_rewards, padded_actions, padded_weights

#     def padding_states_weights_difference(self, states, weights_difference, psi):
#         # Find the maximum length of trajectories
#         max_length = max(len(trajectory) for trajectory in states)

#         zero_padding = 0

#         # Pad each trajectory to make them all the same length
#         padded_states = [
#             [list(item) for item in trajectory] + [[0, 0]] * (max_length - len(trajectory))
#             for trajectory in states
#         ]

#         padded_weights_difference = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in weights_difference]

#         padded_psi = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in psi]
#         return padded_states, padded_weights_difference, padded_psi
    
#     def tensorize_padded_terms(self, padded_states, padded_weights_difference, padded_psi):
#         padded_state_tensors = torch.tensor(padded_states, dtype = self.dtype)
#         padded_weight_diff_tensors = torch.tensor(padded_weights_difference, dtype = self.dtype)
#         padded_weight_diff_tensors = padded_weight_diff_tensors.unsqueeze(-1)
#         padded_psi_tensors = torch.tensor(padded_psi, dtype = self.dtype)
#         padded_psi_tensors = padded_psi_tensors.unsqueeze(-1)

#         return padded_state_tensors, padded_weight_diff_tensors, padded_psi_tensors

#     def tensorize_last_and_first_terms(self, states_first, states_last, gamma_last, weights_last):
#         states_first_tensor = torch.tensor(states_first, dtype = self.dtype)
#         states_last_tensor = torch.tensor(states_last, dtype = self.dtype)
#         gamma_last_tensor = torch.tensor(gamma_last, dtype = self.dtype)
#         weights_last_tensor = torch.tensor(weights_last, dtype = self.dtype)

#         return states_first_tensor, states_last_tensor, gamma_last_tensor, weights_last_tensor

#     def calc_IS_terms(self, gamma, timesteps, rewards, weights):
#         gtrw = np.power(gamma, timesteps)*rewards*weights

#         IS_tensor = torch.sum(torch.tensor(gtrw, dtype = self.dtype), dim = 1, keepdim = True)

#         return IS_tensor

#     def calc_gamma_weight_last(self, gamma, gamma_last, weights_last):
#         gamma_weight_last = np.power(gamma, gamma_last)*weights_last

#         gamma_weight_last_tensor = torch.tensor(gamma_weight_last, dtype = self.dtype).unsqueeze(-1)

#         return gamma_weight_last_tensor

#     def tensorize_weight_first(self, weights_first):
      
#       weight_first_tensor = torch.tensor(weights_first, dtype = torch.float64).unsqueeze(-1) 

#       return weight_first_tensor        
#     def bootstrap_IS_terms(self, IS_tensor, num_samples):
#         seed = 42
#         torch.manual_seed(seed)

#         num_bootstraps = num_samples*len(IS_tensor)

#         # Sample indices with replacement
#         sampled_indices = torch.randint(0, len(IS_tensor), size=(num_bootstraps,), dtype=torch.long)

#         # new_size = (num_samples, IS_tensor.shape[0], IS_tensor.shape[1])
#         new_size = (num_samples, IS_tensor.shape[0])

#         IS_bootstraps = IS_tensor[sampled_indices].view(new_size)

#         # sampled_tensor = IS_bootstraps.view(new_size)

#         return IS_bootstraps

#     def states_weight_diff_sums(self, states_output, padded_weight_diff_tensors):
#         states_weight_diff = states_output * padded_weight_diff_tensors
#         sums_states_weight_diff = torch.sum(states_weight_diff, dim =1)

#         return sums_states_weight_diff

#     def last_first_terms_operations(self, gamma_weights_last_tensor, states_last_output, states_first_output, weight_first_tensor):
#         gamma_weights_states_last_sub_states_first = gamma_weights_last_tensor*states_last_output -  weight_first_tensor*states_first_output

#         return gamma_weights_states_last_sub_states_first
      

#     def bootstrap_shaping_terms(self, sums_states_weight_diff, gamma_weights_states_last_sub_states_first, IS_tensor):

#         seed = 42
#         torch.manual_seed(seed)

#         num_samples = self.num_bootstraps

#         num_bootstraps_lin = num_samples*sums_states_weight_diff.shape[0]

#         # Sample indices with replacement
#         sampled_indices = torch.randint(0, len(sums_states_weight_diff), size=(num_bootstraps_lin,), dtype=torch.long)

#         reshaped_size = (num_samples, sums_states_weight_diff.shape[0])

#         # Resize samples to shape num_samples x num_trajectories
#         sample_sums_states_weight_diff = sums_states_weight_diff[sampled_indices].view(reshaped_size)
#         samples_gamma_weight_states_last_sub_states_first = gamma_weights_states_last_sub_states_first[sampled_indices].view(reshaped_size)

#         # Sum states_weight_diff and gamma_weights-states_last_sub_states_first
#         sum_terms = sums_states_weight_diff + gamma_weights_states_last_sub_states_first

#         # sample IS terms

#         IS_SCOPE = IS_tensor * sum_terms

#         samples_IS_SCOPE = IS_SCOPE[sampled_indices].view(reshaped_size)


#         sample_all_shaping = sum_terms[sampled_indices].view(reshaped_size)

#         return sample_sums_states_weight_diff, samples_gamma_weight_states_last_sub_states_first, sample_all_shaping, samples_IS_SCOPE

#     def bootstrap_all_terms(self, sums_states_weight_diff, gamma_weights_states_last_sub_states_first, IS_tensor):
#         seed = 42
#         torch.manual_seed(seed)

#         # num_bootstraps = num_samples*len(IS_tensor)

#         # Sample indices with replacement
#         # sampled_indices = torch.randint(0, len(IS_tensor), size=(num_bootstraps,), dtype=torch.long)

#         # new_size = (num_samples, IS_tensor.shape[0], IS_tensor.shape[1])
#         # new_size = (num_samples, IS_tensor.shape[0])

#         # IS_bootstraps = IS_tensor[sampled_indices].view(new_size)

#         num_samples = self.num_bootstraps
#         num_bootstraps_lin = num_samples*sums_states_weight_diff.shape[0]


#         # Sample indices with replacement
#         sampled_indices = torch.randint(0, len(sums_states_weight_diff), size=(num_bootstraps_lin,), dtype=torch.long)

#         reshaped_size = (num_samples, sums_states_weight_diff.shape[0])

#         IS_bootstraps = IS_tensor[sampled_indices].view(reshaped_size)

#         # Resize samples to shape num_samples x num_trajectories
#         sample_sums_states_weight_diff = sums_states_weight_diff[sampled_indices].view(reshaped_size)
#         samples_gamma_weight_states_last_sub_states_first = gamma_weights_states_last_sub_states_first[sampled_indices].view(reshaped_size)

#         # Sum states_weight_diff and gamma_weights-states_last_sub_states_first
#         sum_terms = sums_states_weight_diff + gamma_weights_states_last_sub_states_first

#         # sample IS terms

#         IS_SCOPE = IS_tensor * sum_terms

#         samples_IS_SCOPE = IS_SCOPE[sampled_indices].view(reshaped_size)

#         sample_all_shaping = sum_terms[sampled_indices].view(reshaped_size)

#         return IS_bootstraps, sample_sums_states_weight_diff, samples_gamma_weight_states_last_sub_states_first, sample_all_shaping, samples_IS_SCOPE



#     def pass_states(self,model, padded_state_tensors, states_first_tensor, states_last_tensor):
#         # Get model outputs for states
#         states_output = model(padded_state_tensors)
#         states_first_output = model(states_first_tensor)
#         states_last_output = model(states_last_tensor)
#         return states_output, states_first_output, states_last_output


#     # def prepare(self):
#     #     timesteps, states, states_first, states_last, actions, rewards, gamma_last, weights_last, weights, weights_difference = self.prep_policies()
#     #     padded_timesteps, padded_rewards, padded_actions, padded_weights = self.padding_IS_terms(timesteps, actions, rewards, weights)
#     #     padded_states, padded_weights_difference = self.padding_states_weights_difference(states, weights_difference)
#     #     padded_state_tensors, padded_weight_diff_tensors = self.tensorize_padded_terms(padded_states, padded_weights_difference)
#     #     states_first_tensor, states_last_tensor, gamma_last_tensor, weights_last_tensor = self.tensorize_last_and_first_terms(states_first, states_last, gamma_last, weights_last)
#     #     IS_tensor = self.calc_IS_terms(self.gamma, padded_timesteps, padded_rewards, padded_weights)
#     #     gamma_weights_last_tensor = self.calc_gamma_weight_last(self.gamma, gamma_last, weights_last)
#     #     samples_IS = self.bootstrap_IS_terms(IS_tensor, self.num_bootstraps)

#     #     return IS_tensor, samples_IS, padded_state_tensors, padded_weight_diff_tensors, gamma_weights_last_tensor, states_first_tensor, states_last_tensor


#     def prepare(self):
#         timesteps, states, states_first, states_last, actions, rewards, gamma_last, weights_last, weights, weights_difference, weight_first = self.prep_policies()
#         padded_timesteps, padded_rewards, padded_actions, padded_weights = self.padding_IS_terms(timesteps, actions, rewards, weights)
#         padded_states, padded_weights_difference = self.padding_states_weights_difference(states, weights_difference)
#         padded_state_tensors, padded_weight_diff_tensors = self.tensorize_padded_terms(padded_states, padded_weights_difference)
#         states_first_tensor, states_last_tensor, gamma_last_tensor, weights_last_tensor = self.tensorize_last_and_first_terms(states_first, states_last, gamma_last, weights_last)
#         IS_tensor = self.calc_IS_terms(self.gamma, padded_timesteps, padded_rewards, padded_weights)
#         gamma_weights_last_tensor = self.calc_gamma_weight_last(self.gamma, gamma_last, weights_last)
#         weight_first_tensor = self.tensorize_weight_first(weight_first)
#         # samples_IS = self.bootstrap_IS_terms(IS_tensor, self.num_bootstraps)

#         return IS_tensor, padded_state_tensors, padded_weight_diff_tensors, gamma_weights_last_tensor, states_first_tensor, states_last_tensor, weight_first_tensor
    


    

    



    

