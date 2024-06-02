import numpy as np 
from IS import calculate_importance_weights

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import copy

class SCOPE_straight(object):

  def __init__(self, model, gamma, num_bootstraps, pi_b, P_pi_b, pi_e, P_pi_e, percent_to_estimate_phi, shaping_function, env, dtype):
        self.model = model
        self.gamma = gamma
        self.num_bootstraps = num_bootstraps
        self.pi_b = pi_b
        self.P_pi_b = P_pi_b
        self.P_pi_e = P_pi_e
        self.pi_e = pi_e
        self.shaping_function = shaping_function
        self.env = env
        self.dtype = dtype

        self.percent_to_estimate_phi = percent_to_estimate_phi
        # self.num_epochs = num_epochs

  def subset_policies(self):
    # seed_value = 0
    # np.random.seed(seed_value)
    num_policies = len(self.pi_b)
    num_policies_to_estimate_phi = int(num_policies * self.percent_to_estimate_phi)

    policies_for_scope = self.pi_b[num_policies_to_estimate_phi:]
    policies_for_phi = self.pi_b[:num_policies_to_estimate_phi]

    return policies_for_phi, policies_for_scope


  # ---------------
  # Pre-processing
  # ---------------

  def prep_policies(self, chosen_policies):
      # Initialize lists to store axis data for each policy
      timesteps = []
      # states = []
      # state_first = []
      # state_last = []
      actions = []
      rewards = []
      # gamma_last = []
      # weight_last = []
      # weight_first = []
      # all_weights_temp, weights = calculate_importance_weights(P_pi_e, P_pi_b, pi_b)
      weights = calculate_importance_weights(self.P_pi_e, self.P_pi_b, chosen_policies)
      psi = []

      states_current = []
      states_next = []
      states_all = []

      states_last = []
      psi_last = []

      for index, policy in enumerate(chosen_policies):
          policy_array = np.array(policy)

          timesteps.append(policy_array['timestep'].astype(int))
          actions.append(policy_array['action'])
          rewards.append(policy_array['reward'].astype(float))

          state_last = policy_array['state_next'][-1]
          # last_psi = smallest_distance_to_deadend(state_last, env)
          last_psi = self.shaping_function(state_last, self.env)
          states_last.append(state_last)
          psi_last.append(last_psi)

          # Concatenate psi array with last_psi
          # all_psi = np.concatenate((policy_array['psi'], [last_psi]))
          # psi.append(all_psi)
          psi.append(policy_array['psi'])

          states_next.append(policy_array['state_next'])
          states_current.append(policy_array['state'])
          # all_states = policy_array['state'] + policy_array['state_next'][-1]
          all_states = np.vstack((policy_array['state'],policy_array['state_next'][-1]))
          states_all.append(all_states)

          # states_all.append(np.concatenate((policy_array['state'], policy_array['state_next'][-1])))



      return timesteps, rewards, states_next, states_current, weights, actions, psi, states_last, psi_last

  def padding_IS_terms(self, timesteps, actions, rewards, weights):

    # Find the maximum length among all lists
    max_length = max(len(traj) for traj in timesteps)

    # Define the padding values
    zero_padding = 0

    # Pad each list to match the maximum length
    padded_timesteps = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in timesteps]
    padded_rewards = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in rewards]
    padded_actions = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in actions]
    padded_weights = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in weights]

    return padded_timesteps, padded_rewards, padded_actions, padded_weights


  def tensorize_IS_terms(self, padded_timesteps, padded_rewards, padded_weights):

    padded_timestep_tensors = torch.tensor(padded_timesteps, dtype = self.dtype)
    padded_reward_tensors = torch.tensor(padded_rewards, dtype = self.dtype)
    padded_weight_tensors = torch.tensor(padded_weights, dtype = self.dtype)

    return padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors

  def padding_states_all(self, states_all, psi):
    max_length = max(len(trajectory) for trajectory in states_all)

    zero_padding = 0

    # Pad each trajectory to make them all the same length
    padded_states_all = [
        [list(item) for item in trajectory] + [[0, 0]] * (max_length - len(trajectory))
        for trajectory in states_all
    ]

    padded_psi = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in psi]
    mask = [[1] * len(trajectory) + [0] * (max_length - len(trajectory)) for trajectory in states_all]

    return padded_states_all, padded_psi, mask



  def padding_states(self, states_next, states_current, psi):
    # Find the maximum length of trajectories
    max_length = max(len(trajectory) for trajectory in states_current)

    zero_padding = 0

    # Pad each trajectory to make them all the same length
    padded_states_next = [
        [list(item) for item in trajectory] + [[0, 0]] * (max_length - len(trajectory))
        for trajectory in states_next
    ]

    # Pad each trajectory to make them all the same length
    padded_states_current = [
        [list(item) for item in trajectory] + [[0, 0]] * (max_length - len(trajectory))
        for trajectory in states_current
    ]

    padded_psi = [np.concatenate([traj, [zero_padding] * (max_length - len(traj))]) for traj in psi]

    # Create mask
    mask = [[1] * len(trajectory) + [0] * (max_length - len(trajectory)) for trajectory in states_current]

    return padded_states_next, padded_states_current, padded_psi, mask


  def tensorize_padded_terms(self, padded_states_next, padded_states_current, padded_psi,mask):
    padded_states_next_tensors = torch.tensor(padded_states_next, dtype = self.dtype)
    padded_states_current_tensors = torch.tensor(padded_states_current, dtype = self.dtype)
    padded_psi_tensors = torch.tensor(padded_psi, dtype = self.dtype)

    mask_tensor = torch.tensor(mask, dtype = self.dtype)
    return padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor

  def tensorize_all_states_psi(self, padded_states_all, padded_psi, mask):
    padded_states_all_tensors = torch.tensor(padded_states_all, dtype = self.dtype)
    padded_psi_tensors = torch.tensor(padded_psi, dtype = self.dtype)
    mask_tensor = torch.tensor(mask, dtype = self.dtype)

    return padded_states_all_tensors, padded_psi_tensors, mask_tensor

  def tensorize_last_states_psi(self, states_last, psi_last):
    states_last_tensor = torch.tensor(states_last, dtype = self.dtype)
    psi_last_tensor = torch.tensor(psi_last, dtype = self.dtype)

    return states_last_tensor, psi_last_tensor

  #-----------------------
  # Preparation Functions
  # ----------------------

  def prepare_IS(self):
    timesteps, rewards, states_next, states_current, weights, actions,_,_,_ = self.prep_policies(self.pi_b)
    padded_timesteps, padded_rewards, padded_actions, padded_weights = self.padding_IS_terms(timesteps, actions, rewards, weights)
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors = self.tensorize_IS_terms(padded_timesteps, padded_rewards, padded_weights)

    return padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors

  def prepare_SCOPE(self, policies):
    timesteps, rewards, states_next, states_current, weights, actions, psi,states_last, psi_last = self.prep_policies(policies)
    padded_timesteps, padded_rewards, padded_actions, padded_weights = self.padding_IS_terms(timesteps, actions, rewards, weights)
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors = self.tensorize_IS_terms(padded_timesteps, padded_rewards, padded_weights)
    padded_states_next, padded_states_current, padded_psi, mask = self.padding_states(states_next, states_current, psi)
    padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor = self.tensorize_padded_terms(padded_states_next, padded_states_current, padded_psi, mask)
    states_last_tensor, psi_last_tensor = self.tensorize_last_states_psi(states_last, psi_last)
    return padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor, states_last_tensor, psi_last_tensor

  def prepare_SCOPE_phi(self):
    phi_set,_ = self.subset_policies()
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor, states_last_tensor, psi_last_tensor = self.prepare_SCOPE(phi_set)

    return padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor, states_last_tensor, psi_last_tensor

  def prepare_SCOPE_test(self):
    _, scope_set = self.subset_policies()
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors,_,_,_,_ = self.prepare_SCOPE(scope_set)

    return padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors


  # ----------------
  # IS Calculations
  # ----------------


  def bootstrap_IS(self, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, multi = None):
    
    if multi is None:
      seed = 42
      torch.manual_seed(seed)

    num_samples = self.num_bootstraps
    num_bootstraps_lin = num_samples*padded_timestep_tensors.shape[0]

    # Sample indices with replacement
    sampled_indices = torch.randint(0, len(padded_timestep_tensors), size=(num_bootstraps_lin,), dtype=torch.long)

    reshaped_size = (num_samples, padded_timestep_tensors.shape[0], padded_timestep_tensors.shape[1])

    padded_IS = self.gamma**(padded_timestep_tensors)*padded_weight_tensors*padded_reward_tensors

    IS_bootstraps = padded_IS[sampled_indices].view(reshaped_size)

    # timestep_bootstraps = padded_timestep_tensors[sampled_indices].view(reshaped_size)
    # rewards_bootstraps = padded_reward_tensors[sampled_indices].view(reshaped_size)
    # weights_bootstraps = padded_weight_tensors[sampled_indices].view(reshaped_size)
    # return timestep_bootstraps, rewards_bootstraps, weights_bootstraps, IS_bootstraps
    return IS_bootstraps


  def calc_var_IS(self, IS_bootstraps):
    # Step 1: Sum along the third dimension
    sum_IS_trajectories = torch.sum(IS_bootstraps, dim=2)  # Shape: [1000, 1000]

    # Step 2: Take the mean along the second dimension
    mean_IS_sum = torch.mean(sum_IS_trajectories, dim=1)  # Shape: [1000]

    # Step 3: Calculate the variance across the first dimension
    IS_variance = torch.var(mean_IS_sum)  # A single scalar value

    IS_mean = torch.mean(mean_IS_sum) # A single scalar value

    return IS_mean, IS_variance


  def IS_pipeline(self):
    padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS = self.prepare_IS()
    # timestep_bootstraps_IS, rewards_bootstraps_IS, weights_bootstraps_IS = self.bootstrap_IS(padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS)
    IS_bootstraps = self.bootstrap_IS(padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS)
    # IS_mean, IS_variance = self.calc_variance_IS(timestep_bootstraps_IS, rewards_bootstraps_IS, weights_bootstraps_IS)
    IS_mean, IS_variance = self.calc_var_IS(IS_bootstraps)

    return IS_mean, IS_variance



  # ---------------------
  # SCOPE calculations
  # ---------------------

  def pass_states(self, padded_states_next_tensors, padded_states_current_tensors):
    states_next_output = self.model(padded_states_next_tensors)
    states_current_output = self.model(padded_states_current_tensors)

    return states_next_output.squeeze(), states_current_output.squeeze()

  def bootstrap_straight(self, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output, multi = None):
      
      if multi is None:
        seed = 42
        torch.manual_seed(seed)

      num_samples = self.num_bootstraps
      num_bootstraps_lin = num_samples*padded_timestep_tensors.shape[0]

      # Sample indices with replacement
      sampled_indices = torch.randint(0, len(padded_timestep_tensors), size=(num_bootstraps_lin,), dtype=torch.long)

      reshaped_size = (num_samples, padded_timestep_tensors.shape[0], padded_timestep_tensors.shape[1])

      padded_scope = self.gamma**(padded_timestep_tensors)*padded_weight_tensors*(padded_reward_tensors +self.gamma*states_next_output - states_current_output)
      scope_bootstraps = padded_scope[sampled_indices].view(reshaped_size)

      return scope_bootstraps

  def pass_then_boostraps(self, padded_states_next_tensors, padded_states_current_tensors, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors):
    states_next_output, states_current_output = self.pass_states(padded_states_next_tensors, padded_states_current_tensors)
    # timestep_bootstraps, rewards_bootstraps, weights_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps = self.bootstrap_straight(padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output)
    scope_bootstraps = self.bootstrap_straight(padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output)
    # return timestep_bootstraps, rewards_bootstraps, weights_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps
    return scope_bootstraps

  def calc_var_straight(self, scope_bootstraps):

    # Step 1: Sum along the third dimension
    sum_scope_trajectories = torch.sum(scope_bootstraps, dim=2)  # Shape: [1000, 1000]

    # Step 2: Take the mean along the second dimension
    mean_scope_sum = torch.mean(sum_scope_trajectories, dim=1)  # Shape: [1000]

    # Step 3: Calculate the variance across the first dimension
    scope_variance = torch.var(mean_scope_sum)  # A single scalar value

    scope_mean = torch.mean(mean_scope_sum) # A single scalar value

    return scope_mean, scope_variance

  def train_var_scope(self, num_epochs, learning_rate, shaping_coefficient, scope_weight=1, mse_weight=1): #, folder_name, filename)

      # IS terms for comparison to SCOPE
      padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS = self.prepare_IS()
      # timestep_bootstraps_IS, rewards_bootstraps_IS, weights_bootstraps_IS = self.bootstrap_IS(padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS)
      # IS_mean, IS_variance = self.calc_variance_IS(timestep_bootstraps_IS, rewards_bootstraps_IS, weights_bootstraps_IS)

      IS_bootstraps = self.bootstrap_IS(padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS)
      IS_mean, IS_variance = self.calc_var_IS(IS_bootstraps)

      # SCOPE terms for training phi
      padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor, states_last_tensor, psi_last_tensor = self.prepare_SCOPE_phi()


      self.model.train()

      # Enable anomaly detection
      torch.autograd.set_detect_anomaly(True)

      optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
      # Initialize empty list to store metrics and model state
      all_metrics = []
      model_states = []

      for epoch in range(num_epochs):
          total_loss = 0


          # timestep_bootstraps, rewards_bootstraps, weights_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps = self.pass_then_boostraps(padded_states_next_tensors, padded_states_current_tensors, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors)

          states_next_output, states_current_output = self.pass_states(padded_states_next_tensors, padded_states_current_tensors)
          # timestep_bootstraps, rewards_bootstraps, weights_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps = self.bootstrap_straight(padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output)
          # SCOPE_mean, SCOPE_variance = self.calc_variance_straight(timestep_bootstraps, weights_bootstraps, rewards_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps)

          scope_bootstraps = self.bootstrap_straight(padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output)
          SCOPE_mean, SCOPE_variance = self.calc_var_straight(scope_bootstraps)

          # mse_loss = F.mse_loss(states_current_output, 0.2*padded_psi_tensors)
          mse_loss = F.mse_loss(states_current_output, shaping_coefficient*padded_psi_tensors, reduction='none')
          masked_mse_loss = mse_loss * mask_tensor

          states_last_output = self.model(states_last_tensor)
          mse_states_last_loss = F.mse_loss(states_last_output.squeeze(),shaping_coefficient*psi_last_tensor, reduction = 'none')

          # mean_mse_loss = masked_mse_loss.mean()
          sum_mse_loss = torch.sum(masked_mse_loss, dim = 1)

          mean_mse_loss = torch.mean(sum_mse_loss + mse_states_last_loss)


          print(f"Epoch {epoch+1}")
          print(f"IS mean: {IS_mean},IS variance: {IS_variance}")
          print("SCOPE Var loss: ", SCOPE_variance)
          print("MSE loss: ", mean_mse_loss.item())


          # Testing evaluaton
          self.model.eval()
          scope_mean, scope_var = self.evaluate_scope()
          print(f"SCOPE mean: {scope_mean}, SCOPE var: {scope_var}")

          self.model.train()


          # tot = SCOPE_variance
          # tot = SCOPE_variance + mse_loss
          tot = scope_weight*SCOPE_variance + mse_weight*mean_mse_loss

          optimizer.zero_grad()

          # Retain the graph to avoid clearing it before backward pass
          tot.backward(retain_graph=True)

          optimizer.step()

          total_loss += tot.item()

          print(f"Total Loss: {total_loss}")
          print("-" * 40)
          # Append metrics to the list
          epoch_metrics = {
              "epoch": epoch + 1,
              # "IS_mean": IS_mean.item(),
              # "IS_variance": IS_variance.item(),
              "Train_mean": SCOPE_mean.item(),
              "Train_variance": SCOPE_variance.item(),
              "Train_mse_loss": mean_mse_loss.item(),
              "total_loss": total_loss,
              "Test_mean": scope_mean.item(),
              "Test_variance": scope_var.item()
          }

          all_metrics.append(epoch_metrics)

          # temporary_model_weights = self.model.state_dict()
          # model_weight_epoch = temporary_model_weights.copy()

          model_state = self.model.state_dict()

          # print(self.model.state_dict())
          # print(model_state)
          # Save model weights every 25 epochs
          if (epoch + 1) % 25 == 0:
              # Append model state to the model states list
              model_states.append({"epoch": epoch + 1, "model_state":  copy.deepcopy(model_state)})

      experiment_metrics = {"per_epoch": all_metrics, "model_weights": model_states}

      # Disable anomaly detection after running the code
      torch.autograd.set_detect_anomaly(False)

      # for name, param in self.model.named_parameters():
      #     if param.requires_grad:
      #         print(f"Parameter name: {name}")
      #         print(f"Weights: {param.data}")

      return experiment_metrics #all_metrics #,self.model #, sum_mse_loss, mse_states_last_loss, all_metrics

  def evaluate_scope(self):
    self.model.eval()
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors = self.prepare_SCOPE_test()
    # timestep_bootstraps, rewards_bootstraps, weights_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps = self.pass_then_boostraps(padded_states_next_tensors, padded_states_current_tensors, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors)
    # SCOPE_mean, SCOPE_variance = self.calc_variance_straight(timestep_bootstraps, weights_bootstraps, rewards_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps)

    scope_bootstraps = self.pass_then_boostraps(padded_states_next_tensors, padded_states_current_tensors, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors)
    SCOPE_mean, SCOPE_variance = self.calc_var_straight(scope_bootstraps)

    return SCOPE_mean, SCOPE_variance

  '''
  Multi Experiment Bootstrapping Error Bars
  '''

  # def bootstrap_straight(self, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output):
  #   seed = 42
  #   torch.manual_seed(seed)

  #   num_samples = self.num_bootstraps
  #   num_bootstraps_lin = num_samples*padded_timestep_tensors.shape[0]

  #   # Sample indices with replacement
  #   sampled_indices = torch.randint(0, len(padded_timestep_tensors), size=(num_bootstraps_lin,), dtype=torch.long)

  #   reshaped_size = (num_samples, padded_timestep_tensors.shape[0], padded_timestep_tensors.shape[1])

  #   padded_scope = self.gamma**(padded_timestep_tensors)*padded_weight_tensors*(padded_reward_tensors +self.gamma*states_next_output - states_current_output)
  #   scope_bootstraps = padded_scope[sampled_indices].view(reshaped_size)

  #   return scope_bootstraps


  def IS_pipeline_multi(self, num_multi):
    padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS = self.prepare_IS()
    # timestep_bootstraps_IS, rewards_bootstraps_IS, weights_bootstraps_IS = self.bootstrap_IS(padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS)
    IS_all_means = []
    IS_all_variances = []
    for i in range(num_multi):
             
      IS_bootstraps = self.bootstrap_IS(padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS, multi = num_multi)
      # IS_mean, IS_variance = self.calc_variance_IS(timestep_bootstraps_IS, rewards_bootstraps_IS, weights_bootstraps_IS)
      IS_mean, IS_variance = self.calc_var_IS(IS_bootstraps)
      IS_all_means.append(IS_mean)
      IS_all_variances.append(IS_variance)

    return IS_all_means, IS_all_variances

  def SCOPE_pipeline_multi(self, num_multi):

      Train_means, Train_variances = [], []
      Test_means, Test_variances = [], []
      
      self.model.eval()
      padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor, states_last_tensor, psi_last_tensor = self.prepare_SCOPE_phi()
      states_next_output, states_current_output = self.pass_states(padded_states_next_tensors, padded_states_current_tensors)
      
      for i in range(num_multi):
        
        scope_bootstraps = self.bootstrap_straight(padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output, multi=num_multi)
        Train_mean, Train_variance = self.calc_var_straight(scope_bootstraps)
        Train_means.append(Train_mean)
        Train_variances.append(Train_variance)

      
      # self.model.eval()
      padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors = self.prepare_SCOPE_test()        
      states_next_output, states_current_output = self.pass_states(padded_states_next_tensors, padded_states_current_tensors)
      
      for j in range(num_multi):
         
         scope_bootstraps = self.bootstrap_straight(padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output, multi=num_multi)
         Test_mean, Test_variance = self.calc_var_straight(scope_bootstraps)
         Test_means.append(Test_mean)
         Test_variances.append(Test_variance)
      
      return Train_means, Train_variances, Test_means, Test_variances
           


  # def multi_experiment(self, num_experiments):
  #    IS_all_means, IS_all_variances = self.IS_pipeline_multi(num_experiments)
     
  #    return IS_all_means, IS_all_variances, SCOPE_mean, SCOPE_variance
     

  '''
  On Policy Calculations
  '''
  def calc_V_pi_e(self):
      all_timesteps = []
      gamma = self.gamma
      for j in range(len(self.pi_e)):
          Timestep_values = []
          for i in range(len(self.pi_e[j])):
            # print(i)
            timestep = gamma ** (i) * self.pi_e[j][i][2]
            Timestep_values.append(timestep)

          all_timesteps.append(Timestep_values)

      V_est = sum([sum(sublist) for sublist in all_timesteps])/len(self.pi_e)
      return V_est

