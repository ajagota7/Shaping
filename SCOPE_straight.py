import numpy as np 
from IS import calculate_importance_weights

import torch 

class SCOPE_straight(object):

  def __init__(self, model, gamma, num_bootstraps, pi_b, P_pi_b, P_pi_e, percent_to_estimate_phi, dtype):
        self.model = model
        self.gamma = gamma
        self.num_bootstraps = num_bootstraps
        self.pi_b = pi_b
        self.P_pi_b = P_pi_b
        self.P_pi_e = P_pi_e
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

      for index, policy in enumerate(chosen_policies):
          policy_array = np.array(policy)

          timesteps.append(policy_array['timestep'].astype(int))
          actions.append(policy_array['action'])
          rewards.append(policy_array['reward'].astype(float))
          psi.append(policy_array['psi'])

          states_next.append(policy_array['state_next'])
          states_current.append(policy_array['state'])

      return timesteps, rewards, states_next, states_current, weights, actions, psi

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
    # padded_psi_tensors = padded_psi_tensors.unsqueeze(-1)
    mask_tensor = torch.tensor(mask, dtype = self.dtype)
    return padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor

  #-----------------------
  # Preparation Functions
  # ---------------------- 

  def prepare_IS(self):
    timesteps, rewards, states_next, states_current, weights, actions,_ = self.prep_policies(self.pi_b)
    padded_timesteps, padded_rewards, padded_actions, padded_weights = self.padding_IS_terms(timesteps, actions, rewards, weights)
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors = self.tensorize_IS_terms(padded_timesteps, padded_rewards, padded_weights)

    return padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors

  def prepare_SCOPE(self, policies):
    timesteps, rewards, states_next, states_current, weights, actions, psi = self.prep_policies(policies)
    padded_timesteps, padded_rewards, padded_actions, padded_weights = self.padding_IS_terms(timesteps, actions, rewards, weights)
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors = self.tensorize_IS_terms(padded_timesteps, padded_rewards, padded_weights)
    padded_states_next, padded_states_current, padded_psi, mask = self.padding_states(states_next, states_current, psi)
    padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor = self.tensorize_padded_terms(padded_states_next, padded_states_current, padded_psi, mask)

    return padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor

  def prepare_SCOPE_phi(self):
    phi_set,_ = self.subset_policies()
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor = self.prepare_SCOPE(phi_set)

    return padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor

  def prepare_SCOPE_test(self):
    _, scope_set = self.subset_policies()
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors,_,_ = self.prepare_SCOPE(scope_set)

    return padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors


  # ----------------
  # IS Calculations
  # ----------------
 

  def bootstrap_IS(self, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors):
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

  # def calc_variance_IS(self, timestep_bootstraps, weights_bootstraps, rewards_bootstraps):
  #   IS_bootstraps = self.gamma**(timestep_bootstraps)* weights_bootstraps *(rewards_bootstraps)

  #   # Step 1: Sum along the third dimension
  #   sum_IS_trajectories = torch.sum(IS_bootstraps, dim=2)  # Shape: [1000, 1000]

  #   # Step 2: Take the mean along the second dimension
  #   mean_IS_sum = torch.mean(sum_IS_trajectories, dim=1)  # Shape: [1000]

  #   # Step 3: Calculate the variance across the first dimension
  #   IS_variance = torch.var(mean_IS_sum)  # A single scalar value

  #   IS_mean = torch.mean(mean_IS_sum) # A single scalar value

  #   return IS_mean, IS_variance

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

  def bootstrap_straight(self, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output):
      seed = 42
      torch.manual_seed(seed)

      num_samples = self.num_bootstraps
      num_bootstraps_lin = num_samples*padded_timestep_tensors.shape[0]

      # Sample indices with replacement
      sampled_indices = torch.randint(0, len(padded_timestep_tensors), size=(num_bootstraps_lin,), dtype=torch.long)

      reshaped_size = (num_samples, padded_timestep_tensors.shape[0], padded_timestep_tensors.shape[1])

      padded_scope = self.gamma**(padded_timestep_tensors)*padded_weight_tensors*(padded_reward_tensors +self.gamma*states_next_output - states_current_output)
      scope_bootstraps = padded_scope[sampled_indices].view(reshaped_size)

      # timestep_bootstraps = padded_timestep_tensors[sampled_indices].view(reshaped_size)
      # rewards_bootstraps = padded_reward_tensors[sampled_indices].view(reshaped_size)
      # weights_bootstraps = padded_weight_tensors[sampled_indices].view(reshaped_size)

      # phi_states_next_bootstraps = states_next_output[sampled_indices].view(reshaped_size)
      # phi_states_current_bootstraps = states_current_output[sampled_indices].view(reshaped_size)
      # return timestep_bootstraps, rewards_bootstraps, weights_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps
      
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

  # def calc_variance_straight(self, timestep_bootstraps, weights_bootstraps, rewards_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps):

  #   # IS_boostraps = self.gamma**(timestep_bootstraps)* weights_bootstraps *(rewards_bootstraps)
  #   scope_bootstraps = self.gamma**(timestep_bootstraps)* weights_bootstraps *(rewards_bootstraps  +self.gamma*phi_states_next_bootstraps - phi_states_current_bootstraps)

  #   # Step 1: Sum along the third dimension
  #   sum_scope_trajectories = torch.sum(scope_bootstraps, dim=2)  # Shape: [1000, 1000]

  #   # Step 2: Take the mean along the second dimension
  #   mean_scope_sum = torch.mean(sum_scope_trajectories, dim=1)  # Shape: [1000]

  #   # Step 3: Calculate the variance across the first dimension
  #   scope_variance = torch.var(mean_scope_sum)  # A single scalar value

  #   scope_mean = torch.mean(mean_scope_sum) # A single scalar value

  #   return scope_mean, scope_variance

  def train_var_scope(self, num_epochs, learning_rate, scope_weight=1, mse_weight=1):

      # IS terms for comparison to SCOPE
      padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS = self.prepare_IS()
      # timestep_bootstraps_IS, rewards_bootstraps_IS, weights_bootstraps_IS = self.bootstrap_IS(padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS)
      # IS_mean, IS_variance = self.calc_variance_IS(timestep_bootstraps_IS, rewards_bootstraps_IS, weights_bootstraps_IS)

      IS_bootstraps = self.bootstrap_IS(padded_timestep_tensors_IS, padded_reward_tensors_IS, padded_weight_tensors_IS)
      IS_mean, IS_variance = self.calc_var_IS(IS_bootstraps)

      # SCOPE terms for training phi
      padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors, padded_psi_tensors, mask_tensor = self.prepare_SCOPE_phi()


      self.model.train()

      # Enable anomaly detection
      torch.autograd.set_detect_anomaly(True)

      optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

      for epoch in range(num_epochs):
          total_loss = 0


          # timestep_bootstraps, rewards_bootstraps, weights_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps = self.pass_then_boostraps(padded_states_next_tensors, padded_states_current_tensors, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors)

          states_next_output, states_current_output = self.pass_states(padded_states_next_tensors, padded_states_current_tensors)
          # timestep_bootstraps, rewards_bootstraps, weights_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps = self.bootstrap_straight(padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output)
          # SCOPE_mean, SCOPE_variance = self.calc_variance_straight(timestep_bootstraps, weights_bootstraps, rewards_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps)
          
          scope_bootstraps = self.bootstrap_straight(padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, states_next_output, states_current_output)
          SCOPE_mean, SCOPE_variance = self.calc_var_straight(scope_bootstraps)
          
          # mse_loss = F.mse_loss(states_current_output, 0.2*padded_psi_tensors)
          mse_loss = F.mse_loss(states_current_output, 0.1*padded_psi_tensors, reduction='none')
          masked_mse_loss = mse_loss * mask_tensor

          mean_mse_loss = masked_mse_loss.mean()


          print(f"Epoch {epoch+1}")
          print("IS variance: ", IS_variance)
          print("SCOPE Var loss: ", SCOPE_variance)
          print("MSE loss: ", mean_mse_loss.item())

          # Testing evaluaton
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

      # Disable anomaly detection after running the code
      torch.autograd.set_detect_anomaly(False)

      for name, param in self.model.named_parameters():
          if param.requires_grad:
              print(f"Parameter name: {name}")
              print(f"Weights: {param.data}")

      return self.model

  def evaluate_scope(self):
    self.model.eval()
    padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors, padded_states_next_tensors, padded_states_current_tensors = self.prepare_SCOPE_test()
    # timestep_bootstraps, rewards_bootstraps, weights_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps = self.pass_then_boostraps(padded_states_next_tensors, padded_states_current_tensors, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors)
    # SCOPE_mean, SCOPE_variance = self.calc_variance_straight(timestep_bootstraps, weights_bootstraps, rewards_bootstraps, phi_states_next_bootstraps, phi_states_current_bootstraps)

    scope_bootstraps = self.pass_then_boostraps(padded_states_next_tensors, padded_states_current_tensors, padded_timestep_tensors, padded_reward_tensors, padded_weight_tensors)
    SCOPE_mean, SCOPE_variance = self.calc_var_straight(scope_bootstraps)

    return SCOPE_mean, SCOPE_variance


  # -----------------------
  # Heatmaps for lifegate
  # -----------------------
  def get_model_output_dict(self):

    self.model.eval()

    # Initialize an empty dictionary to store data
    data = {}

    # Loop through all combinations from [0,0] to [9,9]
    for i in range(10):
      for j in range(10):
          # Prepare input data
          input_data = torch.tensor([i, j], dtype=torch.float64)

          # Pass input through the self.model
          output = self.model(input_data)

          # Store data in the dictionary
          data[(i, j)] = output.item()

    return data

  def plot_heatmap(self, data):
    values = np.zeros((10, 10))
    for (x, y), value in data.items():
        values[y, x] = value

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(z=values, colorscale='viridis'))

    # Add colorbar
    fig.update_layout(coloraxis_colorbar=dict(title='Values',
                                              ticks='outside',
                                              tickvals=[np.min(values), np.max(values)],
                                              ticktext=[np.min(values), np.max(values)]))

    # Add labels and title
    fig.update_layout(xaxis=dict(tickvals=np.arange(10), ticktext=list(range(10)), title='X'),
                      yaxis=dict(tickvals=np.arange(9, -1, -1), ticktext=list(range(9, -1, -1)), title='Y', autorange="reversed"),
                      title='Heatmap')

    fig.show()

  def get_heatmap(self):
    data = self.get_model_output_dict()
    self.plot_heatmap(data)

  # ---------------------
  # State Visitation Heatmap
  # ---------------------

  def count_state_visits(self):
    state_visit_counts = {}
    for trajectory in self.pi_b:
        for data_point in trajectory:
            state = tuple(data_point['state'])
            if state not in state_visit_counts:
                state_visit_counts[state] = 0
            state_visit_counts[state] += 1

        # Include last state_next of the trajectory
        last_state_next = tuple(trajectory[-1]['state_next'])
        if last_state_next not in state_visit_counts:
            state_visit_counts[last_state_next] = 0
        state_visit_counts[last_state_next] += 1

    return state_visit_counts

  def create_state_visit_dict(self):
      state_visit_dict = {}
      for i in range(10):
          for j in range(10):
              state_visit_dict[(i, j)] = 0
      return state_visit_dict

  def fill_state_visit_dict(self,state_visit_counts):
      state_visit_dict = self.create_state_visit_dict()
      for state, count in state_visit_counts.items():
          state_visit_dict[state] = count
      return state_visit_dict


  def plot_state_visitations_heatmap(self, state_visit_dict):
    # Create lists to store x, y, and z values
    x = []
    y = []
    z = []

    # Iterate through the state visit dictionary
    for state, count in state_visit_dict.items():
        x.append(state[0])
        y.append(9 - state[1])  # Flip y-axis to have (0, 0) at the bottom-left
        z.append(count)

    # Create the heatmap trace
    trace = go.Heatmap(
        x=x,
        y=y,
        z=z,
        colorscale='Viridis',  # Choose a colorscale
        colorbar=dict(title='Visits'),
        zmin=0,
        zmax=max(z)  # Set maximum value for the color scale
    )

    # Create layout
    layout = go.Layout(
        title='State Visitations Heatmap',
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis', tickvals=list(range(10)), ticktext=list(range(9, -1, -1))),
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


  def get_state_visitation_heatmap(self):

    # Count state visits
    state_visit_counts = self.count_state_visits()

    # Fill state visit dictionary
    state_visit_dict = self.fill_state_visit_dict(state_visit_counts)

    # Assuming state_visit_dict is your dictionary with state visitations
    self.plot_state_visitations_heatmap(state_visit_dict)


