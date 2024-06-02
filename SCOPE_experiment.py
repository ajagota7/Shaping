from typing import List, Union, Callable
import numpy as np
import torch
import os 
from neural_net import NN_l1_l2_reg
from SCOPE_straight import SCOPE_straight

from lifegate import LifeGate


class SCOPE_experiment():
    def __init__(self,
                #  Parameters related to policy generation
                 pi_b_top_k: int,
                 pi_b_epsilon: float,
                 pi_e_top_k: int,
                 pi_e_epsilon: float,
                 q_table,#: List[List[float]],
                 gamma: float,
                 num_trajectories: int,
                 num_bootstraps: int,
                 percent_to_estimate_phi: float,
                 shaping_feature: Callable,
                 shaping_coefficient: float,
                #  Parameters related to neural network architecture and training
                 hidden_dims: List[int],
                 learning_rate: float,
                 dropout_prob: float,
                 l1_reg: float,
                 l2_reg: float,
                 scope_weight: float,
                 mse_weight: float,
                 num_epochs: int,
                #  Parameters related to environment
                 max_length: int,
                 death_drag: float,

                #  Other general parameters
                 dtype: str,
                 experiment_type: str,
                 folder_path: str):
                # folder_name):

        self.pi_b_top_k = pi_b_top_k
        self.pi_b_epsilon = pi_b_epsilon
        self.pi_e_top_k = pi_e_top_k
        self.pi_e_epsilon = pi_e_epsilon
        self.q_table = q_table
        self.gamma = gamma
        self.num_trajectories = num_trajectories
        self.num_bootstraps = num_bootstraps
        self.percent_to_estimate_phi = percent_to_estimate_phi
        self.shaping_feature = shaping_feature
        self.shaping_coefficient = shaping_coefficient

        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.scope_weight = scope_weight
        self.mse_weight = mse_weight
        self.num_epochs = num_epochs

        self.max_length = max_length
        self.death_drag = death_drag

        self.dtype = dtype
        self.experiment_type = experiment_type
        self.folder_path = folder_path
        # self.folder_name = folder_name

    '''
    Infrastructure for generating policies
    Generating transition distributions
    Choosing actions
      Chosen shaping features
    Traversing trajectory
    '''

    def action_probs_top_n_epsilon(self, top_k, epsilon):
      """
      Calculate action probabilities with epsilon-greedy strategy for top actions.

      Parameters:
      - n: Number of top actions
      - epsilon: Exploration-exploitation trade-off parameter

      Returns:
      - action_probs: Calculated action probabilities
      """

      num_actions = self.q_table.shape[-1]

      # Initialize a 2D array to represent action probabilities
      action_probs = np.zeros_like(self.q_table)

      # For each state, set the probability for the top two actions
      for i in range(self.q_table.shape[0]):
          for j in range(self.q_table.shape[1]):
              sorted_actions = np.argsort(self.q_table[j, i])  # Get the indices of all actions, sorted by Q-value
              top_actions = sorted_actions[-top_k:]  # Get the indices of the top two actions
              non_top_actions = sorted_actions[:-top_k]
              action_probs[i, j, top_actions] = (1 - epsilon) / top_k + epsilon/num_actions  # Split the probability evenly between the top two actions
              action_probs[i, j, non_top_actions] = epsilon/num_actions

      return action_probs


    def initialize_env(self):
      # Initalize lifegate class
      env = LifeGate(max_steps=self.max_length, state_mode='tabular',
                        rendering=True, image_saving=False, render_dir=None, rng=np.random.RandomState(1234), death_drag = self.death_drag)

      return env


    def choose_action(self, state, action_probs):
        # Get the probability distribution for the given state
        state_probs = action_probs[state]

        # Choose an action based on the probabilities
        action = np.random.choice(len(state_probs), p=state_probs)

        return action


    def experiment_actions(self, nb_trajectories, env, action_probs):#, shaping_feature):
        """
        Run the experiment for a specified number of episodes.

        Parameters:
        - nb_trajectories: Number of episodes
        - env: Experiment environment
        - action_probs: Action probabilities

        Returns:
        - policies: List of policies (pi_b or pi_e)
        """
        # Define the dtype for the structured array
        dtype = [
            ('state', np.float64, (2,)),
            ('action', np.int64),
            ('reward', np.float64),
            ('state_next', np.float64, (2,)),
            ('timestep', np.int64),
            ('psi', np.float64)

        ]

        policies = []
        for i in range(nb_trajectories):
            trajectory = np.empty(0, dtype=dtype)
            s = env.reset()
            env.render()
            term = False
            timestep = 0
            while not term:
                state_last = s
                action = self.choose_action(tuple(s), action_probs)
                s, r, term, _ = env.step(action)


                # psi = smallest_distance_to_deadend(state_last, env)
                psi = self.shaping_feature(state_last, env)

                data_point = np.array([(state_last, action, r, s, timestep, psi)], dtype=dtype)
                trajectory = np.append(trajectory, data_point)
                timestep += 1

            policies.append(trajectory)

        # with open('policies.pkl', 'wb') as f:
        #     pickle.dump(policies, f)

        return policies


    '''
    Running experiment:
      Preare environment
      Prepare model
      Prepare filename
      Running experiment
    '''

    def initalize_model(self):
      model = NN_l1_l2_reg(input_dim=2, hidden_dims=self.hidden_dims, dropout_prob=self.dropout_prob, output_dim=1, dtype = torch.float64, l1_lambda=self.l1_reg, l2_lambda = self.l2_reg)

    def prepare_experiment(self):
      env = self.initialize_env()
      P_pi_b = self.action_probs_top_n_epsilon(self.pi_b_top_k, self.pi_b_epsilon)
      P_pi_e = self.action_probs_top_n_epsilon(self.pi_e_top_k, self.pi_e_epsilon)
      pi_b = self.experiment_actions(self.num_trajectories, env, P_pi_b)
      pi_e = self.experiment_actions(1000, env, P_pi_e)

      # consider changing this to method within class
      model = NN_l1_l2_reg(input_dim=2, hidden_dims=self.hidden_dims, output_dim=1, dtype = self.dtype, l1_lambda=self.l1_reg, l2_lambda = self.l2_reg)

      experiment_class = SCOPE_straight(model, self.gamma, self.num_bootstraps, pi_b, P_pi_b, pi_e, P_pi_e, self.percent_to_estimate_phi, self.shaping_feature, env, self.dtype)

      return pi_b, pi_e, model, experiment_class

    def generate_file_name(self):
      shaping_function = self.shaping_feature.__name__
      hidden_dims_str = '_'.join(map(str, self.hidden_dims))  # Convert hidden_dims to a string
      return f"{self.num_trajectories}_{self.gamma}_{self.percent_to_estimate_phi}_{shaping_function}_{self.shaping_coefficient}_{hidden_dims_str}_{self.dropout_prob}_{self.learning_rate}_{self.l1_reg}_{self.l2_reg}_{self.scope_weight}_{self.mse_weight}_{self.max_length}"

    def run_experiment(self):
      filename = self.generate_file_name()

      # generate file path with folder and filename
      file_path = os.path.join(self.folder_path, f"{filename}.pt")

      # Check if experiment exists
      if os.path.exists(file_path):
        print(f"The file {filename}.pt already exists in the folder.")
      else:
        pi_b, pi_e, model, experiment_class = self.prepare_experiment()
        all_metrics = experiment_class.train_var_scope(self.num_epochs, self.learning_rate, self.shaping_coefficient, self.scope_weight, self.mse_weight)
        on_policy_estimate = experiment_class.calc_V_pi_e()
        IS_mean, IS_variance = experiment_class.IS_pipeline()


        experiment_data = {
            "Experiment Parameters": self.__dict__,
            "Experiment Metrics": all_metrics,
            "On Policy Estimate": on_policy_estimate,
            "pi_b": pi_b,
            "pi_e": pi_e,
            "IS Estimate": {"Estimate": IS_mean.item(), "Variance": IS_variance.item()}
        }

        # torch.save(experiment_data, f"{self.folder_path}/{filename}.pt")
        torch.save(experiment_data, file_path)

    def continue_training(self, epoch):
      filename = self.generate_file_name()
      # generate file path with folder and filename
      file_path = os.path.join(self.folder_path, f"{filename}.pt")
      loaded_data = torch.load(file_path)
      experiment_parameters = loaded_data["Experiment Parameters"]
      
      # # Access specific data from the loaded dictionary
      # experiment_metrics = loaded_data["Experiment Metrics"]

      # per_epoch = experiment_metrics["per_epoch"]
      # model_weights = experiment_metrics["model_weights"]
      env = self.initialize_env()
      pi_b = loaded_data["pi_b"]
      pi_e = loaded_data["pi_e"]
      P_pi_b = self.action_probs_top_n_epsilon(self.pi_b_top_k, self.pi_b_epsilon)
      P_pi_e = self.action_probs_top_n_epsilon(self.pi_e_top_k, self.pi_e_epsilon)
      
      latest_weights = loaded_data["Experiment Metrics"]["model_weights"][-1]['model_state']

      model = NN_l1_l2_reg(input_dim=2,
                      hidden_dims=experiment_parameters["hidden_dims"],
                      output_dim=1, dtype = experiment_parameters["dtype"],
                      l1_lambda=experiment_parameters["l1_reg"],
                      l2_lambda = experiment_parameters["l2_reg"])

      # Load the final weights into the model
      model.load_state_dict(latest_weights)

      experiment_class = SCOPE_straight(model, self.gamma, self.num_bootstraps, pi_b, P_pi_b, pi_e, P_pi_e, self.percent_to_estimate_phi, self.shaping_feature, env, self.dtype)
      all_metrics_new = experiment_class.train_var_scope(epoch, self.learning_rate, self.shaping_coefficient, self.scope_weight, self.mse_weight)
      
      final_epoch = loaded_data["Experiment Metrics"]["model_weights"][-1]['epoch']

      # experiment_metrics = loaded_data["Experiment Metrics"]


      # Update epochs in the new metrics
      for e in all_metrics_new["model_weights"]:
          e['epoch'] += final_epoch
      for ep in all_metrics_new["per_epoch"]:
          ep['epoch'] += final_epoch

      # Append new metrics to the existing metrics
      loaded_data["Experiment Metrics"]["model_weights"].extend(all_metrics_new["model_weights"])
      loaded_data["Experiment Metrics"]["per_epoch"].extend(all_metrics_new["per_epoch"])

      # Save the updated data back to the same file
      torch.save(loaded_data, file_path)



    def get_multi_experiment(self):
        filename = self.generate_file_name()
        # generate file path with folder and filename
        file_path = os.path.join(self.folder_path, f"{filename}.pt")
        loaded_data = torch.load(file_path)
        experiment_parameters = loaded_data["Experiment Parameters"]
        
        env = self.initialize_env()
        pi_b = loaded_data["pi_b"]
        pi_e = loaded_data["pi_e"]
        P_pi_b = self.action_probs_top_n_epsilon(self.pi_b_top_k, self.pi_b_epsilon)
        P_pi_e = self.action_probs_top_n_epsilon(self.pi_e_top_k, self.pi_e_epsilon)
        
        latest_weights = loaded_data["Experiment Metrics"]["model_weights"][-1]['model_state']

        model = NN_l1_l2_reg(input_dim=2,
                        hidden_dims=experiment_parameters["hidden_dims"],
                        output_dim=1, dtype = experiment_parameters["dtype"],
                        l1_lambda=experiment_parameters["l1_reg"],
                        l2_lambda = experiment_parameters["l2_reg"])

        # Load the final weights into the model
        model.load_state_dict(latest_weights)

        experiment_class = SCOPE_straight(model, self.gamma, self.num_bootstraps, pi_b, P_pi_b, pi_e, P_pi_e, self.percent_to_estimate_phi, self.shaping_feature, env, self.dtype)
        IS_all_means, IS_all_variances = experiment_class.IS_pipeline_multi(num_multi=5)

        Train_means, Train_variances, Test_means, Test_variances = experiment_class.SCOPE_pipeline_multi(self, num_multi = 5)

        return IS_all_means, IS_all_variances, Train_means, Train_variances, Test_means, Test_variances

    def load_experiment(self):
      filename = self.generate_file_name()
      # generate file path with folder and filename
      file_path = os.path.join(self.folder_path, f"{filename}.pt")

      loaded_data = torch.load(file_path)

      return loaded_data

'''
    Need to add functions for accessing chosen model weights and training from weights
    Choose from stored epochs 
    initialize model with those weights
    continue training, potentially with modified parameters if need be
    If parameters modified, save new file 
'''



