import numpy as np

# def get_e_greedy_action(s, epsilon, qnet, explore = True):
#     """
#     Get epsilon-greedy action.

#     Parameters:
#     - s: Current state
#     - epsilon: Exploration-exploitation trade-off parameter
#     - qnet: Q-network or Q-table
#     - explore: Flag to enable exploration

#     Returns:
#     - action: Selected action
#     """
#     if explore and qnet.rng.binomial(1, epsilon):
#         action = qnet.rng.randint(env.nb_actions)
#     else:
#         action = qnet.get_max_action(s)
#     return action

def choose_action(state, action_probs):
    # Get the probability distribution for the given state
    state_probs = action_probs[state]

    # Choose an action based on the probabilities
    action = np.random.choice(len(state_probs), p=state_probs)

    return action

import numpy as np




def action_probs_top_n_epsilon(q_table, n, epsilon):
  """
  Calculate action probabilities with epsilon-greedy strategy for top actions.

  Parameters:
  - q_table: Q-table
  - n: Number of top actions
  - epsilon: Exploration-exploitation trade-off parameter

  Returns:
  - action_probs: Calculated action probabilities
  """
  # Define your epsilon value
  epsilon = 0.01  # Adjust the value of epsilon as needed

  num_actions = q_table.shape[-1]

  # Initialize a 2D array to represent action probabilities
  action_probs = np.zeros_like(q_table)

  # For each state, set the probability for the top two actions
  for i in range(q_table.shape[0]):
      for j in range(q_table.shape[1]):
          sorted_actions = np.argsort(q_table[j, i])  # Get the indices of all actions, sorted by Q-value
          top_actions = sorted_actions[-n:]  # Get the indices of the top two actions
          non_top_actions = sorted_actions[:-n]
          action_probs[i, j, top_actions] = (1 - epsilon) / n + epsilon/num_actions  # Split the probability evenly between the top two actions
          action_probs[i, j, non_top_actions] = epsilon/num_actions

  return action_probs
