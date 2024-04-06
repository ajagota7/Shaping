import choose_actions
from choose_actions import choose_action

import shaping_features
from shaping_features import smallest_distance_to_deadend
from shaping_features import smallest_distance_to_death
from shaping_features import smallest_distance_to_recovery


import pickle
import numpy as np


def experiment_actions(nb_episodes, env, action_probs):
    """
    Run the experiment for a specified number of episodes.

    Parameters:
    - nb_episodes: Number of episodes
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
        # ('dead_end_dist', np.float64),
        # ('recovery_dist', np.float64),
        # ('psi', np.float64, (2,))
        ('psi', np.float64)

    ]

    policies = []
    for i in range(nb_episodes):
        trajectory = np.empty(0, dtype=dtype)
        s = env.reset()
        env.render()
        term = False
        timestep = 0
        while not term:
            state_last = s
            action = choose_action(tuple(s), action_probs)
            s, r, term, _ = env.step(action)
            
            # dead_end_dist = smallest_distance_to_deadend(state_last, env)
            # recovery_dist = smallest_distance_to_recovery(state_last, env)
            # psi = [dead_end_dist, recovery_dist]            

            psi = smallest_distance_to_deadend(state_last, env)

            data_point = np.array([(state_last, action, r, s, timestep, psi)], dtype=dtype)
            trajectory = np.append(trajectory, data_point)
            timestep += 1

        policies.append(trajectory)
    
    with open('policies.pkl', 'wb') as f:
        pickle.dump(policies, f)
    
    return policies