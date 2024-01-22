

def experiment_actions(nb_episodes, env, action_probs):
    """
    Run the experiment for a specified number of episodes.

    Parameters:
    - nb_episodes: Number of episodes
    - env: Experiment environment
    - action_probs: Action probabilities

    Returns:
    - evaluation_policies: List of evaluation policies
    """
    evaluation_policies = []
    for i in range(nb_episodes):
        trajectory = []
        s = env.reset()
        env.render()
        term = False
        timestep = 0
        while not term:
            state_last = s
            # action = get_e_greedy_action(s, epsilon, qnet, explore = True)
            action = choose_action(tuple(s), action_probs)
            s, r, term, _ = env.step(action)
            # trajectory.append([state_last, action, r, s])
            psi = smallest_distance_to_deadend(state_last, env)
            trajectory.append(np.array([state_last, action, r, s, timestep, psi]))
            timestep +=1

        evaluation_policies.append(trajectory)
    with open('evaluation_policies.pkl', 'wb') as f:
        pickle.dump(evaluation_policies, f)
    return evaluation_policies