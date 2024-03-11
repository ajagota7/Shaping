import numpy as np

def calculate_importance_weights(eval_policy, behav_policy, behavior_policies):
    """
    Calculate importance weights for behavior policies.

    Parameters:
    - eval_policy: Evaluation policy
    - behav_policy: Behavior policy
    - behavior_policies: List of behavior policies

    Returns:
    - all_weights: List of importance weights
    """
    all_weights_temp = []
    for trajectory in behavior_policies:
        cum_ratio = 1
        cumul_weights = []
        for step in trajectory:
            # eval_action_probs = get_quadrant_policy(step[0], eval_policy)
            # behav_action_probs = get_quadrant_policy(step[0], behav_policy)

            P_pi_b = behav_policy[tuple(np.append(step[0].astype(int) , (step[1],)))]
            P_pi_e = eval_policy[tuple(np.append(step[0].astype(int) , (step[1],)))]

            # ratio = (0.8*eval_action_probs[step[1]] +0.2*0.25)/ (0.8*behav_action_probs[step[1]]+0.2*0.25)
            ratio = P_pi_e/P_pi_b
            cum_ratio *= ratio
            cumul_weights.append(cum_ratio)
        all_weights_temp.append(cumul_weights)

        # all_weights = [list(np.cumprod(i)) for i in all_weights_temp]

    return all_weights_temp

