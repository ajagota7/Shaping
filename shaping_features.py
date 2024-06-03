import math

def euclidean_distance(state1, state2):
    return math.sqrt((state1[0] - state2[0]) ** 2 + (state1[1] - state2[1]) ** 2)

def manhattan_distance(state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

def smallest_distance_to_deadend(current_state, env):

    return min(manhattan_distance(current_state, dead_end) for dead_end in env.dead_ends)

def smallest_distance_to_death(current_state, env):

    return min(manhattan_distance(current_state, deaths) for deaths in env.main_deaths)

def smallest_distance_to_recovery(current_state, env):

    return min(manhattan_distance(current_state, recovery) for recovery in env.possible_recoveries)


def smallest_distance_to_dead_end_or_death(current_state,env):

    all_death = env.main_deaths + env.dead_ends
    return min(manhattan_distance(current_state, dead_end) for dead_end in all_death)





def bottleneck_four_regions_k_p9_a_1(current_state, env,k = 0.9,a = 1):
    
    # Bottom left region
    if current_state[0] <4 and current_state[1]>4:
        dist_to_bottleneck = manhattan_distance(current_state, [0,5])
        return k*math.exp(-a*dist_to_bottleneck)
    
    # Top left region
    elif current_state[0] <5 and current_state[1] <5:
        dist_to_recovery = smallest_distance_to_recovery(current_state,env)
        return k*math.exp(-a*dist_to_recovery) + 0.1
    
    # Top right region
    elif current_state[0]>4 and current_state[1]<5:
        dist_to_recovery = smallest_distance_to_recovery(current_state,env)
        dist_to_ded_death = smallest_distance_to_dead_end_or_death(current_state, env)
        # diff_distance = dist_to_recovery - dist_to_deadend
        return k*math.exp(-a*dist_to_recovery) - k*math.exp(-a*dist_to_ded_death)
    
    # Bottom right region
    elif current_state[0]>3 and current_state[1]>4:
        dist_to_ded_death = smallest_distance_to_dead_end_or_death(current_state, env)
        return -k*math.exp(-a*dist_to_ded_death) - 0.1


# def recovery_dead_end_balance():



# def bottleneck_three_regions(current_state, env, k, a, b=0, A=0, c=0, mode='basic'):
#     """
#     Calculates the value based on distance to a bottleneck in three regions using various exponential decay functions.

#     Parameters:
#     current_state (tuple): The current state (x, y) coordinates.
#     env (list): The environment (not used in this implementation).
#     k (float): The scaling constant.
#     a (float): The decay constant.
#     b (float): The shift constant (used in shifted mode).
#     A (float): The asymptote value (used in asymptote mode).
#     c (float): The initial offset value (used in offset mode).
#     mode (str): The mode of exponential decay function to use. 
#                 Options are 'basic', 'shifted', 'asymptote', 'offset', 'double'.

#     Returns:
#     float: The calculated value based on the distance.
#     """
#     if current_state[0] < 4:
#         dist_to_bottleneck = manhattan_distance(current_state, [0, 5])
        
#         if mode == 'basic':
#             return k * math.exp(-a * dist_to_bottleneck)
        
#         elif mode == 'shifted':
#             return k * math.exp(-a * (dist_to_bottleneck + b))
        
#         elif mode == 'asymptote':
#             return A + k * math.exp(-a * dist_to_bottleneck)
        
#         elif mode == 'offset':
#             return k * math.exp(-a * dist_to_bottleneck) + c
        
#         elif mode == 'double':
#             k1, a1, k2, a2 = k  # assuming k is a tuple (k1, a1, k2, a2) for double exponential decay
#             return k1 * math.exp(-a1 * dist_to_bottleneck) + k2 * math.exp(-a2 * dist_to_bottleneck)
        
#         else:
#             raise ValueError("Invalid mode. Choose from 'basic', 'shifted', 'asymptote', 'offset', 'double'.")

#     return 0