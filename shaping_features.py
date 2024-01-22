import math

def euclidean_distance(state1, state2):
    return math.sqrt((state1[0] - state2[0]) ** 2 + (state1[1] - state2[1]) ** 2)

def manhattan_distance(state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

def smallest_distance_to_deadend(current_state, env):

    return min(manhattan_distance(current_state, dead_end) for dead_end in env.dead_ends)