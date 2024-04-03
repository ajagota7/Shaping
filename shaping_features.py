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

# def recovery_dead_end_balance():