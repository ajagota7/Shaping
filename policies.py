import numpy as np

P_pi_b_good = np.zeros((10, 10, 5))

P_pi_b_good[2,9,3] = 1
P_pi_b_good[1,9,3] = 1
P_pi_b_good[0,9,1] = 1
P_pi_b_good[0,8,1] = 1
P_pi_b_good[0,7,1] = 1
P_pi_b_good[0,6,1] = 1
P_pi_b_good[0,5,1] = 1
P_pi_b_good[0,4,4] = 1
P_pi_b_good[1,4,4] = 1
P_pi_b_good[2,4,4] = 1
P_pi_b_good[3,4,4] = 1
P_pi_b_good[4,4,4] = 1
P_pi_b_good[5,4,1] = 1
P_pi_b_good[5,5,1] = 1
P_pi_b_good[5,4,1] = 1
P_pi_b_good[5,3,1] = 1
P_pi_b_good[5,2,1] = 1
P_pi_b_good[5,1,1] = 1



P_pi_b_good_stoch = np.zeros((10, 10, 5))

P_pi_b_good_stoch[2,9,3] = 0.5
P_pi_b_good_stoch[2,9,1] = 0.5

P_pi_b_good_stoch[2,8,3] = 0.5
P_pi_b_good_stoch[2,8,1] = 0.5

P_pi_b_good_stoch[2,7,3] = 0.5
P_pi_b_good_stoch[2,7,1] = 0.5

P_pi_b_good_stoch[1,9,3] = 0.5
P_pi_b_good_stoch[1,9,1] = 0.5

P_pi_b_good_stoch[1,8,3] = 0.5
P_pi_b_good_stoch[1,8,1] = 0.5

P_pi_b_good_stoch[1,7,3] = 0.5
P_pi_b_good_stoch[1,7,1] = 0.5

P_pi_b_good_stoch[2,6,3] = 1
P_pi_b_good_stoch[1,6,3] = 1
P_pi_b_good_stoch[1,5,3] = 1


P_pi_b_good_stoch[0,9,1] = 1
P_pi_b_good_stoch[0,8,1] = 1
P_pi_b_good_stoch[0,7,1] = 1
P_pi_b_good_stoch[0,6,1] = 1
P_pi_b_good_stoch[0,5,1] = 1

P_pi_b_good_stoch[0,4,1] = 0.5
P_pi_b_good_stoch[0,4,4] = 0.5
P_pi_b_good_stoch[0,4,1] = 0.5
P_pi_b_good_stoch[0,4,4] = 0.5
P_pi_b_good_stoch[0,3,1] = 0.5
P_pi_b_good_stoch[0,3,4] = 0.5
P_pi_b_good_stoch[0,2,1] = 0.5
P_pi_b_good_stoch[0,2,4] = 0.5

P_pi_b_good_stoch[1,4,1] = 0.5
P_pi_b_good_stoch[1,4,4] = 0.5
P_pi_b_good_stoch[1,4,1] = 0.5
P_pi_b_good_stoch[1,4,4] = 0.5
P_pi_b_good_stoch[1,3,1] = 0.5
P_pi_b_good_stoch[1,3,4] = 0.5
P_pi_b_good_stoch[1,2,1] = 0.5
P_pi_b_good_stoch[1,2,4] = 0.5

P_pi_b_good_stoch[2,4,1] = 0.5
P_pi_b_good_stoch[2,4,4] = 0.5
P_pi_b_good_stoch[2,4,1] = 0.5
P_pi_b_good_stoch[2,4,4] = 0.5
P_pi_b_good_stoch[2,3,1] = 0.5
P_pi_b_good_stoch[2,3,4] = 0.5
P_pi_b_good_stoch[2,2,1] = 0.5
P_pi_b_good_stoch[2,2,4] = 0.5

P_pi_b_good_stoch[3,4,1] = 0.5
P_pi_b_good_stoch[3,4,4] = 0.5
P_pi_b_good_stoch[3,4,1] = 0.5
P_pi_b_good_stoch[3,4,4] = 0.5
P_pi_b_good_stoch[3,3,1] = 0.5
P_pi_b_good_stoch[3,3,4] = 0.5
P_pi_b_good_stoch[3,2,1] = 0.5
P_pi_b_good_stoch[3,2,4] = 0.5

P_pi_b_good_stoch[4,4,1] = 0.5
P_pi_b_good_stoch[4,4,4] = 0.5
P_pi_b_good_stoch[4,4,1] = 0.5
P_pi_b_good_stoch[4,4,4] = 0.5
P_pi_b_good_stoch[4,3,1] = 0.5
P_pi_b_good_stoch[4,3,4] = 0.5
P_pi_b_good_stoch[4,2,1] = 0.5
P_pi_b_good_stoch[4,2,4] = 0.5

P_pi_b_good_stoch[0,1,4] = 1
P_pi_b_good_stoch[1,1,4] = 1
P_pi_b_good_stoch[2,1,4] = 1
P_pi_b_good_stoch[3,1,4] = 1
P_pi_b_good_stoch[4,1,4] = 1

P_pi_b_good_stoch[5,4,1] = 1
P_pi_b_good_stoch[5,3,1] = 1
P_pi_b_good_stoch[5,2,1] = 1
P_pi_b_good_stoch[5,1,1] = 1

