import numpy as np
import gym
from copy import copy, deepcopy


def discretized_space():
    """
    Maps state vector into a state number

    Parameters:
    K-- the number of blocks we want to discretize each one of the variables of state vector

    Saves into two files:
    states_matrix -- a matrix with shape (k^4, 5) where the first column represents the number of the state
    states_matrix_aux -- a matrix with shape (k^4, 5) which represents the bound of each state from states_matrix

    Example:

    The first row of 'states_matrix' represents the values of the variables of the state for state number 1 and the 
    first row of 'states_matrix_aux' represents the the bound for those variables for the state number 1
    I.e. states_matrix establishes where a state begins and states_matrix_aux where that state ends. If an observed state falls
    within that interval then it belongs to state 1 
    """

    K = 10
    env = gym.make('CartPole-v0')

    high_bound = env.observation_space.high.copy()

    low_bound = env.observation_space.low.copy()

    v_len = low_bound.shape[0]

    space_range = high_bound - low_bound

    for i in range(v_len):
        if np.isinf(space_range[i]):
            space_range[i] = space_range[i-1]
            high_bound[i] = high_bound[i-1]
            low_bound[i] = low_bound[i-1]

    steps = space_range/10
    x = np.arange(low_bound[0], high_bound[0] + steps[0], steps[1])
    y = np.concatenate(([env.observation_space.low[1] - steps[1]], np.arange(low_bound[1],
                                                                             high_bound[1] + steps[1], steps[1]), [env.observation_space.high[1] + steps[1]]))
    z = np.arange(low_bound[2], high_bound[2] + steps[2], steps[2])
    w = np.concatenate(([env.observation_space.low[3] - steps[3]], np.arange(low_bound[3],
                                                                             high_bound[3] + steps[3], steps[3]), [env.observation_space.high[3] + steps[3]]))

    xx, yy, zz, ww = np.meshgrid(x, y, z, w)

    states_matrix = np.c_[xx.ravel(), yy.ravel(), zz.ravel(), ww.ravel()]

    a = []
    [a.append(i) for i, j in enumerate(states_matrix)]
    a = np.array(a, int).reshape((-1, 1))
    states_matrix = np.c_[a, states_matrix]


    states_matrix_aux = states_matrix[:, 1:].copy()

    list_ = []
    for i in states_matrix_aux:
        mask = states_matrix_aux > i
        full_mask = np.all(mask, axis=1)

        if True in full_mask:
            state = states_matrix_aux[full_mask][0, :]
            list_.append(state)
        else:
            state_list = []

            for j in range(states_matrix_aux.shape[1]):
                t = states_matrix_aux[:, j] > i[j]

                if True not in t:
                    state_list.append(i[j])
                else:
                    state_list.append(states_matrix_aux[t][0, j])

            state = np.array(state_list)
            list_.append(state)

    states_matrix_aux2 = np.array(list_)

    np.savetxt("states_matrix.csv", states_matrix, delimiter= ",")
    np.savetxt("states_matrix_aux.csv", states_matrix_aux2, delimiter= ",")
    

if __name__ == '__main__':
    discretized_space()
    #print(_)