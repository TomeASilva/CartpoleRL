import numpy as np
import discret_env
from copy import copy, deepcopy


def get_state(vector_state, states_matrix, states_matrix_aux2):
    """
    Converts a vector state of shape (1, num_varialbles) into an integer representing a state

    Arguments:
    vector_state -- vector of shape (1, number of state variables)
    state_matrix -- matrix of shape (number of  states, number of state variables), where number of states
    = k^4, where k is number we want to discretize each state variable into
    step_size -- the difference between two discretized elements of each of the state variables shape (1, number of state variables)

    Returns:
    state -- an integer that represents the sate number of a vector state (1, k^4 -1)


    """
    states_matrix_aux = states_matrix[:, 1:].copy()

    #print("Shape aux2", states_matrix_aux2.shape)
    mask = vector_state >= states_matrix_aux
    #print("Mask1", mask.shape)
    mask2 = vector_state < states_matrix_aux2
    #print("Mask2", mask2.shape)
    #print("Mask1", mask[-1, :])

    C = np.all(mask, axis=1)
    #print("C ", states_matrix[:, 0][C])
    D = np.all(mask2, axis=1)
    #print("D ", states_matrix[:, 0][D])

    # print(states_matrix_aux[0])
    # for i in range(mask.shape[0]):
    #     print(mask[i], mask2[i])
    full_mask = np.c_[mask, mask2]
    #print("full Mask", full_mask.shape)
    full_mask = np.all(full_mask, axis=1)
    #print("full Mask", full_mask.shape)

    # print(states_matrix[full_mask])
    return int(states_matrix[:, 0][full_mask])


if __name__ == "__main__":
    """Testing"""
    vector_state = np.array(
        [-0.01751509,  0.36540166, -0.02045427, -0.60214984]).reshape(1, 4)

    vector_state_2 = np.array(
            [-0.01751509,  0.36540166, -0.02045427, -0.60214984]).reshape(1, 4)
    state_space_matrix, states_matrix_aux2 = discret_env.discretized_space()

    a = get_state(vector_state, state_space_matrix, states_matrix_aux2)
    print(a)
    b = get_state(vector_state_2, state_space_matrix, states_matrix_aux2)
    print(b)
