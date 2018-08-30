import gym
import numpy as np
import get_state as get
import discret_env
import matplotlib.pyplot as plt
from copy import copy, deepcopy


env = gym.make('CartPole-v0')
env.reset()

# Import a discritization of the state space computed by the discrete_env module

state_space_matrix = np.genfromtxt('states_matrix.csv', delimiter=",")
state_space_aux = np.genfromtxt('states_matrix_aux.csv', delimiter=",")


# Important constants
num_States = state_space_matrix.shape[0]
episodes_without_improv = 0
# stops to run episodes if 20 times in a Row value iteration converged on the first pass
No_Improvement_Treshold = 20
num_Actions = 2
i_epsisode = 0

######################Book keeping####################

time_steps_booking = []


# Variable for counting the number of State- Action - State visits
state_transition_count = np.zeros(
    (num_States, num_Actions, num_States), dtype=np.float32)

# initialize vector with state rewards
state_rewards_record = np.zeros((num_States, 1))
state_rewards_count = np.zeros((num_States, 1))
state_rewards = np.zeros((num_States, 1))

# initialize the states value vector

states_value = np.random.rand(num_States, 1) * 0.1
# Initiliazing state_transition probabilities as a uniform distribution

state_transition_probs = np.ones(
    (num_States, num_Actions, num_States), dtype=np.float32)/num_States

while episodes_without_improv < No_Improvement_Treshold and i_epsisode < 100:
    observation = env.reset()  # gives a vector that represents the state of the enviroment

    new_state = get.get_state(observation, state_space_matrix, state_space_aux)

    for t in range(1000):  # at maximum I will let my simulation run for t steps

        state = new_state
        # print("1 Estado ", state)
        # if i_epsisode % 10 == 0:
        #     env.render()

        action = int(np.argmax(
            np.dot(state_transition_probs[state], states_value), axis=0))

        # action = np.dot(state_transition_probs[state], states_value).shape

        # print("Action ", type(action))
        # print("Action ", action)

        # exit()
        observation, reward, done, info = env.step(action)

        new_state = get.get_state(
            observation, state_space_matrix, state_space_aux)

        # print("2 Estado ", new_state)

        # Count the state transition
        state_transition_count[state, action, new_state] += 1

        if done:
            if t < 199:  # The simulation is cut when when we achieve 200 steps therefore I don't want to penalize when that happens
                state_rewards_count[new_state] += 1
                state_rewards_record[new_state] += -1

            i_epsisode += 1
            # print("Termination State Reward ", reward)
            # Record how many step were achieved
            time_steps_booking.append(t+1)
            print("Episode {} finished in {} timestamps ". format(i_epsisode, t+1))

            break

    # Update state transition probabilities taking into account only the action pairs observed on the episode

    total_state_transition_count = np.sum(state_transition_count, axis=2)
    mask = total_state_transition_count > 0

    state_transition_probs[mask] = state_transition_count[mask] / \
        total_state_transition_count[mask].reshape(-1, 1)

    # Update reward vector

    mask0 = state_rewards_count > 0
    state_rewards[mask0] = state_rewards_record[mask0] / \
        state_rewards_count[mask0]

    # Run Value iteration after every episode
    #-----------------------------------------

    DISCOUNT_FACTOR = 0.8
    convergence = 1
    num_iters = 0

    while convergence >= 0.001:

        previous_states_values = states_value
        states_value = state_rewards + \
            np.max(DISCOUNT_FACTOR * np.dot(state_transition_probs,
                                            previous_states_values), axis=1)
        convergence = max(np.abs(previous_states_values-states_value))

        num_iters += 1

    if num_iters == 1:
        episodes_without_improv += 1

    else:
        episodes_without_improv = 0


# Graphing Algorithm's performance
#----------------------------------------------------

fig, ax = plt.subplots()
ax.plot(time_steps_booking)
ax.set(ylabel='#step ', xlabel='# episode', title="Learning phase")
fig.show()


#----------------------------------------------------


# Testing the learnt policy
time_steps_booking_test = []


for i in range(100):
    observation = env.reset()  # gives a vector that represents the state of the enviroment

    new_state = get.get_state(observation, state_space_matrix, state_space_aux)

    for t in range(1000):  # at maximum I will let my simulation run for t steps

        state = new_state
        # print("1 Estado ", state)

        if i <= 10:  # show first 10 tests of the learn policy
            env.render()

        action = int(np.argmax(
            np.dot(state_transition_probs[state], states_value), axis=0))

        # action = np.dot(state_transition_probs[state], states_value).shape

        # print("Action ", type(action))
        # print("Action ", action)

        # exit()
        observation, reward, done, info = env.step(action)

        new_state = get.get_state(
            observation, state_space_matrix, state_space_aux)

        # print("2 Estado ", new_state)

        # Count the state transition
        state_transition_count[state, action, new_state] += 1

        if done:

            # print("Termination State Reward ", reward)
            time_steps_booking_test.append(t+1)
            print("Episode {} finished in {} timestamps ". format(i, t+1))

            break

fig1, ax1 = plt.subplots()
ax1.plot(time_steps_booking_test)
ax1.set(ylabel='#step ', xlabel='# episode', title="Testing Phase phase")
plt.show()
