import gym
import numpy as np
import get_state as get
import discret_env
from copy import copy, deepcopy

env = gym.make('CartPole-v0')
env.reset()

state_space_matrix = np.genfromtxt('states_matrix.csv', delimiter=",")
state_space_aux = np.genfromtxt('states_matrix_aux.csv', delimiter=",")
# Important constants
num_States = state_space_matrix.shape[0]
print(num_States)

num_Actions = 2

# Variable for counting the number of State- Action - State visits
state_transition_count = np.zeros(
    (num_States, num_Actions, num_States), dtype=np.float32)

# initialize vector with state rewards
state_rewards_record = np.zeros((num_States, 1))
state_rewards_count = np.zeros((num_States, 1))
state_rewards = np.zeros((num_States, 1))

# Initiliazing state_transition probabilities as a uniform distribution

state_transition_probs = np.ones(
    (num_States, num_Actions, num_States), dtype=np.float32)/num_States

for i_epsisode in range(20):
    observation = env.reset()  # gives th vector that represents the state of the enviroment

    new_state = get.get_state(observation, state_space_matrix, state_space_aux)

    for t in range(100):  # at maximum I will let my simulation run for 100 decision epochs

        state = new_state
        print("1 Estado ", state)
        env.render()

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        new_state = get.get_state(
            observation, state_space_matrix, state_space_aux)

        print("2 Estado ", new_state)

        # Count the state transition
        state_transition_count[state, action, new_state] += 1

        if done:
            state_rewards_count[new_state] += 1
            state_rewards_record[new_state] += -1

            # print("Termination State Reward ", reward)
            print("Episode finished {} timestamps ". format(t+1))

            break

    # Update state transition probabilities taking into account only the action pairs observed

    total_state_transition_count = np.sum(state_transition_count, axis=2)
    mask = total_state_transition_count > 0

    state_transition_probs[mask] = state_transition_count[mask] / \
        total_state_transition_count[mask].reshape(-1, 1)

    # Update reward vector
    print(state_rewards_count.shape)
    print(state_rewards_record.shape)
    print(state_rewards.shape)
    mask0 = state_rewards_count > 0
    state_rewards[mask0] = state_rewards_record[mask0] / \
        state_rewards_count[mask0]
    exit()
    # Run Value iteration after every episode
