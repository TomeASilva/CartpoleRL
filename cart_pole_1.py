import gym
import numpy as np
import get_state
import discret_env


env = gym.make('CartPole-v0')
env.reset()

state_space_matrix = np.genfromtxt('states_matrix.csv', delimiter=",")
state_space_aux = np.genfromtxt('states_matrix_aux.csv', delimiter=",")


for i_epsisode in range(20):
    observation = env.reset()  # gives th vector that represents the state of the enviroment
    for t in range(100):  # at maximum I will let my simulation run for 100 decision epochs
        env.render()

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        print(observation)
        print(get_state.get_state(observation, state_space_matrix, state_space_aux))
        if done:
            print("Episode finished {} timestamps ". format(t+1))
            break
