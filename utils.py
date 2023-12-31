import numpy as np
import random

def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state][:])
    return action

def epsilon_greedy_policy(Qtable, env, state, epsilon):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(Qtable, state)
    else:
        action = env.action_space.sample()

    return action
