import numpy as np
from tqdm.notebook import tqdm
from utils import initialize_q_table, epsilon_greedy_policy
import gym

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state = env.reset()
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            if terminated or truncated:
                break

            state = new_state

    return Qtable

if __name__ == "__main__":
    # Your training configuration and Q-table initialization here
    # ...

    Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)
