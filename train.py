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
    n_training_episodes = 10000     # total training episode
    learning_rate = 0.7             # learning rate

    n_eval_episodes = 100           # total test episode

    env_id = "FrozenLake-v1"        # environment name
    max_steps = 99                  # max step per episode
    gamma = 0.95                    # discount rate
    eval_seed = []                  # eval seed environment

    max_epsilon = 1.0               # exploration probability at start
    min_epsilon = 0.05              # minimum exploration probability
    decay_rate = 0.0005             # exponential decay rate for exploration probability

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")

    state_space = env.observation_space.n
    action_space = env.action_space.n

    Qtable_frozenlake = initialize_q_table(state_space, action_space)

    Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)
