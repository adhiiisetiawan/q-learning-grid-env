import numpy as np
import imageio
from utils import greedy_policy
import gym

def record_video(env, Qtable, out_directory, fps=1):
    images = []
    terminated = False
    truncated = False
    state = env.reset(seed=np.random.randint(0, 500))
    img = env.render()
    images.append(img)
    
    while not terminated or truncated:
        action = greedy_policy(Qtable, state)
        state, _, terminated, truncated, _ = env.step(action)
        img = env.render()
        images.append(img)
        
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

if __name__ == "__main__":
    # Your recording configuration and Q-table loading here
    # ...

    record_video(env, Qtable_frozenlake, "/home/adhi/reinforcement-learning-exploration/huggingface/unit2/replay.mp4", fps=1)
