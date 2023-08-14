import numpy as np
import imageio
from utils import greedy_policy

def record_video(env, Qtable, out_directory, fps=1):
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=np.random.randint(0, 500))
    img = env.render()
    images.append(img)
    
    while not terminated or truncated:
        action = greedy_policy(Qtable, state)
        state, _, terminated, truncated, _ = env.step(action)
        img = env.render()
        images.append(img)
        
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)
