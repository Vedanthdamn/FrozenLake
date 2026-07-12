import numpy as np
import gymnasium as gym
import imageio

env_name = "FrozenLake-v1"
map_name = "8x8"
is_slippery = False

num_demo_episodes = 3
max_steps_per_episode = 100
max_attempts_per_episode = 50

env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery, render_mode="rgb_array")

q_table = np.load("q_table_8x8_deterministic.npy")

frames = []

for episode in range(num_demo_episodes):
    for attempt in range(max_attempts_per_episode):
        episode_frames = []
        state, info = env.reset()
        episode_frames.append(env.render())
        reached_goal = False

        for step in range(max_steps_per_episode):
            action = np.argmax(q_table[state, :])
            state, reward, terminated, truncated, info = env.step(action)
            episode_frames.append(env.render())

            if terminated or truncated:
                reached_goal = reward == 1
                break

        if reached_goal:
            frames.extend(episode_frames)
            break

env.close()

imageio.mimsave("demo_8x8_deterministic.gif", frames, duration=0.4)
print("Saved demo_8x8_deterministic.gif")
