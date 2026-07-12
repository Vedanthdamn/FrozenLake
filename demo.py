import argparse
import numpy as np
import gymnasium as gym
import imageio

from frozenlake_common import DEMO_MAX_ATTEMPTS, MAX_STEPS_PER_EPISODE, q_table_path, demo_path

parser = argparse.ArgumentParser()
parser.add_argument("--map", choices=["4x4", "8x8"], required=True)
parser.add_argument("--slippery", action=argparse.BooleanOptionalAction, required=True)
args = parser.parse_args()

map_name = args.map
is_slippery = args.slippery
env_name = "FrozenLake-v1"

num_demo_episodes = 3
max_attempts_per_episode = DEMO_MAX_ATTEMPTS[(map_name, is_slippery)]
q_table_file = q_table_path(map_name, is_slippery)
demo_file = demo_path(map_name, is_slippery)

env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery, render_mode="rgb_array")

q_table = np.load(q_table_file)

frames = []

for episode in range(num_demo_episodes):
    reached_goal = False

    for attempt in range(max_attempts_per_episode):
        episode_frames = []
        state, info = env.reset()
        episode_frames.append(env.render())

        for step in range(MAX_STEPS_PER_EPISODE):
            action = np.argmax(q_table[state, :])
            state, reward, terminated, truncated, info = env.step(action)
            episode_frames.append(env.render())

            if terminated or truncated:
                reached_goal = reward == 1
                break

        if reached_goal:
            frames.extend(episode_frames)
            print(f"Episode {episode + 1}: reached the goal after {attempt + 1} attempt(s)")
            break

    if not reached_goal:
        print(f"Episode {episode + 1}: did not reach the goal in {max_attempts_per_episode} attempts, skipping")

env.close()

if frames:
    imageio.mimsave(demo_file, frames, duration=0.4)
    print(f"Saved {demo_file}")
else:
    print(f"No successful episodes captured, {demo_file} not saved")
