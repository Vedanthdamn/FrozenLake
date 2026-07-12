import argparse
import numpy as np
import matplotlib.pyplot as plt

from frozenlake_common import rewards_path, curve_path

parser = argparse.ArgumentParser()
parser.add_argument("--map", choices=["4x4", "8x8"], required=True)
parser.add_argument("--slippery", action=argparse.BooleanOptionalAction, required=True)
args = parser.parse_args()

map_name = args.map
is_slippery = args.slippery

rewards_file = rewards_path(map_name, is_slippery)
curve_file = curve_path(map_name, is_slippery)

rewards_per_episode = np.load(rewards_file)

window_size = 100
rolling_average = np.convolve(rewards_per_episode, np.ones(window_size) / window_size, mode="valid")

slippery_label = "slippery" if is_slippery else "deterministic"

plt.figure(figsize=(10, 6))
plt.plot(rolling_average)
plt.xlabel("Episode")
plt.ylabel(f"Success rate (rolling average over {window_size} episodes)")
plt.title(f"Q-learning training progress on FrozenLake-v1 ({map_name}, {slippery_label})")
plt.savefig(curve_file)

print(f"Saved {curve_file}")
