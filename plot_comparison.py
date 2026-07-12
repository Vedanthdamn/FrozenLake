import numpy as np
import matplotlib.pyplot as plt

from frozenlake_common import rewards_path

window_size = 100
configs = [
    ("4x4", True, "4x4 slippery"),
    ("4x4", False, "4x4 deterministic"),
    ("8x8", True, "8x8 slippery"),
    ("8x8", False, "8x8 deterministic"),
]

plt.figure(figsize=(10, 6))

for map_name, is_slippery, label in configs:
    rewards_per_episode = np.load(rewards_path(map_name, is_slippery))
    rolling_average = np.convolve(rewards_per_episode, np.ones(window_size) / window_size, mode="valid")
    episodes = np.arange(1, len(rolling_average) + 1)
    plt.plot(episodes, rolling_average, label=label)

plt.xscale("log")
plt.xlabel("Episode (log scale)")
plt.ylabel(f"Success rate (rolling average over {window_size} episodes)")
plt.title("Q-learning training progress across all four configurations")
plt.legend()
plt.savefig("training_curve_comparison.png")

print("Saved training_curve_comparison.png")
