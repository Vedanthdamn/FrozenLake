import numpy as np
import matplotlib.pyplot as plt

rewards_per_episode = np.load("rewards_per_episode_8x8.npy")

window_size = 100
rolling_average = np.convolve(rewards_per_episode, np.ones(window_size) / window_size, mode="valid")

plt.figure(figsize=(10, 6))
plt.plot(rolling_average)
plt.xlabel("Episode")
plt.ylabel(f"Success rate (rolling average over {window_size} episodes)")
plt.title("Q-learning training progress on FrozenLake-v1 (8x8)")
plt.savefig("training_curve_8x8.png")

print("Saved training_curve_8x8.png")
