import numpy as np
import gymnasium as gym
import imageio

env_name = "FrozenLake-v1"
map_name = "8x8"
is_slippery = True

num_demo_episodes = 3
max_steps_per_episode = 100
max_attempts_per_episode = 500

env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery, render_mode="rgb_array")

q_table = np.load("q_table_8x8.npy")

frames = []

for episode in range(num_demo_episodes):
    reached_goal = False

    for attempt in range(max_attempts_per_episode):
        episode_frames = []
        state, info = env.reset()
        episode_frames.append(env.render())

        for step in range(max_steps_per_episode):
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
    imageio.mimsave("demo_8x8.gif", frames, duration=0.4)
    print("Saved demo_8x8.gif")
else:
    print("No successful episodes captured, demo_8x8.gif not saved")
