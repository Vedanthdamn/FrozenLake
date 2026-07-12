import numpy as np
import gymnasium as gym

env_name = "FrozenLake-v1"
map_name = "4x4"
is_slippery = False

num_episodes = 5000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay_rate = 0.001

env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)

num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

rewards_per_episode = []

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        best_next_value = np.max(q_table[next_state, :])
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_rate * best_next_value - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 500 == 0:
        recent_success_rate = np.mean(rewards_per_episode[-500:]) * 100
        print(f"Episode {episode + 1}/{num_episodes} - success rate last 500 episodes: {recent_success_rate:.1f}% - epsilon: {epsilon:.3f}")

env.close()

np.save("q_table_4x4_deterministic.npy", q_table)
np.save("rewards_per_episode_4x4_deterministic.npy", np.array(rewards_per_episode))

print("Training finished, Q-table saved to q_table_4x4_deterministic.npy")
