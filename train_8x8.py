import time
import numpy as np
import gymnasium as gym

env_name = "FrozenLake-v1"
map_name = "8x8"
is_slippery = True
max_num_episodes = 3000000
num_envs = 128

start_learning_rate = 0.1
end_learning_rate = 0.01
lr_decay_episodes = 1500000
discount_rate = 0.99
optimistic_init_value = 1.0

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay_rate = 0.000003

checkpoint_interval = 5000
greedy_eval_interval = 50000
greedy_eval_episodes = 2000
smoothing_window_evals = 4
plateau_window_evals = 6
plateau_threshold = 1.5
min_episodes_before_plateau_check = 1200000

envs = gym.make_vec(env_name, num_envs=num_envs, vectorization_mode="sync", map_name=map_name, is_slippery=is_slippery)
num_states = envs.single_observation_space.n
num_actions = envs.single_action_space.n
q_table = np.full((num_states, num_actions), optimistic_init_value)

eval_env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)

def greedy_eval(num_episodes):
    success_count = 0
    for episode in range(num_episodes):
        state, info = eval_env.reset()
        for step in range(100):
            action = np.argmax(q_table[state, :])
            state, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                success_count += reward == 1
                break
    return success_count / num_episodes * 100

states, infos = envs.reset()
needs_reset = np.zeros(num_envs, dtype=bool)

episode_outcomes = []
greedy_eval_success_rates = []
episodes_completed = 0
next_greedy_eval_at = greedy_eval_interval
start_time = time.time()
training_finished = False

while episodes_completed < max_num_episodes:
    random_actions = np.random.randint(0, num_actions, size=num_envs)
    greedy_actions = np.argmax(q_table[states], axis=1)
    explore_mask = np.random.uniform(size=num_envs) < epsilon
    actions = np.where(explore_mask, random_actions, greedy_actions)

    next_states, rewards, terminations, truncations, infos = envs.step(actions)
    dones = terminations | truncations

    valid = ~needs_reset
    valid_idx = np.nonzero(valid)[0]

    current_lr = start_learning_rate - (start_learning_rate - end_learning_rate) * min(
        episodes_completed / lr_decay_episodes, 1.0
    )

    s = states[valid_idx]
    a = actions[valid_idx]
    r = rewards[valid_idx]
    ns = next_states[valid_idx]
    not_done = ~dones[valid_idx]
    best_next_values = np.max(q_table[ns], axis=1) * not_done
    td_errors = r + discount_rate * best_next_values - q_table[s, a]
    np.add.at(q_table, (s, a), current_lr * td_errors)

    completed_idx = valid_idx[dones[valid_idx]]
    for idx in completed_idx:
        episode_outcomes.append(rewards[idx])
        episodes_completed += 1

        if episodes_completed % checkpoint_interval == 0:
            recent_success_rate = np.mean(episode_outcomes[-checkpoint_interval:]) * 100
            print(
                f"Episode {episodes_completed}/{max_num_episodes} - "
                f"training rollout success last {checkpoint_interval}: {recent_success_rate:.1f}% - "
                f"epsilon: {epsilon:.4f} - lr: {current_lr:.4f}"
            )

        if episodes_completed >= next_greedy_eval_at:
            greedy_success_rate = greedy_eval(greedy_eval_episodes)
            greedy_eval_success_rates.append(greedy_success_rate)
            smoothed_success_rate = np.mean(greedy_eval_success_rates[-smoothing_window_evals:])
            print(
                f"  greedy eval at {episodes_completed}: {greedy_success_rate:.1f}% success over "
                f"{greedy_eval_episodes} episodes (smoothed: {smoothed_success_rate:.1f}%)"
            )
            next_greedy_eval_at += greedy_eval_interval

            if (
                len(greedy_eval_success_rates) > plateau_window_evals + smoothing_window_evals
                and episodes_completed >= min_episodes_before_plateau_check
            ):
                previous_smoothed_rate = np.mean(
                    greedy_eval_success_rates[-plateau_window_evals - smoothing_window_evals : -plateau_window_evals]
                )
                improvement = smoothed_success_rate - previous_smoothed_rate
                if improvement < plateau_threshold:
                    print(
                        f"Smoothed greedy eval success rate plateaued (+{improvement:.1f} points over last "
                        f"{plateau_window_evals * greedy_eval_interval} episodes), stopping training"
                    )
                    training_finished = True
                    break

    if training_finished:
        break

    needs_reset = valid & dones
    states = next_states

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episodes_completed)

envs.close()
eval_env.close()

elapsed_time = time.time() - start_time

np.save("q_table_8x8.npy", q_table)
np.save("rewards_per_episode_8x8.npy", np.array(episode_outcomes))

print(f"Training finished after {len(episode_outcomes)} episodes in {elapsed_time:.1f} seconds")
print("Q-table saved to q_table_8x8.npy")
