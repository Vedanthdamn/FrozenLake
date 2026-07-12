import argparse
import time
import numpy as np
import gymnasium as gym

from frozenlake_common import TRAIN_CONFIGS, MAX_STEPS_PER_EPISODE, q_table_path, rewards_path
from q_learning import choose_action, update_q_table, decay_epsilon

parser = argparse.ArgumentParser()
parser.add_argument("--map", choices=["4x4", "8x8"], required=True)
parser.add_argument("--slippery", action=argparse.BooleanOptionalAction, required=True)
args = parser.parse_args()

map_name = args.map
is_slippery = args.slippery
env_name = "FrozenLake-v1"
config = TRAIN_CONFIGS[(map_name, is_slippery)]

q_table_file = q_table_path(map_name, is_slippery)
rewards_file = rewards_path(map_name, is_slippery)

if config["approach"] == "simple":
    num_episodes = config["num_episodes"]
    learning_rate = config["learning_rate"]
    discount_rate = config["discount_rate"]
    max_epsilon = 1.0
    min_epsilon = config["min_epsilon"]
    epsilon_decay_rate = config["epsilon_decay_rate"]
    checkpoint_interval = config["checkpoint_interval"]

    env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))

    epsilon = max_epsilon
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = choose_action(q_table, state, epsilon, num_actions)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            update_q_table(q_table, state, action, reward, next_state, done, learning_rate, discount_rate)

            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = decay_epsilon(episode, max_epsilon, min_epsilon, epsilon_decay_rate)
        rewards_per_episode.append(total_reward)

        if (episode + 1) % checkpoint_interval == 0:
            recent_success_rate = np.mean(rewards_per_episode[-checkpoint_interval:]) * 100
            print(
                f"Episode {episode + 1}/{num_episodes} - "
                f"success rate last {checkpoint_interval}: {recent_success_rate:.1f}% - epsilon: {epsilon:.3f}"
            )

    env.close()

    np.save(q_table_file, q_table)
    np.save(rewards_file, np.array(rewards_per_episode))

    print(f"Training finished, Q-table saved to {q_table_file}")

else:
    max_num_episodes = config["max_num_episodes"]
    num_envs = config["num_envs"]
    start_learning_rate = config["start_learning_rate"]
    end_learning_rate = config["end_learning_rate"]
    lr_decay_episodes = config["lr_decay_episodes"]
    discount_rate = config["discount_rate"]
    optimistic_init_value = config["optimistic_init_value"]
    max_epsilon = 1.0
    min_epsilon = config["min_epsilon"]
    epsilon_decay_rate = config["epsilon_decay_rate"]
    checkpoint_interval = config["checkpoint_interval"]
    greedy_eval_interval = config["greedy_eval_interval"]
    greedy_eval_episodes = config["greedy_eval_episodes"]
    smoothing_window_evals = config["smoothing_window_evals"]
    plateau_window_evals = config["plateau_window_evals"]
    plateau_threshold = config["plateau_threshold"]
    min_episodes_before_plateau_check = config["min_episodes_before_plateau_check"]

    envs = gym.make_vec(env_name, num_envs=num_envs, vectorization_mode="sync", map_name=map_name, is_slippery=is_slippery)
    num_states = envs.single_observation_space.n
    num_actions = envs.single_action_space.n
    q_table = np.full((num_states, num_actions), optimistic_init_value)

    eval_env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)

    def greedy_eval(num_episodes):
        success_count = 0
        for episode in range(num_episodes):
            state, info = eval_env.reset()
            for step in range(MAX_STEPS_PER_EPISODE):
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
    epsilon = max_epsilon

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

        epsilon = decay_epsilon(episodes_completed, max_epsilon, min_epsilon, epsilon_decay_rate)

    envs.close()
    eval_env.close()

    elapsed_time = time.time() - start_time

    np.save(q_table_file, q_table)
    np.save(rewards_file, np.array(episode_outcomes))

    print(f"Training finished after {len(episode_outcomes)} episodes in {elapsed_time:.1f} seconds")
    print(f"Q-table saved to {q_table_file}")
