import argparse
import numpy as np
import gymnasium as gym

from frozenlake_common import EVAL_EPISODES, MAX_STEPS_PER_EPISODE, q_table_path

parser = argparse.ArgumentParser()
parser.add_argument("--map", choices=["4x4", "8x8"], required=True)
parser.add_argument("--slippery", action=argparse.BooleanOptionalAction, required=True)
args = parser.parse_args()

map_name = args.map
is_slippery = args.slippery
env_name = "FrozenLake-v1"

num_eval_episodes = EVAL_EPISODES[(map_name, is_slippery)]
q_table_file = q_table_path(map_name, is_slippery)

env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)

def run_episodes(policy_fn, num_episodes):
    success_count = 0
    step_counts = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        steps = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = policy_fn(state)
            state, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

            if done:
                if reward == 1:
                    success_count += 1
                break

        step_counts.append(steps)

    success_rate = success_count / num_episodes * 100
    average_steps = np.mean(step_counts)
    return success_rate, average_steps

q_table = np.load(q_table_file)

def trained_policy(state):
    return np.argmax(q_table[state, :])

def random_policy(state):
    return env.action_space.sample()

trained_success_rate, trained_average_steps = run_episodes(trained_policy, num_eval_episodes)
random_success_rate, random_average_steps = run_episodes(random_policy, num_eval_episodes)

env.close()

print(f"Trained agent over {num_eval_episodes} episodes:")
print(f"  success rate: {trained_success_rate:.1f}%")
print(f"  average steps per episode: {trained_average_steps:.1f}")
print()
print(f"Random policy baseline over {num_eval_episodes} episodes:")
print(f"  success rate: {random_success_rate:.1f}%")
print(f"  average steps per episode: {random_average_steps:.1f}")
