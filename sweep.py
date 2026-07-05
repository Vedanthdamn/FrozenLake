import time
import numpy as np
import gymnasium as gym

env_name = "FrozenLake-v1"
map_name = "4x4"
is_slippery = True
max_steps_per_episode = 100
num_eval_episodes = 1000

env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)
num_states = env.observation_space.n
num_actions = env.action_space.n


def make_learning_rate_fn(learning_rate):
    if learning_rate == "decay":
        start_lr = 0.1
        end_lr = 0.01

        def learning_rate_fn(episode, num_episodes):
            return start_lr - (start_lr - end_lr) * episode / num_episodes

        return learning_rate_fn, "decay(0.1->0.01)"

    def learning_rate_fn(episode, num_episodes):
        return learning_rate

    return learning_rate_fn, str(learning_rate)


def train_q_table(num_episodes, gamma, epsilon_decay, epsilon_min, learning_rate):
    learning_rate_fn, _ = make_learning_rate_fn(learning_rate)
    q_table = np.zeros((num_states, num_actions))
    epsilon = 1.0

    for episode in range(num_episodes):
        state, info = env.reset()
        current_lr = learning_rate_fn(episode, num_episodes)

        for step in range(max_steps_per_episode):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, terminated, truncated, info = env.step(action)
            best_next_value = np.max(q_table[next_state, :])
            q_table[state, action] += current_lr * (
                reward + gamma * best_next_value - q_table[state, action]
            )
            state = next_state

            if terminated or truncated:
                break

        epsilon = epsilon_min + (1.0 - epsilon_min) * np.exp(-epsilon_decay * episode)

    return q_table


def evaluate_q_table(q_table):
    success_count = 0
    step_counts = []

    for episode in range(num_eval_episodes):
        state, info = env.reset()
        for step in range(max_steps_per_episode):
            action = np.argmax(q_table[state, :])
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if reward == 1:
                    success_count += 1
                step_counts.append(step + 1)
                break

    success_rate = success_count / num_eval_episodes * 100
    average_steps = np.mean(step_counts)
    return success_rate, average_steps


def run_combo(num_episodes, gamma, epsilon_decay, epsilon_min, learning_rate):
    _, lr_label = make_learning_rate_fn(learning_rate)
    start_time = time.time()
    q_table = train_q_table(num_episodes, gamma, epsilon_decay, epsilon_min, learning_rate)
    success_rate, average_steps = evaluate_q_table(q_table)
    elapsed = time.time() - start_time

    result = {
        "episodes": num_episodes,
        "learning_rate": lr_label,
        "learning_rate_value": learning_rate,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min,
        "success_rate": success_rate,
        "average_steps": average_steps,
    }

    print(
        f"episodes={num_episodes:<7} lr={lr_label:<16} gamma={gamma:<5} "
        f"eps_decay={epsilon_decay:<9} -> success={success_rate:5.1f}% "
        f"avg_steps={average_steps:5.1f} ({elapsed:.1f}s)"
    )

    return result, q_table


results = []
best_q_tables = {}

print("stage 1: screening gamma and epsilon_decay at 50000 episodes, lr=0.1")
stage1_combos = [
    (0.99, 0.00008),
    (0.99, 0.0001),
    (0.99, 0.00015),
    (0.95, 0.00008),
    (0.95, 0.0001),
    (0.95, 0.00015),
    (0.9, 0.00008),
    (0.9, 0.0001),
    (0.9, 0.00015),
]

for gamma, epsilon_decay in stage1_combos:
    result, q_table = run_combo(50000, gamma, epsilon_decay, 0.01, 0.1)
    results.append(result)
    best_q_tables[len(results) - 1] = q_table

stage1_sorted = sorted(results, key=lambda r: r["success_rate"], reverse=True)
top_two_stage1 = stage1_sorted[:2]

print()
print("stage 2: trying alternate learning rates on the top 2 stage 1 combos")
for top_result in top_two_stage1:
    gamma = top_result["gamma"]
    epsilon_decay = top_result["epsilon_decay"]

    for learning_rate in [0.05, "decay"]:
        result, q_table = run_combo(50000, gamma, epsilon_decay, 0.01, learning_rate)
        results.append(result)
        best_q_tables[len(results) - 1] = q_table

stage1_and_2_sorted = sorted(results, key=lambda r: r["success_rate"], reverse=True)
top_two_overall = stage1_and_2_sorted[:2]

print()
print("stage 3: retesting the top 2 combos with 100000 episodes")
for top_result in top_two_overall:
    result, q_table = run_combo(
        100000,
        top_result["gamma"],
        top_result["epsilon_decay"],
        top_result["epsilon_min"],
        top_result["learning_rate_value"],
    )
    results.append(result)
    best_q_tables[len(results) - 1] = q_table

final_sorted = sorted(range(len(results)), key=lambda i: results[i]["success_rate"], reverse=True)

print()
print("summary of all combinations tested, sorted by success rate")
print(f"{'episodes':<10}{'lr':<18}{'gamma':<8}{'eps_decay':<12}{'success_rate':<14}{'avg_steps':<10}")
for i in final_sorted:
    r = results[i]
    print(
        f"{r['episodes']:<10}{r['learning_rate']:<18}{r['gamma']:<8}"
        f"{r['epsilon_decay']:<12}{r['success_rate']:<14.1f}{r['average_steps']:<10.1f}"
    )

best_index = final_sorted[0]
best_result = results[best_index]

print()
print("best combination found:")
print(best_result)

env.close()
