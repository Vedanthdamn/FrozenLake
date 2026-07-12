import numpy as np


def choose_action(q_table, state, epsilon, num_actions):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(q_table[state, :])


def update_q_table(q_table, state, action, reward, next_state, done, learning_rate, discount_rate):
    best_next_value = 0.0 if done else np.max(q_table[next_state, :])
    td_target = reward + discount_rate * best_next_value
    q_table[state, action] += learning_rate * (td_target - q_table[state, action])
    return q_table


def decay_epsilon(episode, max_epsilon, min_epsilon, decay_rate):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
