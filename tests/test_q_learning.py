import numpy as np

from q_learning import choose_action, update_q_table, decay_epsilon


def test_update_q_table_known_example():
    q_table = np.zeros((2, 2))
    q_table[1, 0] = 0.5

    update_q_table(
        q_table, state=0, action=0, reward=1.0, next_state=1,
        done=False, learning_rate=0.5, discount_rate=0.9,
    )

    expected = 0.5 * (1.0 + 0.9 * 0.5)
    assert np.isclose(q_table[0, 0], expected)


def test_update_q_table_does_not_bootstrap_through_terminal_state():
    q_table = np.zeros((2, 2))
    q_table[1, :] = 10.0

    update_q_table(
        q_table, state=0, action=0, reward=1.0, next_state=1,
        done=True, learning_rate=0.5, discount_rate=0.9,
    )

    assert np.isclose(q_table[0, 0], 0.5)


def test_choose_action_epsilon_one_is_random():
    np.random.seed(0)
    q_table = np.zeros((1, 4))
    q_table[0, 2] = 5.0

    actions = {choose_action(q_table, 0, epsilon=1.0, num_actions=4) for _ in range(200)}

    assert actions == {0, 1, 2, 3}


def test_choose_action_epsilon_zero_is_greedy():
    q_table = np.zeros((1, 4))
    q_table[0, 2] = 5.0

    for _ in range(50):
        assert choose_action(q_table, 0, epsilon=0.0, num_actions=4) == 2


def test_decay_epsilon_is_monotonically_decreasing():
    values = [decay_epsilon(episode, max_epsilon=1.0, min_epsilon=0.05, decay_rate=0.01) for episode in range(500)]

    assert all(a >= b for a, b in zip(values, values[1:]))


def test_decay_epsilon_respects_min_epsilon_floor():
    values = [decay_epsilon(episode, max_epsilon=1.0, min_epsilon=0.05, decay_rate=0.01) for episode in range(2000)]

    assert all(v >= 0.05 for v in values)
    assert values[0] == 1.0
    assert values[-1] < 0.06
