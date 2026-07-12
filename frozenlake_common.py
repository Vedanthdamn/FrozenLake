def get_suffix(map_name, slippery):
    if slippery:
        return "" if map_name == "4x4" else f"_{map_name}"
    return f"_{map_name}_deterministic"


def q_table_path(map_name, slippery):
    return f"q_table{get_suffix(map_name, slippery)}.npy"


def rewards_path(map_name, slippery):
    return f"rewards_per_episode{get_suffix(map_name, slippery)}.npy"


def demo_path(map_name, slippery):
    return f"demo{get_suffix(map_name, slippery)}.gif"


def curve_path(map_name, slippery):
    return f"training_curve{get_suffix(map_name, slippery)}.png"


TRAIN_CONFIGS = {
    ("4x4", True): dict(
        approach="simple",
        num_episodes=50000,
        learning_rate=0.1,
        discount_rate=0.99,
        min_epsilon=0.01,
        epsilon_decay_rate=0.0001,
        checkpoint_interval=1000,
    ),
    ("4x4", False): dict(
        approach="simple",
        num_episodes=5000,
        learning_rate=0.1,
        discount_rate=0.99,
        min_epsilon=0.01,
        epsilon_decay_rate=0.001,
        checkpoint_interval=500,
    ),
    ("8x8", False): dict(
        approach="simple",
        num_episodes=50000,
        learning_rate=0.1,
        discount_rate=0.99,
        min_epsilon=0.01,
        epsilon_decay_rate=0.00005,
        checkpoint_interval=1000,
    ),
    ("8x8", True): dict(
        approach="vectorized",
        max_num_episodes=3000000,
        num_envs=128,
        start_learning_rate=0.1,
        end_learning_rate=0.01,
        lr_decay_episodes=1500000,
        discount_rate=0.99,
        optimistic_init_value=1.0,
        min_epsilon=0.01,
        epsilon_decay_rate=0.000003,
        checkpoint_interval=5000,
        greedy_eval_interval=50000,
        greedy_eval_episodes=2000,
        smoothing_window_evals=4,
        plateau_window_evals=6,
        plateau_threshold=1.5,
        min_episodes_before_plateau_check=1200000,
    ),
}

EVAL_EPISODES = {
    ("4x4", True): 1000,
    ("8x8", True): 2000,
    ("4x4", False): 1000,
    ("8x8", False): 1000,
}

DEMO_MAX_ATTEMPTS = {
    ("4x4", True): 50,
    ("8x8", True): 500,
    ("4x4", False): 50,
    ("8x8", False): 50,
}

MAX_STEPS_PER_EPISODE = 100
