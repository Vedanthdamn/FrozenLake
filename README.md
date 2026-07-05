# FrozenLake Q-Learning Agent

A tabular Q-learning agent trained from scratch to solve the FrozenLake-v1 environment from Gymnasium.

## What is Reinforcement Learning?

Reinforcement learning (RL) is a way of training an agent to make decisions by letting it interact with an environment and learn from the outcomes of its actions. The agent observes a state, takes an action, and receives a reward. Over many attempts, it learns which actions lead to good outcomes and which don't, without ever being told the "correct" action directly. This is different from supervised learning, where a model is given labeled examples of correct answers.

## What is Q-Learning?

Q-learning is one of the simplest RL algorithms. It keeps a table (the Q-table) with one row per state and one column per action. Each entry, Q(state, action), estimates how good it is to take that action from that state, in terms of expected future reward.

The agent updates this table using the rule:

```
Q(s, a) = Q(s, a) + learning_rate * (reward + discount_rate * max(Q(s', a')) - Q(s, a))
```

While training, the agent uses an epsilon-greedy strategy: with probability epsilon it takes a random action (exploration), and otherwise it takes the action with the highest Q-value (exploitation). Epsilon starts high so the agent explores a lot early on, then decays over time so the agent increasingly relies on what it has learned.

Once training is done, the Q-table itself is the agent's policy: in any state, just pick the action with the highest Q-value.

## What is FrozenLake?

FrozenLake is a classic grid-world environment from Gymnasium. The agent starts at one corner of a 4x4 grid of ice and has to reach a goal tile at the opposite corner without falling into any holes. Each tile is one of:

- `S` start tile
- `F` frozen tile, safe to walk on
- `H` hole, falling in ends the episode with no reward
- `G` goal, reaching it ends the episode with a reward of 1

The environment can run in `is_slippery=True` mode, where the ice is slippery and the agent's action only succeeds with some probability, sometimes sliding it sideways instead. This makes the environment genuinely stochastic, not just a fixed maze to memorize.

### Why `is_slippery=True`

This project trains on the slippery version of the map. It is the harder, more realistic setting: since actions do not always do what you expect, the agent has to learn a policy that is robust to randomness rather than a single fixed path. It also makes Q-learning's convergence properties much more visible in the training curve, since a non-slippery 4x4 map is small enough that an agent can trivially reach 100% success rate after very little training, which is not an interesting demonstration of the algorithm. With enough training episodes, tabular Q-learning still converges to a strong policy on the slippery map, just with a success rate below 100% since some falls into holes are due to chance and unavoidable even with the optimal policy.

## Project Files

- `train.py` trains the Q-learning agent and saves the Q-table to `q_table.npy`
- `evaluate.py` loads the saved Q-table and reports success rate and average steps, compared against a random policy baseline
- `demo.py` renders the trained agent solving the environment and saves the result as `demo.gif`
- `plot_training.py` plots the training curve from the saved reward history and saves `training_curve.png`
- `sweep.py` the script I used to try out different hyperparameter combinations, not needed to reproduce the final results

## Installation

Requires Python 3.10 or later.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Train the agent:

```
python3 train.py
```

This runs 50,000 training episodes and saves `q_table.npy` and `rewards_per_episode.npy`.

Evaluate the trained agent against a random baseline:

```
python3 evaluate.py
```

Plot the training curve:

```
python3 plot_training.py
```

Generate a GIF of the trained agent solving the environment:

```
python3 demo.py
```

## Hyperparameter Tuning

After getting a working baseline, I wrote a small `sweep.py` script to try out different combinations of episodes, learning rate, discount factor (gamma), and epsilon decay, evaluating each one over 1000 greedy episodes instead of 100 so the success rate numbers weren't just noise. Gamma mattered the most out of everything I tried: dropping it from 0.99 to 0.95 or 0.9 consistently hurt performance, since a lower gamma makes the agent care less about the reward at the goal, which is far away in terms of steps. The best combination I found was 50,000 episodes, learning rate 0.1, gamma 0.99, and a slightly slower epsilon decay rate of 0.0001 (compared to 0.00015 in my first version), and I updated `train.py` to use these settings. `sweep.py` is left in the repo as the script I used to do this, it's not part of the main pipeline.

## Results

Evaluated over 1000 episodes with a greedy policy (no exploration), compared to 1000 episodes of a random policy baseline. I bumped this up from 100 episodes in an earlier version since 100 episodes was giving noisy success rate estimates during tuning:

| Policy | Success Rate | Average Steps |
|---|---|---|
| Random (before training) | 1.4% | 7.5 |
| Trained Q-learning agent (after training) | 73.8% | 45.5 |

The random policy almost always falls into a hole quickly. The trained agent reaches the goal in the large majority of episodes, and takes longer paths on average because it favors safer routes around holes rather than the shortest path, which matters on slippery ice where a shorter but riskier path is more likely to end in a fall.

### Training Curve

![Training curve](training_curve.png)

The curve shows the rolling average success rate over training. The agent starts at close to 0% (all random exploration) and climbs steadily as epsilon decays, leveling off once the Q-table has converged.

### Demo

![Demo](demo.gif)

Three episodes of the trained agent navigating the ice grid to reach the goal.
