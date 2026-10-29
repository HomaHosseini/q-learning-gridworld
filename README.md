# Q-Learning Grid World

An educational reinforcement learning project that explores tabular Q-learning for grid-world navigation. An agent learns to reach a goal while navigating a grid containing static walls and moving obstacles.

The project visualizes episode rewards, the agent's learned path, and the maximum Q-values associated with grid positions.

## Features

- **Tabular Q-learning:** Learns action values through reward-based interaction.
- **Grid-world environment:** Generates a grid with a starting point, a goal, and randomly placed walls.
- **Moving obstacles:** Simulates obstacles that move within defined boundaries and reverse direction when blocked.
- **Epsilon-greedy exploration:** Balances random exploration with choosing actions based on learned Q-values.
- **Reward-based learning:** Applies rewards and penalties for reaching the goal, moving through the grid, hitting walls, and revisiting states.
- **Visualizations:** Plots episode rewards, animates the learned path, and displays a heatmap of maximum Q-values.

## How It Works

The environment is represented as a two-dimensional grid. The agent starts at the top-left cell and attempts to reach the goal at the bottom-right cell.

The agent can choose from four actions:

- Move up
- Move down
- Move left
- Move right

During training, the agent uses an epsilon-greedy policy to select actions and updates its Q-table according to the rewards it receives.

The Q-table is updated using the Q-learning rule:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

where:

- `s`: Current state
- `a`: Selected action
- `r`: Received reward
- `s'`: Next state
- `alpha (α)`: Learning rate
- `gamma (γ)`: Discount factor
- `max Q(s', a')`: Highest estimated future action value across all possible next actions.


## Configuration

The main parameters are defined in `main.py`.

| Parameter | Value | Description |
|---|---:|---|
| Grid size | 30 × 30 | Number of rows and columns |
| Wall probability | 0.3 | Probability of a cell initially containing a wall |
| Training episodes | 5,000 | Number of learning episodes |
| Learning rate | 0.1 | Controls the size of Q-value updates |
| Discount factor | 0.9 | Determines the importance of future rewards |
| Initial epsilon | 1.0 | Initial probability of random exploration |
| Epsilon decay | 0.995 | Exploration decay factor per episode |
| Minimum epsilon | 0.01 | Lower bound for exploration |
| Obstacle count | 9 | Number of moving obstacles in a 30 × 30 grid |

## Requirements

- Python 3.9 or newer
- NumPy
- Matplotlib


## Visualizations

The script produces three visual outputs:

1. **Episode reward curve:** Shows the total reward collected during each training episode.
2. **Learned path animation:** Displays the agent's evaluated path and the movement of obstacles.
3. **Q-value heatmap:** Visualizes the maximum learned action value for each grid position.

These visualizations help explore the relationship between reward-based learning, navigation behavior, and the Q-table.


## Learning Objectives

This project provides practical experience with:

- Reinforcement learning fundamentals.
- Tabular Q-learning.
- Exploration versus exploitation.
- Reward design and Q-value updates.
- Grid-based environment modeling.
- Dynamic obstacles and navigation.
- Visualizing learning behavior with Matplotlib.
