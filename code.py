import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

class MovingObstacle:
    def __init__(self, initial_pos, movement_dir, boundaries, grid):
        self.pos = list(initial_pos)
        self.movement_dir = movement_dir
        self.boundaries = boundaries
        self.grid = grid

    def update_position(self, agent_pos):
        new_pos = list(self.pos)
        if self.movement_dir == 'up':
            new_pos[0] -= 1
        elif self.movement_dir == 'down':
            new_pos[0] += 1
        elif self.movement_dir == 'left':
            new_pos[1] -= 1
        elif self.movement_dir == 'right':
            new_pos[1] += 1

        if (self.boundaries[0] <= new_pos[0] <= self.boundaries[1] and
            self.boundaries[2] <= new_pos[1] <= self.boundaries[3] and
            self.grid[new_pos[0], new_pos[1]] != 'W' and
            tuple(new_pos) != agent_pos):
            self.pos = new_pos
        else:
            if self.movement_dir == 'up':
                self.movement_dir = 'down'
            elif self.movement_dir == 'down':
                self.movement_dir = 'up'
            elif self.movement_dir == 'left':
                self.movement_dir = 'right'
            elif self.movement_dir == 'right':
                self.movement_dir = 'left'

    def get_pos(self):
        return tuple(self.pos)

def create_grid(rows, cols, obstacle_prob=0.3):
    grid = np.random.choice(['.', 'W'], size=(rows, cols), p=[1 - obstacle_prob, obstacle_prob])
    grid[0, 0] = 'S'
    grid[-1, -1] = 'G'
    return grid

def is_grid_valid(grid):
    rows, cols = grid.shape
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque([start])
    visited[start] = True
    
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for direction in directions:
            new_row = current[0] + direction[0]
            new_col = current[1] + direction[1]
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if grid[new_row, new_col] != 'W' and not visited[new_row, new_col]:
                    visited[new_row, new_col] = True
                    queue.append((new_row, new_col))
    return False

def create_valid_grid(rows, cols, obstacle_prob=0.3):
    while True:
        grid = create_grid(rows, cols, obstacle_prob)
        if is_grid_valid(grid):
            return grid

rows, cols = 30, 30
grid = create_valid_grid(rows, cols)

actions = ['up', 'down', 'left', 'right']
action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

rewards = {
    '.': -1,
    'W': -10,
    'G': 10,
    'S': -1,
}

q_matrix = np.zeros((grid.shape[0], grid.shape[1], len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5000

def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return actions[np.argmax(q_matrix[state[0], state[1]])]

def move_agent(state, action, obstacle_positions):
    new_state = list(state)
    if action == 'up' and state[0] > 0:
        new_state[0] -= 1
    elif action == 'down' and state[0] < grid.shape[0] - 1:
        new_state[0] += 1
    elif action == 'left' and state[1] > 0:
        new_state[1] -= 1
    elif action == 'right' and state[1] < grid.shape[1] - 1:
        new_state[1] += 1

    new_state = tuple(new_state)

    if grid[new_state] == 'W' or new_state in obstacle_positions:
        return state, rewards['W']
    else:
        cell_value = grid[new_state]
        reward = rewards.get(cell_value, -1)
        return new_state, reward

def is_valid_position(pos, grid, boundaries):
    return (boundaries[0] <= pos[0] <= boundaries[1] and
            boundaries[2] <= pos[1] <= boundaries[3] and
            grid[pos] != 'W')

num_obstacles = rows * cols // 100
obstacles = []
boundaries = (1, rows-2, 1, cols-2)
for _ in range(num_obstacles):
    while True:
        initial_pos = (np.random.randint(1, rows-1), np.random.randint(1, cols-1))
        if is_valid_position(initial_pos, grid, boundaries):
            break
    movement_dir = np.random.choice(['up', 'down', 'left', 'right'])
    obstacles.append(MovingObstacle(initial_pos, movement_dir, boundaries, grid))

episode_rewards = []

for episode in range(episodes):
    state = (0, 0)
    done = False
    total_reward = 0
    visited_states = set()
    steps = 0

    while not done and steps < rows * cols:
        obstacle_positions = [obstacle.get_pos() for obstacle in obstacles]
        for obstacle in obstacles:
            obstacle.update_position(state)

        action = select_action(state)
        new_state, reward = move_agent(state, action, obstacle_positions)
        
        if new_state in visited_states:
            reward -= 5
        
        old_value = q_matrix[state[0], state[1], action_map[action]]
        next_max = np.max(q_matrix[new_state[0], new_state[1]])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_matrix[state[0], state[1], action_map[action]] = new_value

        state = new_state
        total_reward += reward
        steps += 1
        visited_states.add(state)

        if grid[state] == 'G':
            done = True

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

plt.figure(figsize=(10, 6))
plt.plot(range(len(episode_rewards)), episode_rewards, label='Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('Rewards During Learning')
plt.legend()
plt.grid()
plt.show()

state = (0, 0)
path = [state]
done = False
total_reward = 0
visited_states = set()
steps = 0
max_steps = rows * cols

while not done and steps < max_steps:
    obstacle_positions = [obstacle.get_pos() for obstacle in obstacles]
    for obstacle in obstacles:
        obstacle.update_position(state)

    action = actions[np.argmax(q_matrix[state[0], state[1]])]
    new_state, reward = move_agent(state, action, obstacle_positions)
    
    total_reward += reward
    
    if steps >= max_steps:
        print("Agent is stuck in a loop.")
        break
    
    path.append(new_state)
    state = new_state
    steps += 1
    visited_states.add(state)

    if grid[state] == 'G':
        done = True

if done:
    print(f"Total rewards : {total_reward}")
else:
    print("Agent did not reach the goal.")

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(grid == 'W', cmap='gray', interpolation='none')

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if grid[i, j] == 'S':
            ax.text(j, i, 'S', ha='center', va='center', color='red', fontsize=12)
        elif grid[i, j] == 'G':
            ax.text(j, i, 'G', ha='center', va='center', color='green', fontsize=12)
        elif grid[i, j] == 'W':
            ax.text(j, i, 'W', ha='center', va='center', color='black', fontsize=12)
        elif grid[i, j] == '.':
            ax.text(j, i, '.', ha='center', va='center', color='gray', fontsize=8)

obstacle_texts = []
for obstacle in obstacles:
    pos = obstacle.get_pos()
    obstacle_texts.append(ax.text(pos[1], pos[0], 'D', ha='center', va='center', color='yellow', fontsize=12))

def update(frame):
    for obstacle in obstacles:
        obstacle.update_position(path[frame])
    for i, obstacle in enumerate(obstacles):
        pos = obstacle.get_pos()
        obstacle_texts[i].set_position((pos[1], pos[0]))
    ax.text(path[frame][1], path[frame][0], 'X', ha='center', va='center', color='blue', fontsize=12)
    return ax

ani = FuncAnimation(fig, update, frames=len(path), interval=200, repeat=False)
plt.title('Grid with Learned Path')
plt.show()

max_q_values = np.max(q_matrix, axis=2)

min_q_value = np.min(max_q_values)
max_q_value = np.max(max_q_values)

plt.figure(figsize=(10, 10))
plt.imshow(max_q_values, cmap='viridis', vmin=min_q_value, vmax=max_q_value, interpolation='none')
plt.colorbar(label='Max Q-value')

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if grid[i, j] == 'W':
            plt.text(j, i, 'W', ha='center', va='center', color='black', fontsize=12)
        elif grid[i, j] == 'S':
            plt.text(j, i, 'S', ha='center', va='center', color='red', fontsize=12)
        elif grid[i, j] == 'G':
            plt.text(j, i, 'G', ha='center', va='center', color='green', fontsize=12)

for obstacle in obstacles:
    pos = obstacle.get_pos()
    plt.text(pos[1], pos[0], 'D', ha='center', va='center', color='blue', fontsize=12)

plt.title('Heatmap of Max Q-values')
plt.show()