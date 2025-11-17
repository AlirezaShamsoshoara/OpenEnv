# Snake Environment Examples

This directory contains comprehensive examples and tests for the Snake environment in OpenEnv.

## Running Options

All scripts support **two modes**:

### Option 1: Docker Mode (Recommended)
Automatically starts/stops a Docker container.

```bash
python examples/snake_rl/test_snake_env.py --mode docker
```

**Prerequisites:**
- Docker installed and running
- Image built: `docker build -t snake-env:latest -f src/envs/snake_env/server/Dockerfile .`

### Option 2: Local Server Mode
Connects to a running local server (faster for development).

```bash
# Terminal 1: Start the server
cd src/envs/snake_env
pip install -e .
uv run --project . server
# (or: python -m server.app)

# Terminal 2: Run the tests
python examples/snake_rl/test_snake_env.py --mode local --url http://localhost:8000
```

**Prerequisites:**
- marlenv installed: `pip install marlenv gym==0.24.1 numpy Pillow`
- Server dependencies installed

## Files

### 1. `test_snake_env.py` - Comprehensive Test Suite

A complete test suite that validates all features of the snake environment.

**Tests included:**
- ✓ Environment reset
- ✓ Action execution (turn left, right, no-op)
- ✓ Observation structure (grid, encoded observations, statistics)
- ✓ Reward system (fruit collection, death penalties)
- ✓ State tracking (episode ID, step count)
- ✓ Episode termination
- ✓ Multiple consecutive episodes

**Usage:**
```bash
# Docker mode (recommended)
python examples/snake_rl/test_snake_env.py --mode docker

# Local server mode
python examples/snake_rl/test_snake_env.py --mode local --url http://localhost:8000
```

**Prerequisites:**
- Docker image must be built: `docker build -t snake-env:latest -f src/envs/snake_env/server/Dockerfile .`

### 2. `visualize_snake.py` - Visualization Examples

Demonstrates multiple ways to visualize the snake game.

**Visualization methods:**
1. **ASCII text rendering** - Simple text-based grid display
2. **Matplotlib grid visualization** - Colored grid showing game state
3. **Observation channels** - Individual feature channels visualization
4. **Episode statistics** - Comprehensive episode analysis with plots

**Usage:**
```bash
# Docker mode
python examples/snake_rl/visualize_snake.py --mode docker

# Local server mode
python examples/snake_rl/visualize_snake.py --mode local
```

**Examples shown:**
- Single step visualization (before/after)
- Encoded observation channels breakdown
- Full episode visualization with statistics

### 3. `play_snake.py` - Interactive Player

Automated agent that plays the snake game with real-time visualization.

**Features:**
- Automated gameplay with different policies
- Real-time visualization during play
- Multi-episode statistics
- Performance metrics

**Usage:**
```bash
# Docker mode - single episode with random policy
python examples/snake_rl/play_snake.py --mode docker --play-mode auto --policy random --steps 500

# Docker mode - multiple episodes
python examples/snake_rl/play_snake.py --mode docker --play-mode multi --episodes 10 --steps 200

# Local server mode - simple heuristic policy
python examples/snake_rl/play_snake.py --mode local --play-mode auto --policy simple
```

**Arguments:**
- `--mode`: `docker` or `local` (connection mode)
- `--play-mode`: `auto` (single episode) or `multi` (multiple episodes)
- `--policy`: `random` or `simple` (basic heuristic)
- `--episodes`: Number of episodes (for multi mode)
- `--steps`: Maximum steps per episode

## Quick Start

### Option 1: Docker Mode (Recommended)

1. **Build the Docker image:**
   ```bash
   cd /path/to/OpenEnv
   docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
   docker build -t snake-env:latest -f src/envs/snake_env/server/Dockerfile .
   ```

2. **Run the comprehensive test:**
   ```bash
   python examples/snake_rl/test_snake_env.py --mode docker
   ```

3. **Try visualization:**
   ```bash
   python examples/snake_rl/visualize_snake.py --mode docker
   ```

4. **Watch automated gameplay:**
   ```bash
   python examples/snake_rl/play_snake.py --mode docker --play-mode auto
   ```

### Option 2: Local Server Mode (Faster for Development)

1. **Install dependencies:**
   ```bash
   cd src/envs/snake_env
   pip install -e .
   # Or install manually: pip install marlenv gym==0.24.1 numpy Pillow
   ```

2. **Start the server (Terminal 1):**
   ```bash
   cd src/envs/snake_env
   uv run --project . server
   # Or: python -m server.app
   ```

3. **Run tests (Terminal 2):**
   ```bash
   python examples/snake_rl/test_snake_env.py --mode local
   ```

4. **Try visualization:**
   ```bash
   python examples/snake_rl/visualize_snake.py --mode local
   ```

## Understanding the Observations

The snake environment provides rich observations with multiple components:

### 1. Grid (Full Game State)
A 2D array representing the full game grid:
- `0` = Empty cell
- `1` = Wall
- `2` = Fruit
- `3` = Snake body
- `4` = Snake tail
- `5` = Snake head

**Example:**
```python
result = client.reset()
grid = result.observation.grid  # Shape: (height, width)
print(f"Grid shape: {len(grid)}x{len(grid[0])}")
```

### 2. Encoded Observation
A multi-channel tensor encoding game features:
- Shape: `(height, width, channels)`
- Channels (typically 8):
  - 0-1: Environment objects (walls, fruits)
  - 2-4: Own snake features (head, body, tail)
  - 5-7: Other snake features (for multi-agent)

**Example:**
```python
obs = result.observation.observation  # Shape: (H, W, 8)
# Access specific channel
walls_channel = obs[:, :, 0]
fruits_channel = obs[:, :, 1]
own_head_channel = obs[:, :, 2]
```

### 3. Episode Statistics
Track performance metrics:
```python
result.observation.episode_score    # Cumulative score
result.observation.episode_steps    # Number of steps taken
result.observation.episode_fruits   # Fruits collected
result.observation.episode_kills    # Kills (multi-agent)
result.observation.alive            # Snake status
```

### 4. Reward & Done Flags
```python
result.reward  # Step reward (positive for fruit, negative for death)
result.done    # Episode termination flag
```

## Visualization Approaches

### ASCII Visualization
Simple text-based rendering:
```python
from examples.snake_rl.visualize_snake import print_ascii_grid

result = client.reset()
print_ascii_grid(result.observation.grid)
```

Output:
```
# # # # # # # # # #
# . . . o . . . . #
# . . . . . . . . #
# . H b t . . . . #
# . . . . . . . . #
# # # # # # # # # #
```

Legend: `#` = wall, `.` = empty, `o` = fruit, `H` = head, `b` = body, `t` = tail

### Matplotlib Visualization
Color-coded grid display:
```python
from examples.snake_rl.visualize_snake import visualize_grid_matplotlib
import matplotlib.pyplot as plt

fig, ax = visualize_grid_matplotlib(result.observation.grid)
plt.show()
```

Colors:
- Light gray: Empty
- Dark gray: Wall
- Red: Fruit
- Bright green: Snake head
- Green: Snake body
- Dark green: Snake tail

### Channel Visualization
View individual observation channels:
```python
from examples.snake_rl.visualize_snake import visualize_observation_channels

fig, axes = visualize_observation_channels(result.observation.observation)
plt.show()
```

This creates a subplot for each channel showing spatial features.

### Episode Visualization
Complete episode analysis:
```python
from examples.snake_rl.visualize_snake import visualize_episode

fig = visualize_episode(client, num_steps=100)
plt.show()
```

Shows:
- Final grid state
- Reward history
- Score progression
- Summary statistics

## Training an RL Agent

To train a reinforcement learning agent, you can use the snake environment with frameworks like:

### Example with Random Policy
```python
from envs.snake_env import SnakeAction, SnakeEnv
import random

client = SnakeEnv.from_docker_image("snake-env:latest")

for episode in range(10):
    result = client.reset()
    total_reward = 0

    while not result.done:
        # Your RL agent would choose action here
        action = random.randint(0, 2)
        result = client.step(SnakeAction(action=action))
        total_reward += result.reward

    print(f"Episode {episode}: Reward={total_reward:.2f}, "
          f"Fruits={result.observation.episode_fruits}")

client.close()
```

### Example with torchforge/TRL
```python
# See examples/grpo_blackjack/ for a complete GRPO training example
# The snake environment follows the same interface
```

## Troubleshooting

### Docker Image Not Found
```bash
# Build the base image
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Build the snake environment
docker build -t snake-env:latest -f src/envs/snake_env/server/Dockerfile .
```

### Import Errors
Make sure you're running from the repository root:
```bash
cd /path/to/OpenEnv
python examples/snake_rl/test_snake_env.py
```

### Visualization Issues
Install matplotlib if needed:
```bash
pip install matplotlib numpy
```

## Performance Tips

1. **Use vision range** for partial observability (smaller observations):
   ```python
   # In snake_environment.py, set vision_range=5
   ```

2. **Adjust grid size** for faster episodes:
   ```python
   # Smaller grid = faster episodes
   env = SnakeEnvironment(height=10, width=10)
   ```

3. **Customize rewards** to shape learning:
   ```python
   reward_dict = {
       'fruit': 10.0,    # High reward for eating
       'lose': -10.0,    # Penalty for dying
       'time': 0.0,      # No time reward
   }
   ```

## Next Steps

1. Implement a smarter policy (e.g., move towards fruits)
2. Train an RL agent using torchforge or TRL
3. Experiment with different grid sizes and reward functions
4. Try multi-agent scenarios (modify num_snakes > 1)

## References

- Snake environment docs: `src/envs/snake_env/README.md`
- marlenv documentation: https://github.com/kc-ml2/marlenv
- OpenEnv tutorial: `examples/OpenEnv_Tutorial.ipynb`
