#!/usr/bin/env python3
"""
Snake Environment Visualization.

This script demonstrates how to visualize the snake game observations.
It shows multiple visualization methods:
1. ASCII text rendering
2. Matplotlib grid visualization
3. Encoded observation channels visualization

Usage:
    # Option 1: Use Docker (recommended)
    python examples/snake_rl/visualize_snake.py --mode docker

    # Option 2: Use local server (must be running separately)
    python examples/snake_rl/visualize_snake.py --mode local --url http://localhost:8000
"""

import sys
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from envs.snake_env import SnakeAction, SnakeEnv


# Cell type constants (from marlenv)
CELL_EMPTY = 0
CELL_WALL = 1
CELL_FRUIT = 2
CELL_BODY = 3
CELL_HEAD = 5
CELL_TAIL = 4


def print_ascii_grid(grid):
    """
    Print ASCII representation of the game grid.

    Cell values:
        0 = empty (.)
        1 = wall (#)
        2 = fruit (o)
        3-9 = snake body parts (different numbers for different snakes)
    """
    symbol_map = {
        CELL_EMPTY: '.',
        CELL_WALL: '#',
        CELL_FRUIT: 'o',
    }

    print("\n  ASCII Grid:")
    for row in grid:
        line = ""
        for cell in row:
            cell_type = cell % 10  # Get base cell type
            if cell_type in symbol_map:
                line += symbol_map[cell_type] + ' '
            elif cell_type == CELL_BODY:
                line += 'b '  # body
            elif cell_type == CELL_TAIL:
                line += 't '  # tail
            elif cell_type == CELL_HEAD:
                line += 'H '  # head
            else:
                line += '? '
        print(f"    {line}")


def visualize_grid_matplotlib(grid, title="Snake Game Grid"):
    """
    Visualize the game grid using matplotlib.

    Args:
        grid: 2D list of cell values
        title: Title for the plot
    """
    grid_array = np.array(grid)
    height, width = grid_array.shape

    # Create color map
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create colored grid
    colored_grid = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            cell = grid_array[i, j]
            cell_type = cell % 10

            if cell_type == CELL_EMPTY:
                colored_grid[i, j] = [0.95, 0.95, 0.95]  # Light gray
            elif cell_type == CELL_WALL:
                colored_grid[i, j] = [0.2, 0.2, 0.2]  # Dark gray
            elif cell_type == CELL_FRUIT:
                colored_grid[i, j] = [1.0, 0.0, 0.0]  # Red
            elif cell_type == CELL_BODY:
                colored_grid[i, j] = [0.0, 0.8, 0.0]  # Green
            elif cell_type == CELL_HEAD:
                colored_grid[i, j] = [0.0, 1.0, 0.0]  # Bright green
            elif cell_type == CELL_TAIL:
                colored_grid[i, j] = [0.0, 0.6, 0.0]  # Dark green

    ax.imshow(colored_grid, interpolation='nearest')
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Create legend
    legend_elements = [
        mpatches.Patch(color=[0.95, 0.95, 0.95], label='Empty'),
        mpatches.Patch(color=[0.2, 0.2, 0.2], label='Wall'),
        mpatches.Patch(color=[1.0, 0.0, 0.0], label='Fruit'),
        mpatches.Patch(color=[0.0, 1.0, 0.0], label='Head'),
        mpatches.Patch(color=[0.0, 0.8, 0.0], label='Body'),
        mpatches.Patch(color=[0.0, 0.6, 0.0], label='Tail'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    return fig, ax


def visualize_observation_channels(observation, title="Encoded Observation Channels"):
    """
    Visualize the encoded observation channels.

    The observation is typically (H, W, C) where C is the number of channels.
    Channels represent different game elements:
        - Channel 0-1: Environment objects (walls, fruits)
        - Channel 2-7: Snake-specific features (head, body, tail for self and others)
    """
    obs_array = np.array(observation)
    height, width, channels = obs_array.shape

    # Create subplots for each channel
    n_cols = min(4, channels)
    n_rows = (channels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(title, fontsize=16)

    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    channel_names = [
        'Walls', 'Fruits',
        'Own Head', 'Own Body', 'Own Tail',
        'Other Head', 'Other Body', 'Other Tail'
    ]

    for idx in range(channels):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]

        channel_data = obs_array[:, :, idx]
        im = ax.imshow(channel_data, cmap='hot', interpolation='nearest')

        channel_name = channel_names[idx] if idx < len(channel_names) else f'Channel {idx}'
        ax.set_title(channel_name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide extra subplots
    for idx in range(channels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    return fig, axes


def visualize_episode(client, num_steps=50, save_path=None):
    """
    Visualize a full episode with statistics.

    Args:
        client: SnakeEnv client
        num_steps: Maximum number of steps to run
        save_path: Optional path to save the final visualization
    """
    print("\n" + "=" * 70)
    print("  Running Episode Visualization")
    print("=" * 70)

    result = client.reset()

    rewards = []
    scores = []
    fruits = []
    alive_status = []

    print(f"\n  Playing episode for up to {num_steps} steps...")

    for step in range(num_steps):
        if result.done:
            print(f"  Episode ended at step {step}")
            break

        # Random action
        import random
        action = random.randint(0, 2)
        result = client.step(SnakeAction(action=action))

        rewards.append(result.reward)
        scores.append(result.observation.episode_score)
        fruits.append(result.observation.episode_fruits)
        alive_status.append(1 if result.observation.alive else 0)

        if result.observation.episode_fruits > len(fruits) - 1:
            print(f"    Step {step + 1}: Fruit collected! Total: {result.observation.episode_fruits}")

    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Final grid state
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    grid_array = np.array(result.observation.grid)
    height, width = grid_array.shape
    colored_grid = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            cell = grid_array[i, j]
            cell_type = cell % 10

            if cell_type == CELL_EMPTY:
                colored_grid[i, j] = [0.95, 0.95, 0.95]
            elif cell_type == CELL_WALL:
                colored_grid[i, j] = [0.2, 0.2, 0.2]
            elif cell_type == CELL_FRUIT:
                colored_grid[i, j] = [1.0, 0.0, 0.0]
            elif cell_type == CELL_BODY:
                colored_grid[i, j] = [0.0, 0.8, 0.0]
            elif cell_type == CELL_HEAD:
                colored_grid[i, j] = [0.0, 1.0, 0.0]
            elif cell_type == CELL_TAIL:
                colored_grid[i, j] = [0.0, 0.6, 0.0]

    ax1.imshow(colored_grid, interpolation='nearest')
    status = "ALIVE" if result.observation.alive else "DEAD"
    ax1.set_title(f'Final Game State - {status}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')

    # 2. Reward over time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(rewards, 'b-', linewidth=2)
    ax2.set_title('Rewards per Step')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative score
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(scores, 'g-', linewidth=2)
    ax3.set_title('Cumulative Score')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Score')
    ax3.grid(True, alpha=0.3)

    # 4. Episode statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    stats_text = f"""
    EPISODE STATISTICS
    ═══════════════════════════════════════════════════════════════════
    Total Steps:              {result.observation.episode_steps}
    Final Score:              {result.observation.episode_score:.2f}
    Fruits Collected:         {result.observation.episode_fruits}
    Snake Status:             {'🟢 Alive' if result.observation.alive else '🔴 Dead'}
    Total Reward:             {sum(rewards):.2f}
    Average Reward/Step:      {np.mean(rewards):.3f}
    ═══════════════════════════════════════════════════════════════════
    """

    ax4.text(0.5, 0.5, stats_text,
             transform=ax4.transAxes,
             fontsize=11,
             verticalalignment='center',
             horizontalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Snake Environment Episode Visualization', fontsize=18, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Visualization saved to: {save_path}")

    return fig


def main():
    """Run visualization examples."""
    parser = argparse.ArgumentParser(
        description='Visualize Snake Environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Option 1: Use Docker
  python examples/snake_rl/visualize_snake.py --mode docker

  # Option 2: Use local server
  # Terminal 1: cd src/envs/snake_env && uv run --project . server
  # Terminal 2: python examples/snake_rl/visualize_snake.py --mode local
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['docker', 'local'],
        default='docker',
        help='Connection mode: docker or local'
    )
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8000',
        help='Server URL for local mode'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  Snake Environment - Visualization Examples")
    print("=" * 70)

    if args.mode == 'docker':
        print("\n📦 Mode: Docker")
        print("  Building and running containerized environment...")
    else:
        print(f"\n🖥️  Mode: Local Server ({args.url})")
        print("  Connecting to running server...")

    try:
        if args.mode == 'docker':
            print("\nStarting Docker container...")
            client = SnakeEnv.from_docker_image("snake-env:latest")
            print("✓ Container started successfully!\n")
        else:
            print(f"\nConnecting to {args.url}...")
            client = SnakeEnv(base_url=args.url)
            client.reset()  # Test connection
            print("✓ Connected successfully!\n")

        # Example 1: Single step visualization
        print("\n" + "=" * 70)
        print("  Example 1: Single Step Visualization")
        print("=" * 70)

        result = client.reset()

        # ASCII rendering
        print_ascii_grid(result.observation.grid)

        # Matplotlib grid visualization
        print("\n  Creating matplotlib visualization...")
        fig1, _ = visualize_grid_matplotlib(
            result.observation.grid,
            title="Snake Game - Initial State"
        )
        plt.show(block=False)
        plt.pause(2)

        # Take a few steps
        print("\n  Taking 5 steps...")
        for i in range(5):
            result = client.step(SnakeAction(action=0))

        print_ascii_grid(result.observation.grid)
        fig2, _ = visualize_grid_matplotlib(
            result.observation.grid,
            title="Snake Game - After 5 Steps"
        )
        plt.show(block=False)
        plt.pause(2)

        # Example 2: Observation channels visualization
        print("\n" + "=" * 70)
        print("  Example 2: Encoded Observation Channels")
        print("=" * 70)

        print(f"\n  Observation shape: {len(result.observation.observation)}x"
              f"{len(result.observation.observation[0])}x"
              f"{len(result.observation.observation[0][0])}")

        fig3, _ = visualize_observation_channels(result.observation.observation)
        plt.show(block=False)
        plt.pause(3)

        # Example 3: Full episode visualization
        print("\n" + "=" * 70)
        print("  Example 3: Full Episode Visualization")
        print("=" * 70)

        fig4 = visualize_episode(client, num_steps=100)
        plt.show(block=True)

        print("\n" + "=" * 70)
        print("  Visualization Complete!")
        print("=" * 70)
        print("\n  You can visualize:")
        print("    ✓ Grid state (ASCII and matplotlib)")
        print("    ✓ Encoded observation channels")
        print("    ✓ Episode statistics and progression")
        print("\n  All visualization functions are available in this script")

        # Cleanup
        print("\nCleaning up...")
        if args.mode == 'docker':
            client.close()
            print("✓ Container stopped and removed")
        else:
            print("✓ Disconnected from server")

        return True

    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
