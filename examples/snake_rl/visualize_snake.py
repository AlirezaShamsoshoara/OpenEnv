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


# Cell type constants (from marlenv.envs.snake_env.Cell)
CELL_EMPTY = 0
CELL_WALL = 1
CELL_FRUIT = 2
CELL_HEAD = 3  # Snake head
CELL_BODY = 4  # Snake body segments
CELL_TAIL = 5  # Snake tail


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


def record_episode_as_gif(client, num_steps=100, save_path='snake_episode.gif'):
    """
    Record an episode and save as animated GIF.

    This function plays through an episode while collecting frames,
    then saves them as an animated GIF file.

    Args:
        client: SnakeEnv client
        num_steps: Maximum number of steps to record
        save_path: Path to save the GIF file

    Returns:
        Path to the saved GIF file
    """
    print(f"\n  Recording episode to GIF (max {num_steps} steps)...")

    # Get access to the base environment for rendering
    # We need to reach through the client -> server -> base_env
    # For now, we'll collect frames manually using the grid state

    from PIL import Image
    import numpy as np

    frames = []
    result = client.reset()

    for step in range(num_steps):
        if result.done:
            break

        # Convert grid to RGB image
        grid = np.array(result.observation.grid)
        height, width = grid.shape

        # Create RGB image (scale up for visibility)
        scale = 20  # Each cell is 20x20 pixels
        img_array = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                cell = grid[i, j]
                cell_type = cell % 10

                # Color mapping
                if cell_type == CELL_EMPTY:
                    color = (240, 240, 240)  # Light gray
                elif cell_type == CELL_WALL:
                    color = (50, 50, 50)  # Dark gray
                elif cell_type == CELL_FRUIT:
                    color = (255, 0, 0)  # Red
                elif cell_type == CELL_BODY:
                    color = (0, 200, 0)  # Green
                elif cell_type == CELL_HEAD:
                    color = (0, 255, 0)  # Bright green
                elif cell_type == CELL_TAIL:
                    color = (0, 150, 0)  # Dark green
                else:
                    color = (200, 200, 200)  # Default gray

                # Fill the scaled cell
                img_array[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = color

        frames.append(Image.fromarray(img_array))

        # Take random action
        import random
        action = random.randint(0, 2)
        result = client.step(SnakeAction(action=action))

    # Save as GIF
    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,  # 200ms per frame
            loop=0
        )
        print(f"  ✓ GIF saved to: {save_path}")
        print(f"    - Total frames: {len(frames)}")
        print(f"    - Final score: {result.observation.episode_score:.2f}")
        print(f"    - Fruits collected: {result.observation.episode_fruits}")
        return save_path
    else:
        print("  ⚠️  No frames recorded")
        return None


def replay_episode_animation(client, num_steps=100):
    """
    Play an episode and show as matplotlib animation.

    Args:
        client: SnakeEnv client
        num_steps: Maximum number of steps
    """
    print(f"\n  Playing episode with matplotlib animation...")

    # Collect episode data
    frames = []
    result = client.reset()

    for step in range(num_steps):
        if result.done:
            break

        grid = np.array(result.observation.grid)
        frames.append({
            'grid': grid.copy(),
            'score': result.observation.episode_score,
            'fruits': result.observation.episode_fruits,
            'step': step,
            'alive': result.observation.alive
        })

        import random
        action = random.randint(0, 2)
        result = client.step(SnakeAction(action=action))

    # Add final frame
    grid = np.array(result.observation.grid)
    frames.append({
        'grid': grid,
        'score': result.observation.episode_score,
        'fruits': result.observation.episode_fruits,
        'step': len(frames),
        'alive': result.observation.alive
    })

    print(f"  ✓ Collected {len(frames)} frames")

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))

    def create_colored_grid(grid):
        height, width = grid.shape
        colored_grid = np.zeros((height, width, 3))

        for i in range(height):
            for j in range(width):
                cell = grid[i, j]
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

        return colored_grid

    def update_frame(frame_idx):
        ax.clear()
        frame = frames[frame_idx]
        colored_grid = create_colored_grid(frame['grid'])

        ax.imshow(colored_grid, interpolation='nearest')
        status = "🟢 ALIVE" if frame['alive'] else "🔴 DEAD"
        ax.set_title(
            f"Snake Game Replay - {status}\n"
            f"Step: {frame['step']}, Score: {frame['score']:.1f}, Fruits: {frame['fruits']}",
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        return ax,

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=len(frames),
        interval=200,  # 200ms between frames
        repeat=True,
        blit=False
    )

    plt.tight_layout()
    plt.show()

    return anim


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
        plt.show(block=False)
        plt.pause(2)

        # Example 4: Save episode as GIF
        print("\n" + "=" * 70)
        print("  Example 4: Save Episode as Animated GIF")
        print("=" * 70)

        gif_path = record_episode_as_gif(client, num_steps=50, save_path='snake_game.gif')
        if gif_path:
            print(f"\n  ✓ You can open the GIF: open {gif_path}")

        # Example 5: Replay episode with matplotlib animation
        print("\n" + "=" * 70)
        print("  Example 5: Replay Episode Animation")
        print("=" * 70)

        anim = replay_episode_animation(client, num_steps=50)
        # Animation will play in the window

        print("\n" + "=" * 70)
        print("  Visualization Complete!")
        print("=" * 70)
        print("\n  You can visualize:")
        print("    ✓ Grid state (ASCII and matplotlib)")
        print("    ✓ Encoded observation channels")
        print("    ✓ Episode statistics and progression")
        print("    ✓ Animated GIF recordings")
        print("    ✓ Matplotlib replay animations")
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
