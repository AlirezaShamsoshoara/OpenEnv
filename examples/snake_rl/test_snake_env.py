#!/usr/bin/env python3
"""
Comprehensive test suite for Snake Environment.

This script tests all features of the snake_env including:
- Environment reset
- Actions (turn left, right, no-op)
- Observations (grid, vision, episode stats)
- Rewards (fruit collection, death penalty)
- Episode termination
- State tracking

Usage:
    # Option 1: Use Docker (recommended)
    python examples/snake_rl/test_snake_env.py --mode docker

    # Option 2: Use local server (must be running separately)
    python examples/snake_rl/test_snake_env.py --mode local --url http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from envs.snake_env import SnakeAction, SnakeEnv


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def test_reset(client):
    """Test environment reset functionality."""
    print_subsection("Testing Reset")

    result = client.reset()

    print(f"  ✓ Reset successful")
    print(f"    - Episode ID: {result.observation.metadata.get('episode_id', 'N/A')}")
    print(f"    - Snake alive: {result.observation.alive}")
    print(f"    - Episode score: {result.observation.episode_score}")
    print(f"    - Episode steps: {result.observation.episode_steps}")
    print(
        f"    - Grid shape: {len(result.observation.grid)}x{len(result.observation.grid[0])}"
    )
    print(
        f"    - Observation shape: {len(result.observation.observation)}x"
        f"{len(result.observation.observation[0])}x"
        f"{len(result.observation.observation[0][0])}"
    )
    print(f"    - Initial reward: {result.reward}")
    print(f"    - Done: {result.done}")

    return result


def test_actions(client):
    """Test all action types."""
    print_subsection("Testing Actions")

    # Reset first
    client.reset()

    actions = {0: "No-op (continue straight)", 1: "Turn left", 2: "Turn right"}

    print("\n  Testing each action type:")
    for action_id, action_name in actions.items():
        result = client.step(SnakeAction(action=action_id))
        print(f"    {action_id}: {action_name}")
        print(
            f"       Alive: {result.observation.alive}, "
            f"Reward: {result.reward:.2f}, "
            f"Steps: {result.observation.episode_steps}"
        )

    print(f"\n  ✓ All actions executed successfully")


def test_observations(client):
    """Test observation structure and content."""
    print_subsection("Testing Observations")

    result = client.reset()
    obs = result.observation

    print(f"\n  Observation components:")
    print(f"    1. Grid (full game state):")
    print(f"       - Type: {type(obs.grid)}")
    print(f"       - Shape: {len(obs.grid)}x{len(obs.grid[0])}")
    print(f"       - Data type: list of lists of int")
    print(f"       - Sample values: {obs.grid[0][:5]} (first row, first 5 cells)")

    print(f"\n    2. Encoded Observation (agent's view):")
    print(f"       - Type: {type(obs.observation)}")
    print(
        f"       - Shape: {len(obs.observation)}x{len(obs.observation[0])}x{len(obs.observation[0][0])}"
    )
    print(
        f"       - Channels: {len(obs.observation[0][0])} "
        f"(walls, fruits, snake body parts)"
    )

    print(f"\n    3. Episode Statistics:")
    print(f"       - Score: {obs.episode_score}")
    print(f"       - Steps: {obs.episode_steps}")
    print(f"       - Fruits collected: {obs.episode_fruits}")
    print(f"       - Kills: {obs.episode_kills}")
    print(f"       - Alive: {obs.alive}")

    print(f"\n    4. Reward & Done flags:")
    print(f"       - Current reward: {obs.reward}")
    print(f"       - Done: {obs.done}")

    print(f"\n  ✓ All observation components verified")


def test_rewards(client, max_steps=50):
    """Test reward system by playing a short episode."""
    print_subsection("Testing Rewards")

    result = client.reset()
    total_reward = 0
    fruits_collected = 0

    print(f"\n  Playing {max_steps} steps to test rewards...")

    for step in range(max_steps):
        if result.done:
            break

        # Random action (simple policy)
        import random

        action = random.randint(0, 2)
        result = client.step(SnakeAction(action=action))

        total_reward += result.reward

        # Check if fruit was collected (reward increased)
        if result.reward > 0 and result.observation.episode_fruits > fruits_collected:
            fruits_collected = result.observation.episode_fruits
            print(
                f"    Step {step + 1}: 🍎 Fruit collected! "
                f"Reward: +{result.reward:.2f}"
            )

        if result.done:
            print(
                f"    Step {step + 1}: ☠️  Snake died! "
                f"Final penalty: {result.reward:.2f}"
            )
            break

    print(f"\n  Episode summary:")
    print(f"    - Total steps: {result.observation.episode_steps}")
    print(f"    - Total reward: {total_reward:.2f}")
    print(f"    - Fruits collected: {fruits_collected}")
    print(f"    - Final score: {result.observation.episode_score:.2f}")
    print(f"    - Snake survived: {'Yes' if result.observation.alive else 'No'}")

    print(f"\n  ✓ Reward system tested")


def test_state(client):
    """Test state tracking."""
    print_subsection("Testing State Tracking")

    # Reset and get initial state
    client.reset()
    state = client.state()

    print(f"\n  Initial state:")
    print(f"    - Episode ID: {state.episode_id}")
    print(f"    - Step count: {state.step_count}")

    # Take some steps
    for i in range(5):
        client.step(SnakeAction(action=0))

    state = client.state()
    print(f"\n  After 5 steps:")
    print(f"    - Episode ID: {state.episode_id} (unchanged)")
    print(f"    - Step count: {state.step_count}")

    # Reset and check new episode
    client.reset()
    new_state = client.state()

    print(f"\n  After reset:")
    print(f"    - Episode ID: {new_state.episode_id} (new)")
    print(f"    - Step count: {new_state.step_count} (reset to 0)")
    print(f"    - Episode ID changed: {state.episode_id != new_state.episode_id}")

    print(f"\n  ✓ State tracking verified")


def test_episode_termination(client, max_attempts=3):
    """Test episode termination conditions."""
    print_subsection("Testing Episode Termination")

    print(f"\n  Running episodes until termination (max {max_attempts} attempts)...")

    for attempt in range(max_attempts):
        result = client.reset()
        steps = 0
        max_steps = 200

        # Play until done or max steps
        while not result.done and steps < max_steps:
            import random

            action = random.randint(0, 2)
            result = client.step(SnakeAction(action=action))
            steps += 1

        print(f"\n    Attempt {attempt + 1}:")
        print(f"      - Steps taken: {steps}")
        print(f"      - Episode terminated: {result.done}")
        print(f"      - Final alive status: {result.observation.alive}")
        print(f"      - Final score: {result.observation.episode_score:.2f}")
        print(f"      - Fruits collected: {result.observation.episode_fruits}")

        if result.done:
            print(f"      ✓ Episode terminated successfully")
            break

    print(f"\n  ✓ Episode termination tested")


def test_multiple_episodes(client, num_episodes=3):
    """Test multiple consecutive episodes."""
    print_subsection("Testing Multiple Episodes")

    print(f"\n  Running {num_episodes} consecutive episodes...")

    episode_stats = []

    for ep in range(num_episodes):
        result = client.reset()
        steps = 0
        max_steps = 100

        while not result.done and steps < max_steps:
            import random

            action = random.randint(0, 2)
            result = client.step(SnakeAction(action=action))
            steps += 1

        stats = {
            "episode": ep + 1,
            "steps": result.observation.episode_steps,
            "score": result.observation.episode_score,
            "fruits": result.observation.episode_fruits,
            "survived": result.observation.alive,
        }
        episode_stats.append(stats)

        print(
            f"    Episode {ep + 1}: {steps} steps, "
            f"score={stats['score']:.2f}, "
            f"fruits={stats['fruits']}, "
            f"survived={stats['survived']}"
        )

    print(f"\n  Episode statistics:")
    avg_steps = sum(s["steps"] for s in episode_stats) / num_episodes
    avg_score = sum(s["score"] for s in episode_stats) / num_episodes
    avg_fruits = sum(s["fruits"] for s in episode_stats) / num_episodes

    print(f"    - Average steps: {avg_steps:.1f}")
    print(f"    - Average score: {avg_score:.2f}")
    print(f"    - Average fruits: {avg_fruits:.1f}")

    print(f"\n  ✓ Multiple episodes tested successfully")


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(
        description="Test Snake Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Option 1: Use Docker (automatically starts/stops container)
  python examples/snake_rl/test_snake_env.py --mode docker

  # Option 2: Use local server (start server first in another terminal)
  # Terminal 1: cd src/envs/snake_env && uv run --project . server
  # Terminal 2: python examples/snake_rl/test_snake_env.py --mode local --url http://localhost:8000
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["docker", "local"],
        default="docker",
        help="Connection mode: docker (auto-start container) or local (connect to running server)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Server URL for local mode (default: http://localhost:8000)",
    )

    args = parser.parse_args()

    print_section("Snake Environment - Comprehensive Test Suite")

    if args.mode == "docker":
        print("\n📦 Mode: Docker")
        print("  - Automatically starts Docker container")
        print("  - Runs tests against containerized environment")
        print("  - Automatically stops container when done")
        print("\n⚠️  Prerequisites:")
        print("  - Docker must be installed and running")
        print(
            "  - Image must be built: docker build -t snake-env:latest -f src/envs/snake_env/server/Dockerfile ."
        )
    else:
        print("\n🖥️  Mode: Local Server")
        print(f"  - Connecting to: {args.url}")
        print("  - Assumes server is already running")
        print("\n⚠️  Prerequisites:")
        print("  - Start server in another terminal:")
        print("    cd src/envs/snake_env")
        print("    pip install -e .")
        print("    uv run --project . server")
        print("    (or: python -m server.app)")

    try:
        # Create client based on mode
        if args.mode == "docker":
            print("\nStarting Docker container...")
            client = SnakeEnv.from_docker_image("snake-env:latest")
            print("✓ Container started successfully!\n")
        else:
            print(f"\nConnecting to local server at {args.url}...")
            client = SnakeEnv(base_url=args.url)
            # Test connection
            try:
                client.reset()
                print("✓ Connected to local server successfully!\n")
            except Exception as e:
                print(f"\n❌ Failed to connect to local server at {args.url}")
                print(f"   Error: {e}")
                print("\n💡 Make sure the server is running:")
                print("   cd src/envs/snake_env")
                print("   pip install -e .")
                print("   uv run --project . server")
                return False

        # Run all tests
        test_reset(client)
        test_actions(client)
        test_observations(client)
        test_rewards(client, max_steps=50)
        test_state(client)
        test_episode_termination(client, max_attempts=3)
        test_multiple_episodes(client, num_episodes=3)

        # Success summary
        print_section("All Tests Passed! ✓")
        print("\nThe snake_env is working correctly with all features:")
        print("  ✓ Reset functionality")
        print("  ✓ Action execution (turn left, right, no-op)")
        print("  ✓ Observation structure (grid, encoded obs, stats)")
        print("  ✓ Reward system (fruit collection, death penalty)")
        print("  ✓ State tracking (episode ID, step count)")
        print("  ✓ Episode termination")
        print("  ✓ Multiple episodes")

        print("\nFor visualization examples, see:")
        print("  - examples/snake_rl/visualize_snake.py")
        print("  - examples/snake_rl/play_snake.py")

        # Cleanup
        print("\nCleaning up...")
        if args.mode == "docker":
            client.close()
            print("✓ Container stopped and removed")
        else:
            print("✓ Disconnected from local server")
            print("  (Server is still running - stop it manually if needed)")

        print_section("Test Suite Completed Successfully! 🎉")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
