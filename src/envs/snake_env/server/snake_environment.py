# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Snake Environment Implementation.

A multi-agent snake game environment that wraps marlenv's Snake-v1.
This implementation provides a single-agent interface by wrapping the
multi-agent marlenv environment.
"""

from uuid import uuid4
import numpy as np
import gym
import marlenv

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from core.env_server.interfaces import Environment
    from core.env_server.types import State
    from ..models import SnakeAction, SnakeObservation
except ImportError:
    # Standalone imports (when environment is standalone with openenv-core from pip)
    from openenv_core.env_server.interfaces import Environment
    from openenv_core.env_server.types import State
    from models import SnakeAction, SnakeObservation


class SnakeEnvironment(Environment):
    """
    A snake game environment that wraps marlenv's Snake-v1.

    This environment provides a single-agent interface to the multi-agent
    snake game. The snake must navigate a grid, eat fruits, and avoid walls
    and its own body.

    Args:
        height: Height of the grid map (default: 20)
        width: Width of the grid map (default: 20)
        snake_length: Initial length of the snake (default: 3)
        vision_range: Vision range for partial observability (default: None for full grid)
        observer: 'snake' for relative actions or 'human' for global directions (default: 'snake')
        max_episode_steps: Maximum steps per episode (default: 1000)
        reward_dict: Custom reward function (default: fruit=1.0, others=0.0)

    Example:
        >>> env = SnakeEnvironment()
        >>> obs = env.reset()
        >>> print(obs.alive)  # True
        >>>
        >>> obs = env.step(SnakeAction(action=1))  # Turn left
        >>> print(obs.episode_score)
        >>> print(obs.reward)
    """

    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        snake_length: int = 3,
        vision_range: int = None,
        observer: str = "snake",
        max_episode_steps: int = 1000,
        reward_dict: dict = None,
    ):
        """Initialize the snake environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Default reward function
        if reward_dict is None:
            reward_dict = {
                "fruit": 1.0,
                "kill": 0.0,
                "lose": -1.0,
                "win": 0.0,
                "time": 0.0,
            }

        # Create the marlenv snake environment for single agent
        self.env = gym.make(
            "Snake-v1",
            height=height,
            width=width,
            num_snakes=1,  # Single agent
            snake_length=snake_length,
            vision_range=vision_range,
            frame_stack=1,
            observer=observer,
            reward_dict=reward_dict,
            max_episode_steps=max_episode_steps,
        )

        # Wrap with SingleAgent wrapper to unwrap list returns
        self.env = marlenv.wrappers.SingleAgent(self.env)

        # Track episode statistics
        self._episode_score = 0.0
        self._episode_fruits = 0
        self._episode_kills = 0

    def reset(self) -> SnakeObservation:
        """
        Reset the environment.

        Returns:
            SnakeObservation with initial game state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_score = 0.0
        self._episode_fruits = 0
        self._episode_kills = 0

        # Reset the marlenv environment
        obs = self.env.reset()

        # Convert observation to list format
        obs_list = obs.tolist() if isinstance(obs, np.ndarray) else obs

        # Get the grid from the environment
        grid = self.env.grid.tolist() if hasattr(self.env, "grid") else []

        return SnakeObservation(
            grid=grid,
            observation=obs_list,
            episode_score=self._episode_score,
            episode_steps=self._state.step_count,
            episode_fruits=self._episode_fruits,
            episode_kills=self._episode_kills,
            alive=True,
            done=False,
            reward=0.0,
        )

    def step(self, action: SnakeAction) -> SnakeObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: SnakeAction containing the action to take

        Returns:
            SnakeObservation with the result of the action
        """
        self._state.step_count += 1

        # Execute action in marlenv
        obs, reward, done, info = self.env.step(action.action)

        # Update episode statistics
        self._episode_score += reward

        # Convert observation to list format
        obs_list = obs.tolist() if isinstance(obs, np.ndarray) else obs

        # Get the grid from the environment
        grid = self.env.grid.tolist() if hasattr(self.env, "grid") else []

        # Extract episode statistics from info if available
        episode_fruits = info.get("episode_fruits", [self._episode_fruits])[0] if "episode_fruits" in info else self._episode_fruits
        episode_kills = info.get("episode_kills", [self._episode_kills])[0] if "episode_kills" in info else self._episode_kills

        return SnakeObservation(
            grid=grid,
            observation=obs_list,
            episode_score=self._episode_score,
            episode_steps=self._state.step_count,
            episode_fruits=int(episode_fruits),
            episode_kills=int(episode_kills),
            alive=not done,
            done=done,
            reward=float(reward),
            metadata={"info": info},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
