from typing import Optional

import gymnasium as gym
import numpy as np

WORLD_SIZE = 10
MAX_BATTERY = 20
MAX_STEPS = 15

class LineWorldEnv(gym.Env):

    def __init__(self, size: int = WORLD_SIZE, max_battery: int = MAX_BATTERY, max_steps: int = MAX_STEPS,):
        # The size of one-dimensional world
        self.size = size
        self.max_battery = max_battery
        self.max_steps = max_steps

        # Initialize positions - will be set randomly in reset()
        # Using -1 as "uninitialized" state
        self._rover_location = -1
        self._target_location = -1
        self._battery_level = max_battery
        self._step_count = 0
    

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "rover": gym.spaces.Discrete(self.size),
                "target": gym.spaces.Discrete(self.size),
                "battery": gym.spaces.Discrete(self.max_battery + 1)
            }
        )

        # Define what actions are available (moving left or right, or waiting)
        self.action_space = gym.spaces.Discrete(3)

        # Map action numbers to actual movements on the line
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: 0,   # Wait
            1: 1,   # Move right
            2: -1,   # Move left
        }

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with rover and target positions and battery level
        """
        return {"rover": self._rover_location, "target": self._target_location, "battery": self._battery_level}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between rover and target and battery level
        """
        return {
            "distance_to_target": abs(self._rover_location - self._target_location),
            "battery_level": self._battery_level,
            "step_count": self._step_count
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self._battery_level = self.max_battery
        self._step_count = 0

        # Randomly place the rover anywhere on the line
        self._rover_location = int(self.np_random.integers(0, self.size, dtype=int))

        # Randomly place target, ensuring it's different from rover position
        self._target_location = self._rover_location
        while self._target_location == self._rover_location:
            self._target_location = int(self.np_random.integers(0, self.size, dtype=int))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-2 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-2) to a movement direction
        direction = self._action_to_direction[action]

        # Update rover position, ensuring it stays within bounds
        # np.clip prevents the agent from walking off the edge
        self._rover_location = int(
            np.clip(
                self._rover_location + direction,
                0,
                self.size - 1
        ))

        self._battery_level -= 1
        self._step_count += 1

        # Check if rover reached the target
        reached_target = self._rover_location == self._target_location

        # Check if battery is depleted
        battery_depleted = self._battery_level <= 0

        # Reward structure: +100 for reaching target, -100 for battery dying, -1 for every timestep
        terminated = reached_target or battery_depleted
        truncated = self._step_count >= self.max_steps and not terminated

        if reached_target:
            reward = 100
        elif battery_depleted:
            reward = -100
        else:
            reward = -1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

