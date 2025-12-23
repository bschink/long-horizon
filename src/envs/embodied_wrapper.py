"""Embodied-compatible wrapper for BoxMovingEnv.

This module provides a wrapper that converts the functional JAX-based BoxMovingEnv
to the object-oriented interface expected by the Recall2Imagine (R2I) agent.
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add recall2imagine's PARENT directory to path so it can be imported as a package
_r2i_parent = Path(__file__).resolve().parent.parent.parent
_r2i_path = _r2i_parent / 'recall2imagine'
if _r2i_path.exists():
    if str(_r2i_parent) not in sys.path:
        sys.path.insert(0, str(_r2i_parent))
    if str(_r2i_path) not in sys.path:
        sys.path.insert(0, str(_r2i_path))

import embodied

from .block_moving.block_moving_env import BoxMovingEnv
from .block_moving.env_types import BoxMovingConfig, GridStatesEnum


# Number of possible grid states for one-hot encoding
NUM_GRID_STATES = 12


def one_hot_encode_grid(grid: np.ndarray, num_classes: int = NUM_GRID_STATES) -> np.ndarray:
    """One-hot encode a grid of integer values.
    
    Args:
        grid: Integer grid of shape (H, W) with values in [0, num_classes-1]
        num_classes: Number of classes for one-hot encoding
        
    Returns:
        One-hot encoded grid of shape (H, W, num_classes)
    """
    grid_int = np.asarray(grid, dtype=np.int32)
    one_hot = np.eye(num_classes, dtype=np.float32)[grid_int]
    return one_hot


class EmbodiedBoxMovingWrapper(embodied.Env):
    """Wraps BoxMovingEnv to be compatible with embodied.Env interface.
    
    This wrapper:
    - Converts functional JAX env to object-oriented style
    - One-hot encodes grid states
    - Supports both MLP (flattened) and CNN (2D) observation formats
    - Includes separate 'grid' and 'goal' observation keys
    """
    
    def __init__(
        self,
        env_config: BoxMovingConfig,
        mlp_flatten_grid: bool = True,
        cnn_2d_grid: bool = False,
        seed: int = 0,
    ):
        """Initialize the wrapper.
        
        Args:
            env_config: Configuration for BoxMovingEnv
            mlp_flatten_grid: If True, include flattened grid as 'grid' observation
            cnn_2d_grid: If True, include 2D grid as 'grid_2d' observation
            seed: Random seed for environment
        """
        self._env = BoxMovingEnv(
            grid_size=env_config.grid_size,
            episode_length=env_config.episode_length,
            number_of_boxes_min=env_config.number_of_boxes_min,
            number_of_boxes_max=env_config.number_of_boxes_max,
            number_of_moving_boxes_max=env_config.number_of_moving_boxes_max,
            terminate_when_success=env_config.terminate_when_success,
            dense_rewards=env_config.dense_rewards,
            negative_sparse=env_config.negative_sparse,
            level_generator=env_config.level_generator,
            generator_special=env_config.generator_special,
            quarter_size=env_config.quarter_size,
        )
        
        self._mlp_flatten_grid = mlp_flatten_grid
        self._cnn_2d_grid = cnn_2d_grid
        self._grid_size = env_config.grid_size
        self._episode_length = env_config.episode_length
        
        # Initialize random state - use numpy RNG for seed management
        # to avoid JAX device transfer issues during training
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._reset_counter = 0
        
        # Current environment state (will be set on reset)
        self._state = None
        self._done = True
        self._step_count = 0
        
    def __len__(self):
        return 0  # Single env, not batched
    
    @functools.cached_property
    def obs_space(self):
        """Return observation space dictionary."""
        spaces = {}
        
        # Grid observation (one-hot encoded)
        grid_one_hot_size = self._grid_size * self._grid_size * NUM_GRID_STATES
        
        if self._mlp_flatten_grid:
            # Flattened grid for MLP encoder
            spaces['grid'] = embodied.Space(
                np.float32, 
                (grid_one_hot_size,),
                low=0.0, 
                high=1.0
            )
            # Goal in same format
            spaces['goal'] = embodied.Space(
                np.float32,
                (grid_one_hot_size,),
                low=0.0,
                high=1.0
            )
            
        if self._cnn_2d_grid:
            # 2D grid for CNN encoder
            spaces['grid_2d'] = embodied.Space(
                np.float32,
                (self._grid_size, self._grid_size, NUM_GRID_STATES),
                low=0.0,
                high=1.0
            )
            # Goal in same format
            spaces['goal_2d'] = embodied.Space(
                np.float32,
                (self._grid_size, self._grid_size, NUM_GRID_STATES),
                low=0.0,
                high=1.0
            )
        
        # Required keys for R2I
        spaces['reward'] = embodied.Space(np.float32)
        spaces['is_first'] = embodied.Space(bool)
        spaces['is_last'] = embodied.Space(bool)
        spaces['is_terminal'] = embodied.Space(bool)
        
        return spaces
    
    @functools.cached_property
    def act_space(self):
        """Return action space dictionary."""
        return {
            'action': embodied.Space(np.int32, (), 0, self._env.action_space),
            'reset': embodied.Space(bool),
        }
    
    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action: Dictionary with 'action' and 'reset' keys
            
        Returns:
            Observation dictionary
        """
        if action['reset'] or self._done:
            return self._reset()
        
        # Temporarily allow transfers for environment operations (including _make_obs)
        old_guard = jax.config.jax_transfer_guard
        try:
            jax.config.update('jax_transfer_guard', 'allow')
            
            # Take action in environment
            action_int = int(action['action'])
            self._state, reward, done, info = self._env.step(self._state, action_int)
            self._step_count += 1
            
            # Check for truncation
            truncated = info.get('truncated', False)
            if isinstance(truncated, jnp.ndarray):
                truncated = bool(truncated)
            if isinstance(done, jnp.ndarray):
                done = bool(done)
            
            is_last = done or truncated
            is_terminal = done  # Terminal means actual end, not truncation
            
            self._done = is_last
            
            return self._make_obs(
                reward=float(reward),
                is_first=False,
                is_last=is_last,
                is_terminal=is_terminal,
            )
        finally:
            jax.config.update('jax_transfer_guard', old_guard)
    
    def _reset(self):
        """Reset the environment."""
        # Generate a new random seed using numpy and create JAX key on CPU
        # Use transfer_guard context to allow the operation since jaxagent sets it to 'disallow'
        new_seed = int(self._rng.integers(0, 2**31))
        
        # Temporarily allow transfers for environment operations (including _make_obs)
        old_guard = jax.config.jax_transfer_guard
        try:
            jax.config.update('jax_transfer_guard', 'allow')
            reset_key = jax.random.PRNGKey(new_seed)
            self._state, info = self._env.reset(reset_key)
        
            self._done = False
            self._step_count = 0
            self._reset_counter += 1
            
            return self._make_obs(
                reward=0.0,
                is_first=True,
                is_last=False,
                is_terminal=False,
            )
        finally:
            jax.config.update('jax_transfer_guard', old_guard)
    
    def _make_obs(self, reward, is_first, is_last, is_terminal):
        """Create observation dictionary from current state.
        
        Args:
            reward: Reward value
            is_first: Whether this is the first step of an episode
            is_last: Whether this is the last step of an episode
            is_terminal: Whether episode ended (not just truncated)
            
        Returns:
            Observation dictionary
        """
        # Get grid and goal from state
        grid = np.asarray(self._state.grid)
        goal = np.asarray(self._state.goal)
        
        # One-hot encode
        grid_one_hot = one_hot_encode_grid(grid)
        goal_one_hot = one_hot_encode_grid(goal)
        
        obs = {
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }
        
        if self._mlp_flatten_grid:
            obs['grid'] = grid_one_hot.flatten().astype(np.float32)
            obs['goal'] = goal_one_hot.flatten().astype(np.float32)
            
        if self._cnn_2d_grid:
            obs['grid_2d'] = grid_one_hot.astype(np.float32)
            obs['goal_2d'] = goal_one_hot.astype(np.float32)
        
        return obs
    
    def render(self):
        """Render is not supported."""
        raise NotImplementedError("Render not supported for BoxMovingEnv wrapper")
    
    def close(self):
        """Clean up resources."""
        pass


def make_box_moving_env(config, **overrides):
    """Factory function to create a BoxMovingEnv with embodied wrapper.
    
    Args:
        config: Configuration object with env settings
        **overrides: Override any config values
        
    Returns:
        EmbodiedBoxMovingWrapper instance
    """
    # Extract environment config from env.box_moving
    # Handle both dict-style and attribute-style access
    if hasattr(config, 'env') and hasattr(config.env, 'box_moving'):
        env_cfg = config.env.box_moving
    elif hasattr(config, 'env') and isinstance(config.env, dict) and 'box_moving' in config.env:
        env_cfg = config.env['box_moving']
    else:
        env_cfg = {}
    
    # Helper to get config value with fallback
    def get_env_val(key, default):
        if hasattr(env_cfg, key):
            return getattr(env_cfg, key)
        elif isinstance(env_cfg, dict) and key in env_cfg:
            return env_cfg[key]
        return default
    
    env_config = BoxMovingConfig(
        grid_size=get_env_val('grid_size', 5),
        episode_length=get_env_val('episode_length', 100),
        number_of_boxes_min=get_env_val('number_of_boxes_min', 3),
        number_of_boxes_max=get_env_val('number_of_boxes_max', 4),
        number_of_moving_boxes_max=get_env_val('number_of_moving_boxes_max', 2),
        terminate_when_success=get_env_val('terminate_when_success', False),
        dense_rewards=get_env_val('dense_rewards', False),
        negative_sparse=get_env_val('negative_sparse', False),
        level_generator=get_env_val('level_generator', 'default'),
        generator_special=get_env_val('generator_special', False),
        quarter_size=get_env_val('quarter_size', None),
    )
    
    # Get observation format settings
    mlp_flatten_grid = getattr(config, 'mlp_flatten_grid', True)
    cnn_2d_grid = getattr(config, 'cnn_2d_grid', False)
    seed = getattr(config, 'seed', 0)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(env_config, key):
            env_config = env_config.replace(**{key: value})
    
    return EmbodiedBoxMovingWrapper(
        env_config=env_config,
        mlp_flatten_grid=mlp_flatten_grid,
        cnn_2d_grid=cnn_2d_grid,
        seed=seed,
    )

