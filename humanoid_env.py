import os
import gym
import numpy as np
import pybullet as p
from pybullet_envs.deep_mimic.gym_env import HumanoidDeepMimicBackflipBulletEnv
from gym import spaces

class HumanoidImitationEnv(gym.Env):
    """
    A wrapper for PyBullet's DeepMimic humanoid environment that adds additional functionality
    for imitation learning, including reward scaling, action/observation scaling, and debug info.
    """
    
    def __init__(self, renders=False, motion_file='data/Walking.json', rescale_actions=False, 
                 rescale_observations=False):
        """
        Initialize the humanoid imitation environment.
        
        Args:
            renders (bool): Whether to render the environment
            motion_file (str): Path to the motion file to imitate
            rescale_actions (bool): Whether to rescale actions to [-1, 1]
            rescale_observations (bool): Whether to rescale observations
        """
        # Initialize DeepMimic humanoid environment from PyBullet
        # Store motion file if provided to use it later
        self.motion_file = motion_file
        
        # Create the environment - PyBullet env takes motion file through env variables
        if motion_file:
            # Set the environment variable for the motion file before creating the env
            os.environ["PYBULLET_DEEP_MIMIC_MOTION_FILE"] = motion_file
        
        self.env = HumanoidDeepMimicBackflipBulletEnv(renders=renders)
        
        # Set action and observation spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Store action and observation scaling preferences
        self.rescale_actions = rescale_actions
        self.rescale_observations = rescale_observations
        
        # If rescaling actions, redefine action space to be within [-1, 1]
        if self.rescale_actions:
            # DeepMimic has action bounds specific to joint limits
            # We'll map these to [-1, 1] for more effective learning
            self.original_action_low = self.env.action_space.low
            self.original_action_high = self.env.action_space.high
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.env.action_space.shape,
                dtype=np.float32
            )
        
        # If rescaling observations, compute running statistics for normalization
        if self.rescale_observations:
            self.obs_running_mean = None
            self.obs_running_var = None
            self.obs_count = 0
            # We'll update these during training
        
        # Debug mode can be enabled to print additional information
        self.debug = False
        self.total_reward = 0
        self.episode_steps = 0
        self.reward_components = {}
        
        # Metadata for rendering
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 60
        }

    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action (numpy.ndarray): The action to take
            
        Returns:
            observation (numpy.ndarray): The next observation
            reward (float): The reward for the transition
            terminated (bool): Whether the episode is terminated
            truncated (bool): Whether the episode is truncated
            info (dict): Additional information about the transition
        """
        # If rescaling actions, convert from [-1, 1] to original action space
        if self.rescale_actions:
            # Denormalize action from [-1, 1] to original range
            action = self._denormalize_action(action)
        
        # Take a step in the underlying environment
        # Handle both old and new gym API
        result = self.env.step(action)
        
        # Parse result based on its length
        if len(result) == 4:  # Old gym API: obs, reward, done, info
            obs, reward, done, info = result
            terminated, truncated = done, False
        elif len(result) == 5:  # New gym API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            raise ValueError(f"Unexpected result length from environment step: {len(result)}")
        
        # Extract reward components if available from DeepMimic
        reward_components = {}
        if hasattr(self.env, 'rewards'):
            reward_components = self.env.rewards
            
        # Track reward components for debugging
        for key, value in reward_components.items():
            if key not in self.reward_components:
                self.reward_components[key] = value
            else:
                self.reward_components[key] += value
                
        # If rescaling observations, normalize the observation
        if self.rescale_observations:
            obs = self._normalize_observation(obs)
        
        # Track episode progress for debugging
        self.total_reward += reward
        self.episode_steps += 1
        
        # Add debug info
        info['episode_steps'] = self.episode_steps
        info['total_reward'] = self.total_reward
        info['reward_components'] = reward_components
        
        # Print debug information if enabled
        if self.debug and (done or self.episode_steps % 100 == 0):
            print(f"Step {self.episode_steps}: reward={reward:.3f}, total={self.total_reward:.3f}")
            if reward_components:
                components_str = ", ".join([f"{k}={v:.3f}" for k, v in reward_components.items()])
                print(f"Reward components: {components_str}")
        
        # Return result in the same format as received (for compatibility)
        if len(result) == 4:
            return obs, reward, done, info
        else:
            return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation.
        
        Args:
            seed (int, optional): The random seed
            options (dict, optional): Additional options for resetting
            
        Returns:
            observation (numpy.ndarray): The initial observation
            info (dict): Additional information (for newer gym versions)
        """
        # Set seed if provided
        if seed is not None:
            self.seed(seed)
        
        # Reset the underlying environment, handling both old and new gym API
        obs = None
        info = {}
        try:
            # Try new gym API (returns obs and info)
            if options is not None:
                result = self.env.reset(seed=seed, options=options)
            else:
                result = self.env.reset(seed=seed)
                
            # Check if reset returned a tuple (obs, info) or just obs
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs = result
                
        except TypeError:
            # Fall back to old gym API (returns just obs)
            obs = self.env.reset()
        
        # Reset episode tracking variables
        self.total_reward = 0
        self.episode_steps = 0
        self.reward_components = {}
        
        # If rescaling observations, normalize the observation
        if self.rescale_observations:
            obs = self._normalize_observation(obs)
        
        # For gym 0.21.0 compatibility
        if hasattr(self, 'return_info') and self.return_info:
            return obs, info
        return obs

    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The rendering mode
            
        Returns:
            The rendering result
        """
        return self.env.render(mode)

    def close(self):
        """
        Close the environment and release resources.
        """
        self.env.close()

    def seed(self, seed=None):
        """
        Set the random seed for the environment.
        
        Args:
            seed (int): The random seed
            
        Returns:
            The seed used
        """
        return self.env.seed(seed)

    def _denormalize_action(self, action):
        """
        Convert an action from [-1, 1] to the original action space.
        
        Args:
            action (numpy.ndarray): The normalized action
            
        Returns:
            numpy.ndarray: The denormalized action
        """
        # Map from [-1, 1] to [original_low, original_high]
        action_range = (self.original_action_high - self.original_action_low) / 2.0
        action_middle = (self.original_action_high + self.original_action_low) / 2.0
        
        return action * action_range + action_middle

    def _normalize_observation(self, obs):
        """
        Normalize the observation using running statistics.
        
        Args:
            obs (numpy.ndarray): The observation to normalize
            
        Returns:
            numpy.ndarray: The normalized observation
        """
        # Initialize running statistics if not already done
        if self.obs_running_mean is None:
            self.obs_running_mean = np.zeros_like(obs, dtype=np.float64)
            self.obs_running_var = np.ones_like(obs, dtype=np.float64)
            self.obs_count = 0
        
        # Update running statistics using Welford's algorithm
        self.obs_count += 1
        delta = obs - self.obs_running_mean
        self.obs_running_mean += delta / self.obs_count
        delta2 = obs - self.obs_running_mean
        self.obs_running_var += delta * delta2
        
        # Compute standard deviation with a small epsilon for numerical stability
        std = np.sqrt(self.obs_running_var / max(1, self.obs_count - 1)) + 1e-8
        
        # Normalize observation
        normalized_obs = (obs - self.obs_running_mean) / std
        
        # Clip to a reasonable range to avoid extremely large values
        normalized_obs = np.clip(normalized_obs, -10.0, 10.0)
        
        return normalized_obs

    def get_reward_components(self):
        """
        Get the breakdown of the reward components.
        
        Returns:
            dict: The reward components
        """
        return self.reward_components

    def update_motion_file(self, motion_file):
        """
        Update the motion file being imitated.
        
        Args:
            motion_file (str): Path to the new motion file
        """
        # PyBullet doesn't support switching motion files at runtime directly
        # We would need to recreate the environment
        self.motion_file = motion_file
        os.environ["PYBULLET_DEEP_MIMIC_MOTION_FILE"] = motion_file
        print(f"Warning: Motion file updated to {motion_file}, but you'll need to reset the environment for it to take effect.")
        
    def get_pose_info(self):
        """
        Get information about the current pose of the humanoid.
        
        Returns:
            dict: Information about the current pose
        """
        if hasattr(self.env, 'poseInfo'):
            return self.env.poseInfo
        else:
            return None 