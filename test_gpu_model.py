#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import json
import gym
from typing import Dict, Any, Tuple, Optional, List, Union
from collections import deque
import warnings

# Set environment variables for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

# GPU detection with PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"PyTorch detected GPU: {torch.cuda.get_device_name(0)}")
        # Set PyTorch to use GPU memory efficiently
        torch.backends.cudnn.benchmark = True
    else:
        print("PyTorch: No GPU detected, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    cuda_available = False
    print("PyTorch not installed")

# GPU detection with TensorFlow
try:
    import tensorflow as tf
    # Limit TensorFlow GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow detected {len(gpus)} GPUs")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("TensorFlow: No GPU detected, using CPU")
except ImportError:
    print("TensorFlow not installed")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import Stable Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
except ImportError:
    print("Error: stable-baselines3 is not installed. Please install with: pip install stable-baselines3")
    sys.exit(1)

# Import pybullet_envs
try:
    import pybullet_envs
except ImportError:
    print("Warning: pybullet_envs not installed. Installing required for humanoid environments.")

# Custom wrapper to adapt the gym environment interface to stable-baselines3
class GymAdapter(gym.Wrapper):
    def __init__(self, env):
        super(GymAdapter, self).__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

# Wrapper to provide additional info in the environment's info dict
class InfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(InfoWrapper, self).__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        
    def reset(self) -> np.ndarray:
        self.episode_reward = 0
        self.episode_length = 0
        obs = self.env.reset()
        return obs
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        self.total_steps += 1
        
        # Add episode info to the info dict
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        info['total_steps'] = self.total_steps
        
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            info['mean_reward'] = np.mean(self.episode_rewards[-100:])
            info['mean_length'] = np.mean(self.episode_lengths[-100:])
        
        return obs, reward, done, info

def make_env(motion_file: str = None, render: bool = False, debug: bool = False):
    """
    Create a wrapped gym environment for the humanoid
    
    Args:
        motion_file: Path to the motion file
        render: Whether to render the environment
        debug: Whether to print debug information
        
    Returns:
        A wrapped gym environment
    """
    
    # Import the environment here to avoid potential import loops
    try:
        from pybullet_envs.deep_mimic.gym_env import HumanoidDeepMimicBackflipBulletEnv
    except ImportError:
        print("Error: Failed to import HumanoidDeepMimicBackflipBulletEnv. Make sure pybullet_envs is installed.")
        sys.exit(1)
    
    # Create the environment
    env = HumanoidDeepMimicBackflipBulletEnv(render=render)
    
    # Set motion file if provided
    if motion_file is not None:
        env.setMotionFile(motion_file)
    
    # Apply wrappers
    env = GymAdapter(env)
    env = InfoWrapper(env)
    
    if debug:
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    
    return env

def load_model(model_path: str, env, device: str = 'auto'):
    """
    Load a trained model
    
    Args:
        model_path: Path to the model
        env: The environment
        device: Device to run the model on ('auto', 'cuda', 'cpu')
        
    Returns:
        A trained model
    """
    try:
        if TORCH_AVAILABLE and device == 'auto':
            device = 'cuda' if cuda_available else 'cpu'
        
        print(f"Loading model from {model_path} on device: {device}")
        model = PPO.load(model_path, env=env, device=device)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Test a trained model with GPU support')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--vec_normalize_path', type=str, help='Path to saved VecNormalize statistics')
    parser.add_argument('--motion_file', type=str, default='data/Walking.json', help='Path to the motion file')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic actions')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create the environment
    env = make_env(args.motion_file, args.render, args.debug)
    
    # Wrap with DummyVecEnv for VecNormalize compatibility
    env = DummyVecEnv([lambda: env])
    
    # Load vector normalization if provided
    if args.vec_normalize_path:
        try:
            print(f"Loading VecNormalize from {args.vec_normalize_path}")
            env = VecNormalize.load(args.vec_normalize_path, env)
            # Don't update normalization statistics during testing
            env.training = False
            # Don't normalize rewards during testing
            env.norm_reward = False
            print("VecNormalize loaded successfully")
        except Exception as e:
            print(f"Error loading VecNormalize: {e}")
            sys.exit(1)
    
    # Detect best device for model
    if TORCH_AVAILABLE:
        device = 'cuda' if cuda_available else 'cpu'
    else:
        device = 'cpu'
    
    # Load the model
    model = load_model(args.model_path, env, device)
    
    # Log model information
    if args.debug:
        print(f"Model info: {model.policy}")
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nRunning {args.num_episodes} episodes with {'deterministic' if args.deterministic else 'stochastic'} actions...")
    
    for episode in range(args.num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        start_time = time.time()
        
        while not done and step_count < args.max_steps:
            # Get action from model
            action, _ = model.predict(obs, deterministic=args.deterministic)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            step_count += 1
            
            if args.debug and step_count % 100 == 0:
                print(f"Episode: {episode+1}, Step: {step_count}, Reward: {reward[0]:.4f}")
        
        episode_time = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # Get termination reason
        if step_count >= args.max_steps:
            reason = "max steps reached"
        else:
            reason = "environment termination"
        
        print(f"Episode {episode+1}: Reward={episode_reward[0]:.4f}, Steps={step_count}, Time={episode_time:.2f}s, Ended by: {reason}")
    
    # Print summary
    print("\nSummary:")
    print(f"Mean reward: {np.mean(episode_rewards):.4f}")
    print(f"Std reward: {np.std(episode_rewards):.4f}")
    print(f"Min reward: {np.min(episode_rewards):.4f}")
    print(f"Max reward: {np.max(episode_rewards):.4f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main() 