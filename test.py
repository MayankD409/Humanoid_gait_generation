import os
import gym
import time
import argparse
import numpy as np

# Suppress TensorFlow warnings and explicitly configure GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Use private GPU thread mode

# First check for GPU using PyTorch (more reliable GPU detection)
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU is available: {gpu_count} GPU(s) detected")
        print(f"GPU Name: {gpu_name}")
        
        # Tell TensorFlow which GPU to use (even if it can't detect it properly)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        print("No GPU detected by PyTorch, using CPU")
except ImportError:
    gpu_available = False
    print("PyTorch not available for GPU detection, will try with TensorFlow")

# Import TensorFlow after PyTorch GPU check
try:
    import tensorflow as tf
    # Only try TensorFlow GPU setup if PyTorch didn't detect GPU
    if not gpu_available:
        # Set memory growth to avoid consuming all GPU memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU is available: {len(gpus)} GPU(s) detected by TensorFlow")
        else:
            print("No GPU detected by TensorFlow, using CPU")
except ImportError:
    print("TensorFlow not imported, warnings may still appear")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from humanoid_env import HumanoidImitationEnv

# Custom wrapper to handle the gym 0.26.2 API with stable-baselines3 1.2.0
class GymAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        # Handle the gym 0.26.2 reset which returns (obs, info)
        # but stable-baselines3 1.2.0 expects just obs
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0]  # Just return the observation
        return result
        
    def step(self, action):
        # Handle the gym 0.26.2 step which returns (obs, reward, terminated, truncated, info)
        # but stable-baselines3 1.2.0 expects (obs, reward, done, info)
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        return result

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, 
                    help='Path to the trained model file')
parser.add_argument('--vec_normalize_path', type=str, required=True, 
                    help='Path to the saved VecNormalize stats')
parser.add_argument('--motion_file', type=str, default='data/Walking.json', 
                    help='Path to motion file to test on')
parser.add_argument('--episodes', type=int, default=5, 
                    help='Number of episodes to run')
parser.add_argument('--render', action='store_true', 
                    help='Render the environment')
parser.add_argument('--deterministic', action='store_true', 
                    help='Use deterministic actions')
args = parser.parse_args()

# Create environment
env = HumanoidImitationEnv(renders=args.render, motion_file=args.motion_file,
                           rescale_actions=True, rescale_observations=True)
# Wrap with our adapter
env = GymAdapter(env)

# Wrap in dummy vec env for compatibility with VecNormalize
env = DummyVecEnv([lambda: env])

# Load the saved VecNormalize statistics
env = VecNormalize.load(args.vec_normalize_path, env)
# Disable training and reward normalization (testing only)
env.training = False
env.norm_reward = False

# Load the trained agent
model = PPO.load(args.model_path)

# Run test episodes
for episode in range(args.episodes):
    obs = env.reset()
    episode_reward = 0
    episode_steps = 0
    done = False
    
    print(f"Starting episode {episode + 1}/{args.episodes}")
    start_time = time.time()
    
    while not done:
        # Use the model to predict actions
        action, _states = model.predict(obs, deterministic=args.deterministic)
        
        # Take step in environment
        obs, rewards, done, info = env.step(action)
        
        # Handle rewards which could be either scalar or array
        if isinstance(rewards, (list, np.ndarray)):
            reward = rewards[0]
        else:
            reward = rewards
            
        episode_reward += reward
        episode_steps += 1
        
        if done:
            # Measure episode duration
            duration = time.time() - start_time
            print(f"Episode {episode + 1} completed:")
            print(f"  Steps: {episode_steps}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Average FPS: {episode_steps / duration:.2f}")
            break

print("Testing completed!") 