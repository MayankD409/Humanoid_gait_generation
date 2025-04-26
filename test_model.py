import os
import gym
import time
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from humanoid_env import HumanoidImitationWalkEnv, GymAdapter

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

# Custom info-capturing wrapper
class InfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:  # New gym API
            obs, reward, terminated, truncated, info = result
            # Add debug info if possible
            if hasattr(self.env, '_internal_env'):
                if hasattr(self.env._internal_env, 'original_env'):
                    # Try to get more detailed episode termination reason
                    internal_env = self.env._internal_env.original_env
                    info['episode_end_reason'] = getattr(internal_env, 'episode_end_reason', 'unknown')
            return obs, reward, terminated, truncated, info
        elif len(result) == 4:  # SB3 compatible API
            obs, reward, done, info = result
            # Try to add debug info
            if hasattr(self.env, 'env') and hasattr(self.env.env, '_internal_env'):
                if hasattr(self.env.env._internal_env, 'original_env'):
                    # Try to get more detailed episode termination reason
                    internal_env = self.env.env._internal_env.original_env
                    info['episode_end_reason'] = getattr(internal_env, 'episode_end_reason', 'unknown')
            return obs, reward, done, info
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='models', 
                        help='Directory containing the trained model')
    parser.add_argument('--model_name', type=str, default='ppo_humanoid_final', 
                        help='Name of the trained model file (without .zip extension)')
    parser.add_argument('--motion_file', type=str, default='data/Walking.json', 
                        help='Path to motion file to imitate')
    parser.add_argument('--render', action='store_true', 
                        help='Render the environment')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--slowmo', type=float, default=1.0,
                        help='Slow motion factor (1.0 = normal speed, 2.0 = half speed, etc.)')
    args = parser.parse_args()
    
    # Paths to model and normalization stats
    model_path = os.path.join(args.model_dir, "{0}.zip".format(args.model_name))
    stats_path = os.path.join(args.model_dir, "vec_normalize.pkl")
    
    print("Loading model from {0}".format(model_path))
    print("Loading normalization stats from {0}".format(stats_path))
    
    # Create the environment
    env = HumanoidImitationWalkEnv(renders=args.render, motion_file=args.motion_file)
    
    # Enable debug mode if requested
    if hasattr(env, 'debug'):
        env.debug = args.debug
    
    # Apply GymAdapter wrapper for API compatibility
    env = GymAdapter(env)
    
    # Wrap the environment in a DummyVecEnv
    env = DummyVecEnv([lambda: env])
    
    # Load the saved normalization statistics
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        # Don't update the normalization statistics during testing
        env.training = False
        print("Normalization statistics loaded")
    else:
        print("Warning: No normalization statistics found at {0}".format(stats_path))
        env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False)
    
    # Load the saved model
    model = PPO.load(model_path, env=env)
    
    print("Running model...")
    obs = env.reset()
    
    # Calculate frame delay based on slowmo factor (60 FPS at normal speed)
    frame_delay = (1/60) * args.slowmo
    
    # Setup for capturing stats
    episode_rewards = []
    episode_lengths = []
    current_reward = 0
    current_length = 0
    
    # Display slowmo info if enabled
    if args.slowmo > 1.0:
        print(f"Slow motion enabled: {args.slowmo}x slower than normal")
    
    # Run for a fixed number of steps
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        current_reward += rewards[0]
        current_length += 1
        
        # Print reward components if available
        if args.debug and i % 100 == 0:
            print(f"Step {i}, Current reward: {current_reward:.2f}")
            
            if 'reward_components' in info[0]:
                components = info[0]['reward_components']
                components_str = ", ".join(f"{k}: {v:.4f}" for k, v in components.items())
                print(f"Reward components: {components_str}")
        
        # If render is enabled, add a delay to slow down visualization
        if args.render:
            time.sleep(frame_delay)
        
        # If the episode is done, reset and start a new one
        if dones[0]:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            
            print(f"Episode finished after {current_length} steps with reward {current_reward:.4f}")
            print(f"Average episode reward so far: {np.mean(episode_rewards):.4f}")
            
            # Reset episode tracking
            current_reward = 0
            current_length = 0
            
            # Reset environment
            obs = env.reset()
    
    # Print final statistics
    if episode_rewards:
        print("\n===== Final Statistics =====")
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Average episode reward: {np.mean(episode_rewards):.4f}")
        print(f"Average episode length: {np.mean(episode_lengths):.2f}")
        print("============================")

if __name__ == "__main__":
    main() 