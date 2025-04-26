import os
import gym
import time
import numpy as np
import argparse
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

# Wrapper to ignore episode termination (keep going even if the episode would normally end)
class NoTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:  # New gym API
            obs, reward, terminated, truncated, info = result
            # Ignore termination but record it in info for debugging
            info['would_terminate'] = terminated
            info['would_truncate'] = truncated
            return obs, reward, False, False, info
        elif len(result) == 4:  # SB3 compatible API
            obs, reward, done, info = result
            # Ignore termination but record it in info for debugging
            info['would_terminate'] = done
            return obs, reward, False, info
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
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--steps_per_episode', type=int, default=300,
                        help='Fixed number of steps to run per episode')
    parser.add_argument('--ignore_termination', action='store_true', default=True,
                        help='Continue the episode even after normal termination conditions')
    parser.add_argument('--debug', action='store_true',
                        help='Enable additional debug output')
    args = parser.parse_args()
    
    # Paths to model and normalization statistics
    model_path = os.path.join(args.model_dir, f"{args.model_name}.zip")
    vec_normalize_path = os.path.join(args.model_dir, "vec_normalize.pkl")
    
    print(f"Loading model from {model_path}")
    
    # Create the environment
    env = HumanoidImitationEnv(renders=args.render, motion_file=args.motion_file,
                              rescale_actions=True, rescale_observations=True)
    
    # Enable debug mode in the environment if it supports it
    if hasattr(env, 'debug'):
        env.debug = args.debug
    
    env = GymAdapter(env)  # Add the wrapper for gym-SB3 compatibility
    env = InfoWrapper(env)  # Add our info wrapper
    
    # Add the no-termination wrapper if requested
    if args.ignore_termination:
        env = NoTerminationWrapper(env)
        print("Ignoring normal termination conditions - episodes will run for fixed steps")
    
    # Wrap the environment in a DummyVecEnv for compatibility with stable-baselines
    env = DummyVecEnv([lambda: env])
    
    # Load the saved VecNormalize statistics
    env = VecNormalize.load(vec_normalize_path, env)
    
    # Don't update the normalization statistics at test time
    env.training = False
    # Don't normalize the rewards at test time
    env.norm_reward = False
    
    # Load the saved model
    model = PPO.load(model_path)
    
    # Run the model
    for episode in range(args.num_episodes):
        obs = env.reset()
        
        if args.debug:
            print(f"\nObservation shape: {obs.shape}")
            print(f"First few values: {obs[0][:5]}")
        
        episode_reward = 0
        cumulative_rewards = []
        
        print(f"\nStarting episode {episode+1}/{args.num_episodes}")
        
        # Run for a fixed number of steps
        for step in range(args.steps_per_episode):
            # Get the model's action
            action, _ = model.predict(obs, deterministic=True)
            
            if args.debug and step == 0:
                print(f"First action: {action[0][:5]}")
            
            # Apply the action
            obs, reward, done, info = env.step(action)
            
            # Extract info
            step_info = info[0] if isinstance(info, list) else info
            
            # Track would_terminate signals from wrapper
            would_terminate = step_info.get('would_terminate', False)
            
            # Accumulate reward
            episode_reward += reward[0]  # Extract scalar from array
            cumulative_rewards.append(episode_reward)
            
            # Print info periodically
            if step % 10 == 0:
                print(f"Step {step}, Current reward: {episode_reward:.2f}")
            
            # Debug output when termination would have happened
            if would_terminate and args.debug:
                print(f"At step {step}: Model would normally terminate. Continuing anyway.")
            
            if args.render:
                # The rendering happens inside the environment
                time.sleep(1/60)  # Cap at 60 FPS
                
        # Episode completed after fixed steps
        print(f"Episode {episode+1} completed after {args.steps_per_episode} steps")
        print(f"Final cumulative reward: {episode_reward:.2f}")
        
        # Print reward statistics
        if len(cumulative_rewards) > 0:
            print("\nReward statistics:")
            print(f"  Average step reward: {episode_reward / args.steps_per_episode:.4f}")
            
            # Calculate reward rate (reward per step)
            step_rewards = np.diff([0] + cumulative_rewards)
            print(f"  Min step reward: {min(step_rewards):.4f}")
            print(f"  Max step reward: {max(step_rewards):.4f}")
            print(f"  Median step reward: {np.median(step_rewards):.4f}")
    
    # Close the environment
    env.close()
    print("\nTest completed")

if __name__ == "__main__":
    main() 