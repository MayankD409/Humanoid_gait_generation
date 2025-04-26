import os
import gym
import time
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
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

# Force the environment to continue regardless of done state
class ForceRunWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:  # New gym API
            obs, reward, terminated, truncated, info = result
            # Force environment to continue
            if terminated or truncated:
                print("Episode would normally end here with reason: {}".format(
                    info.get('episode_end_reason', 'unknown')))
                print("Forcing environment to continue...")
                # Reset the environment but keep running
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    new_obs = reset_result[0]
                else:
                    new_obs = reset_result
                return new_obs, reward, False, False, info
            return obs, reward, terminated, truncated, info
        elif len(result) == 4:  # SB3 compatible API
            obs, reward, done, info = result
            # Force environment to continue
            if done:
                print("Episode would normally end here with reason: {}".format(
                    info.get('episode_end_reason', 'unknown')))
                print("Forcing environment to continue...")
                # Reset the environment but keep running
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    new_obs = reset_result[0]
                else:
                    new_obs = reset_result
                return new_obs, reward, False, info
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
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Maximum steps per episode (to prevent infinite loops)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable additional debug output')
    parser.add_argument('--force_run', action='store_true',
                        help='Force the environment to continue even if done')
    parser.add_argument('--slowmo', type=float, default=1.0,
                        help='Slow motion factor (1.0 = normal speed, 2.0 = half speed, etc.)')
    args = parser.parse_args()
    
    # Paths to model
    model_path = os.path.join(args.model_dir, "{0}.zip".format(args.model_name))
    
    print("Loading model from {0}".format(model_path))
    
    # Create the environment
    env = HumanoidImitationEnv(renders=args.render, motion_file=args.motion_file,
                              rescale_actions=True, rescale_observations=True)
    
    # Enable debug mode in the environment if it supports it
    if hasattr(env, 'debug'):
        env.debug = args.debug
    
    # Apply wrappers
    env = GymAdapter(env)  # Add the adapter wrapper
    env = InfoWrapper(env)  # Add our info wrapper
    
    # Optionally apply the force run wrapper
    if args.force_run:
        env = ForceRunWrapper(env)
        print("Force run mode enabled - will continue even if done state is reached")
    
    # Wrap the environment in a DummyVecEnv for compatibility with stable-baselines
    env = DummyVecEnv([lambda: env])
    
    # Note: We skip loading VecNormalize statistics here
    # The model should have learned to handle the normalized observations already
    
    # Load the saved model
    model = PPO.load(model_path, print_system_info=True)
    
    print("\nModel and environment loaded successfully")
    print("Observation space: {}".format(env.observation_space))
    print("Action space: {}".format(env.action_space))
    
    # Display slowmo info if enabled
    if args.slowmo > 1.0:
        print(f"Slow motion enabled: {args.slowmo}x slower than normal")
    
    # Calculate frame delay based on slowmo factor (60 FPS at normal speed)
    frame_delay = (1/60) * args.slowmo
    
    # Run the model
    total_rewards = []
    step_counts = []
    
    for episode in range(args.num_episodes):
        obs = env.reset()
        
        if args.debug:
            print("\nObservation shape: {0}".format(obs.shape))
            print("First few values: {0}".format(obs[0][:5]))
        
        done = False
        episode_reward = 0
        step_count = 0
        
        print("\nStarting episode {0}/{1}".format(episode+1, args.num_episodes))
        
        # Store info from each step for debugging
        step_infos = []
        
        while not done and step_count < args.max_steps:
            # Get the model's action
            action, _ = model.predict(obs, deterministic=True)
            
            if args.debug and step_count == 0:
                print("First action: {0}".format(action[0][:5]))
            
            # Apply the action
            obs, reward, done, info = env.step(action)
            
            # Store step info for debugging
            step_infos.append(info[0] if isinstance(info, list) else info)
            
            # Accumulate reward
            episode_reward += reward[0]  # Extract scalar from array
            step_count += 1
            
            if step_count % 100 == 0 or step_count == 1:
                print("Step {0}, Current reward: {1:.2f}".format(step_count, episode_reward))
            
            if args.render:
                # Sleep to slow down the visualization if slowmo is enabled
                time.sleep(frame_delay)
                
            if done:
                # Print detailed termination info
                done_info = info[0] if isinstance(info, list) else info
                reason = done_info.get('episode_end_reason', 'unknown')
                print("Episode ended after {0} steps. Reason: {1}".format(step_count, reason))
                print("Final info: {0}".format(done_info))
                break
                
        total_rewards.append(episode_reward)
        step_counts.append(step_count)
        
        print("Episode {0} finished after {1} steps with reward {2:.2f}".format(episode+1, step_count, episode_reward))
        
        # Print last few step infos if episode was short
        if step_count < 10 and args.debug:
            print("Step infos for short episode:")
            for i, info in enumerate(step_infos):
                print("Step {0} info: {1}".format(i+1, info))
    
    # Calculate statistics
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_steps = np.mean(step_counts)
    std_steps = np.std(step_counts)
    
    print("\n===== Test Results =====")
    print("Number of episodes: {0}".format(args.num_episodes))
    print("Mean reward: {0:.2f} +/- {1:.2f}".format(mean_reward, std_reward))
    print("Mean steps: {0:.2f} +/- {1:.2f}".format(mean_steps, std_steps))
    print("========================")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main() 