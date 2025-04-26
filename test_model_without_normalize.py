import os
import gym
import time
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from humanoid_env import HumanoidImitationWalkEnv, GymAdapter

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
    env = HumanoidImitationWalkEnv(renders=args.render, motion_file=args.motion_file)
    
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
                print(f"Step {step_count}: Reward: {reward[0]:.4f}, Total: {episode_reward:.4f}")
                
                # Print reward components if available
                if 'reward_components' in info[0]:
                    components = info[0]['reward_components']
                    components_str = ", ".join(f"{k}: {v:.4f}" for k, v in components.items())
                    print(f"Reward components: {components_str}")
            
            # Add delay for visualization if render is enabled
            if args.render and args.slowmo > 0:
                time.sleep(frame_delay)
        
        # Print episode stats
        print("\nEpisode {0} completed:".format(episode+1))
        print("  Steps: {0}".format(step_count))
        print("  Total reward: {0:.4f}".format(episode_reward))
        
        # Print termination reason if available
        last_info = step_infos[-1] if step_infos else {}
        if 'episode_end_reason' in last_info:
            print(f"  Termination reason: {last_info['episode_end_reason']}")
        elif 'TimeLimit.truncated' in last_info:
            print("  Termination reason: Time limit reached")
        elif step_count >= args.max_steps:
            print("  Termination reason: Reached maximum steps")
        
        total_rewards.append(episode_reward)
        step_counts.append(step_count)
    
    # Print overall stats
    print("\nTesting complete!")
    print("Average reward: {0:.4f}".format(np.mean(total_rewards)))
    print("Average episode length: {0:.2f}".format(np.mean(step_counts)))
    print("Rewards: {0}".format(total_rewards))
    print("Episode lengths: {0}".format(step_counts))

if __name__ == "__main__":
    main() 