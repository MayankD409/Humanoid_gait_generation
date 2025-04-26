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
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode (to prevent infinite loops)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable additional debug output')
    parser.add_argument('--slowmo', type=float, default=1.0,
                       help='Slow motion factor (1.0 = normal speed, 2.0 = half speed, etc.)')
    parser.add_argument('--analyze_rewards', action='store_true',
                       help='Print detailed analysis of reward components')
    parser.add_argument('--plot_rewards', action='store_true',
                       help='Plot rewards over time if matplotlib is available')
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
    
    env = GymAdapter(env)  # Add the wrapper
    env = InfoWrapper(env)  # Add our info wrapper
    
    # Wrap the environment in a DummyVecEnv for compatibility with stable-baselines
    env = DummyVecEnv([lambda: env])
    
    # Load the saved VecNormalize statistics
    env = VecNormalize.load(vec_normalize_path, env)
    
    # Don't update the normalization statistics at test time
    env.training = False
    # Don't normalize the rewards at test time
    env.norm_reward = False
    
    # Load the saved model
    model = PPO.load(model_path, print_system_info=True)
    
    # Run the model
    total_rewards = []
    step_counts = []
    
    # Calculate frame delay based on slowmo factor (60 FPS at normal speed)
    frame_delay = (1/60) * args.slowmo

    # Display slowmo info if enabled
    if args.slowmo > 1.0:
        print(f"Slow motion enabled: {args.slowmo}x slower than normal")

    # Prepare for reward plotting if requested
    if args.plot_rewards:
        try:
            import matplotlib.pyplot as plt
            reward_history = []
            reward_components = {
                'imitation_reward': [],
                'survival_reward': [],
                'angular_stability_reward': [],
                'upright_reward': [],
                'smooth_motion_reward': []
            }
            steps_history = []
            step_counter = 0
            
            plt.figure(figsize=(12, 8))
            plt.ion()  # Interactive mode on
        except ImportError:
            print("Warning: matplotlib not available, disabling reward plotting")
            args.plot_rewards = False

    for episode in range(args.num_episodes):
        obs = env.reset()
        
        if args.debug:
            print(f"\nObservation shape: {obs.shape}")
            print(f"First few values: {obs[0][:5]}")
        
        done = False
        episode_reward = 0
        step_count = 0
        
        print(f"\nStarting episode {episode+1}/{args.num_episodes}")
        
        # Store info from each step for debugging
        step_infos = []
        
        while not done and step_count < args.max_steps:
            # Get the model's action
            action, _ = model.predict(obs, deterministic=True)
            
            if args.debug and step_count == 0:
                print(f"First action: {action[0][:5]}")
            
            # Apply the action
            obs, reward, done, info = env.step(action)
            
            # Store step info for debugging
            step_infos.append(info[0] if isinstance(info, list) else info)
            
            # Get info object
            current_info = info[0] if isinstance(info, list) else info
            
            # Extract reward components if available
            if args.analyze_rewards and 'imitation_reward' in current_info:
                if step_count % 50 == 0 or step_count == 1:
                    print(f"\nStep {step_count} Reward Breakdown:")
                    for k, v in current_info.items():
                        if 'reward' in k:
                            print(f"  {k}: {v:.4f}")
            
            # Update plotting data if enabled
            if args.plot_rewards:
                step_counter += 1
                reward_history.append(reward[0])
                steps_history.append(step_counter)
                
                # Collect reward components
                for component in reward_components.keys():
                    if component in current_info:
                        reward_components[component].append(current_info[component])
                    else:
                        reward_components[component].append(0)  # Default if not found
                
                # Update plot every 50 steps
                if step_count % 50 == 0:
                    plt.clf()
                    
                    # Main reward plot
                    plt.subplot(2, 1, 1)
                    plt.plot(steps_history, reward_history, 'b-')
                    plt.title('Total Reward')
                    plt.xlabel('Steps')
                    plt.ylabel('Reward')
                    
                    # Reward components
                    plt.subplot(2, 1, 2)
                    for name, values in reward_components.items():
                        if len(values) == len(steps_history):
                            plt.plot(steps_history, values, label=name)
                    plt.legend()
                    plt.title('Reward Components')
                    plt.xlabel('Steps')
                    plt.ylabel('Value')
                    
                    plt.tight_layout()
                    plt.pause(0.01)
            
            # Accumulate reward
            episode_reward += reward[0]  # Extract scalar from array
            step_count += 1
            
            if step_count % 100 == 0 or step_count == 1:
                print(f"Step {step_count}, Current reward: {episode_reward:.2f}")
            
            if args.render:
                # Sleep to slow down the visualization if slowmo is enabled
                time.sleep(frame_delay)
            
            if done:
                # Print detailed termination info
                done_info = info[0] if isinstance(info, list) else info
                reason = done_info.get('episode_end_reason', 'unknown')
                print(f"Episode ended after {step_count} steps. Reason: {reason}")
                print(f"Final info: {done_info}")
                break
                
        total_rewards.append(episode_reward)
        step_counts.append(step_count)
        
        print(f"Episode {episode+1} finished after {step_count} steps with reward {episode_reward:.2f}")
        
        # Print last few step infos if episode was short
        if step_count < 10 and args.debug:
            print("Step infos for short episode:")
            for i, info in enumerate(step_infos):
                print(f"Step {i+1} info: {info}")
    
    # Calculate statistics
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_steps = np.mean(step_counts)
    std_steps = np.std(step_counts)
    
    print("\n===== Test Results =====")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean steps: {mean_steps:.2f} ± {std_steps:.2f}")
    print("========================")
    
    # Close the environment
    env.close()

    # If plotting was enabled, save the final plot
    if args.plot_rewards:
        plt.savefig(f'reward_plot_{args.model_name}.png')
        print(f"Reward plot saved to reward_plot_{args.model_name}.png")
        plt.close()

if __name__ == "__main__":
    main() 