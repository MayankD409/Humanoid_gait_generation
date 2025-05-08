import os
import torch
import numpy as np
import argparse
from pathlib import Path
from functools import partial
import pickle

# For visualization
import time

def import_env(env_name_str):
    if env_name_str=='jvrc_walk':
        from envs.jvrc import JvrcWalkEnv as Env
    elif env_name_str=='jvrc_step':
        from envs.jvrc import JvrcStepEnv as Env
    else:
        raise Exception("Check env name!")
    return Env

def evaluate_policy(policy, env, num_episodes=5, render=True):
    """Evaluate a trained policy on the environment"""
    policy.eval()  # Set policy to evaluation mode
    
    episode_rewards = []
    
    for i in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float)
        done = False
        episode_reward = 0
        step = 0
        
        # Initialize hidden state if recurrent policy
        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()
        
        print(f"Starting episode {i+1}/{num_episodes}")
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                action = policy(state, deterministic=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action.numpy())
            
            episode_reward += reward
            step += 1
            
            if render:
                env.render()
                time.sleep(0.01)  # Slow down rendering
            
            # Print progress
            if step % 50 == 0:
                print(f"Step {step}, Current reward: {episode_reward:.2f}")
            
            # Update state
            state = torch.tensor(next_state, dtype=torch.float)
        
        episode_rewards.append(episode_reward)
        print(f"Episode {i+1} finished with total reward: {episode_reward:.2f} in {step} steps")
    
    # Print summary
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nEvaluation Results:")
    print(f"Mean episode reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained imitation learning policy')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the actor.pt policy file')
    parser.add_argument('--num_episodes', type=int, default=5,
                      help='Number of episodes to evaluate')
    parser.add_argument('--no_render', action='store_true',
                      help='Disable rendering')
    args = parser.parse_args()
    
    # Load policy file
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    policy = torch.load(model_path, weights_only=False)
    print(f"Loaded policy from {model_path}")
    
    # Load experiment args (to get environment info)
    pkl_path = Path(model_path.parent, "experiment.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Experiment data file not found: {pkl_path}")
    
    run_args = pickle.load(open(pkl_path, "rb"))
    
    # Create environment
    Env = import_env(run_args.env)
    if hasattr(run_args, 'yaml') and run_args.yaml is not None:
        yaml_path = Path(run_args.yaml)
        env = Env(path_to_yaml=yaml_path)
    else:
        env = Env()
    
    # Evaluate policy
    print(f"Evaluating policy for {args.num_episodes} episodes...")
    evaluate_policy(
        policy=policy,
        env=env,
        num_episodes=args.num_episodes,
        render=not args.no_render
    )

if __name__ == "__main__":
    main() 