import os
import gym
import time
import numpy as np
from humanoid_env import HumanoidImitationEnv

def main():
    # Create the environment
    print("Creating HumanoidImitationEnv environment...")
    env = HumanoidImitationEnv(renders=True, motion_file='data/Walking.json', rescale_observations=False)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset the environment
    print("Resetting environment...")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Take a few random actions
    print("Taking 100 random actions...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"Step {i+1}: Reward = {reward:.2f}, Done = {done}")
        
        # Render the environment
        env.render()
        time.sleep(1/30)  # 30 FPS
        
        if done:
            print("Episode finished!")
            obs = env.reset()
    
    # Close the environment
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main() 