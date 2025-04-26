import os
import gym
import time
import torch
import argparse
import numpy as np
import datetime

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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from humanoid_env import HumanoidImitationEnv

# Custom callback to print training metrics at the end of each iteration
class IterationMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, timesteps_per_iteration=10000):
        super(IterationMetricsCallback, self).__init__(verbose)
        self.timesteps_per_iteration = timesteps_per_iteration
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_iteration = 0
        self.start_time = time.time()
        self.iteration_start_time = time.time()
    
    def _on_step(self):
        # Collect episode stats
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        
        # Check if we've completed an iteration
        if self.n_calls % self.timesteps_per_iteration == 0:
            self.current_iteration += 1
            iteration_time = time.time() - self.iteration_start_time
            total_time = time.time() - self.start_time
            
            # Calculate metrics for this iteration
            if len(self.episode_rewards) > 0:
                # Get rewards and lengths only for this iteration
                start_idx = max(0, len(self.episode_rewards) - 50)  # Consider up to last 50 episodes in this iteration
                recent_rewards = self.episode_rewards[start_idx:]
                recent_lengths = self.episode_lengths[start_idx:]
                
                mean_reward = np.mean(recent_rewards) 
                mean_length = np.mean(recent_lengths)
                
                print(f"\n{'='*80}")
                print(f"Iteration {self.current_iteration} completed:")
                print(f"  Steps: {self.n_calls}")
                print(f"  Episodes in this iteration: {len(recent_rewards)}")
                print(f"  Mean reward: {mean_reward:.2f}")
                print(f"  Mean episode length: {mean_length:.2f}")
                print(f"  Iteration time: {iteration_time:.2f} seconds")
                print(f"  FPS: {self.timesteps_per_iteration / iteration_time:.2f}")
                print(f"  Total time elapsed: {total_time:.2f} seconds")
                print(f"{'='*80}\n")
            else:
                print(f"\n{'='*80}")
                print(f"Iteration {self.current_iteration} completed:")
                print(f"  Steps: {self.n_calls}")
                print(f"  No completed episodes in this iteration")
                print(f"  Iteration time: {iteration_time:.2f} seconds")
                print(f"  Total time elapsed: {total_time:.2f} seconds")
                print(f"{'='*80}\n")
            
            # Reset the iteration timer
            self.iteration_start_time = time.time()
            
        return True

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
parser.add_argument('--iterations', type=int, default=1000, 
                    help='Number of iterations to train (each iteration is 10000 timesteps by default)')
parser.add_argument('--timesteps_per_iteration', type=int, default=10000,
                    help='Number of timesteps per iteration')
parser.add_argument('--log_dir', type=str, default='logs', 
                    help='Directory to save logs')
parser.add_argument('--model_dir', type=str, default='models', 
                    help='Directory to save models')
parser.add_argument('--checkpoint_dir', type=str, default=None, 
                    help='Directory to save checkpoints (defaults to model_dir if not specified)')
parser.add_argument('--motion_file', type=str, default='data/Walking.json', 
                    help='Path to motion file to imitate')
parser.add_argument('--render', action='store_true', 
                    help='Render the environment')
parser.add_argument('--eval_freq', type=int, default=10000, 
                    help='Evaluate the model every n steps')
parser.add_argument('--save_freq', type=int, default=100000, 
                    help='Save the model every n steps')
parser.add_argument('--n_envs', type=int, default=8, 
                    help='Number of environments to run in parallel')
parser.add_argument('--run_name', type=str, default=None,
                    help='Custom name for this training run (default: timestamp)')
args = parser.parse_args()

# Calculate total timesteps from iterations
total_timesteps = args.iterations * args.timesteps_per_iteration

# Create a unique run name with timestamp if not provided
if args.run_name is None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_name = f"fresh_{timestamp}"

# Create directories with subdirectories for this run
run_log_dir = os.path.join(args.log_dir, args.run_name)
run_model_dir = os.path.join(args.model_dir, args.run_name)

os.makedirs(run_log_dir, exist_ok=True)
os.makedirs(run_model_dir, exist_ok=True)

# Set checkpoint directory (use model_dir if checkpoint_dir is not specified)
if args.checkpoint_dir is not None:
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
else:
    checkpoint_dir = run_model_dir

# Log the training configuration for reference
config_log = os.path.join(run_log_dir, "training_config.txt")
with open(config_log, 'w') as f:
    f.write(f"Training run: {args.run_name}\n")
    f.write(f"Timestamp: {datetime.datetime.now()}\n")
    f.write(f"Iterations: {args.iterations}\n")
    f.write(f"Timesteps per iteration: {args.timesteps_per_iteration}\n")
    f.write(f"Total timesteps: {total_timesteps}\n")
    f.write(f"Motion file: {args.motion_file}\n")
    f.write(f"Environments: {args.n_envs}\n")
    f.write(f"Eval frequency: {args.eval_freq}\n")
    f.write(f"Save frequency: {args.save_freq}\n")

# Create environment function for vectorized env
def make_env(render=False, rank=0, seed=0):
    def _init():
        # Create the environment and wrap it with our adapter
        env = HumanoidImitationEnv(renders=render, motion_file=args.motion_file,
                                  rescale_actions=True, rescale_observations=True)
        env = GymAdapter(env)  # Add the adapter wrapper
        # Wrap environment with monitor
        env = Monitor(env, os.path.join(run_log_dir, f"train_{rank}"), 
                      allow_early_resets=True)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Create evaluation environment with our wrapper
eval_env = HumanoidImitationEnv(renders=args.render, motion_file=args.motion_file,
                              rescale_actions=True, rescale_observations=True)
eval_env = GymAdapter(eval_env)  # Add the adapter wrapper
eval_env = Monitor(eval_env, os.path.join(run_log_dir, 'eval'))

# Convert to vector environment for compatibility with VecNormalize
eval_env = DummyVecEnv([lambda: eval_env])
# Apply normalization to eval environment as well, with better normalization values
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True,
                      clip_obs=10.0, clip_reward=10.0)

# Create vectorized environments for training
env_fns = [make_env(render=False, rank=i) for i in range(args.n_envs)]
env = DummyVecEnv(env_fns)

# Apply normalization with better parameters for humanoid control
env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True,
                  gamma=0.99, clip_obs=10.0, clip_reward=10.0)

# Define the evaluation callback
eval_callback = EvalCallback(eval_env, best_model_save_path=run_model_dir,
                           log_path=run_log_dir, eval_freq=args.eval_freq,
                           deterministic=True, render=False)

# Define the checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=checkpoint_dir,
                                       name_prefix='ppo_humanoid')

# Also add a checkpoint callback that saves with timestep info for easier tracking
timestep_checkpoint_callback = CheckpointCallback(
    save_freq=args.save_freq, 
    save_path=checkpoint_dir,
    name_prefix='ppo_humanoid_steps'
)

# Create the iteration metrics callback
iteration_metrics_callback = IterationMetricsCallback(
    verbose=1, 
    timesteps_per_iteration=args.timesteps_per_iteration
)

# Combine all callbacks
callbacks = [
    eval_callback, 
    checkpoint_callback, 
    timestep_checkpoint_callback, 
    iteration_metrics_callback
]

# Create the PPO model with optimized hyperparameters
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=5e-4,        # Increased learning rate
    n_steps=1024,              # Smaller n_steps for more frequent updates
    batch_size=256,            # Larger batch size for more stable gradients
    gamma=0.99,                # Future reward discount factor
    ent_coef=0.01,             # Increased entropy for more exploration
    clip_range=0.3,            # Wider clip range 
    vf_coef=0.5,
    max_grad_norm=1.0,
    policy_kwargs=dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],  # Deeper network
        activation_fn=torch.nn.ReLU
    ),
    tensorboard_log=run_log_dir,
    verbose=1
)

# Train the model
print(f"Starting training run: {args.run_name}")
model.learn(
    total_timesteps=total_timesteps,
    callback=callbacks
)

# Save the final model
final_model_path = os.path.join(run_model_dir, "ppo_humanoid_final.zip")
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")

# Save the normalization stats too
env.save(os.path.join(run_model_dir, "vec_normalize.pkl"))
print(f"Normalization statistics saved to {os.path.join(run_model_dir, 'vec_normalize.pkl')}")
print("Training complete!") 