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
from humanoid_env import HumanoidImitationWalkEnv, GymAdapter

# Custom callback to print training metrics at the end of each iteration
class IterationMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, timesteps_per_iteration=10000, start_iteration=0):
        super(IterationMetricsCallback, self).__init__(verbose)
        self.timesteps_per_iteration = timesteps_per_iteration
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_iteration = start_iteration  # Start from a specific iteration number
        self.start_time = time.time()
        self.iteration_start_time = time.time()
        
        # New: For tracking reward components
        self.reward_comp_totals = {}
    
    def _on_step(self):
        # Collect episode stats
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
            
            # New: Collect reward components if available
            if 'reward_components' in info:
                for comp_name, comp_value in info['reward_components'].items():
                    if comp_name not in self.reward_comp_totals:
                        self.reward_comp_totals[comp_name] = []
                    self.reward_comp_totals[comp_name].append(comp_value)
        
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
                
                # New: Print mean reward components if available
                if self.reward_comp_totals:
                    print("  Reward components:")
                    for comp_name, values in self.reward_comp_totals.items():
                        if values:  # Check if we have any values
                            mean_value = np.mean(values)
                            print(f"    {comp_name}: {mean_value:.4f}")
                    # Reset component accumulators for next iteration
                    self.reward_comp_totals = {}
                
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

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--timesteps', type=int, default=10000000, 
                    help='Total number of timesteps to train')
parser.add_argument('--timesteps_per_iteration', type=int, default=10000,
                    help='Number of timesteps per iteration (for logging purposes)')
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
# Add new argument for continuing from a checkpoint
parser.add_argument('--continue_from', type=str, default=None,
                    help='Path to model file to continue training from (e.g., models/train_run_2/ppo_humanoid_final.zip)')
parser.add_argument('--reset_num_timesteps', action='store_false',
                    help='If set, do not reset the number of timesteps (default: reset)')
parser.add_argument('--additional_timesteps', type=int, default=None,
                    help='Number of additional timesteps to train when continuing from a checkpoint')
args = parser.parse_args()

# Use total_timesteps directly
total_timesteps = args.timesteps

# If we're continuing from a checkpoint, extract the step count
start_iteration = 0
start_steps = 0
if args.continue_from:
    # Try to extract the step count from the filename
    checkpoint_path = args.continue_from
    if '_steps_' in checkpoint_path:
        try:
            # Extract the step count from the file name format: ppo_humanoid_steps_XXXXXX_steps.zip
            steps_str = checkpoint_path.split('_steps_')[1].split('_steps.zip')[0]
            start_steps = int(steps_str)
            start_iteration = start_steps // args.timesteps_per_iteration
            print(f"Continuing from checkpoint at step {start_steps} (iteration {start_iteration})")
        except (ValueError, IndexError):
            print("Could not parse step count from checkpoint filename, starting from iteration 0")
    else:
        # Try to parse if it's the final model (which doesn't have steps in the filename)
        # Look for other checkpoint files in the same directory to estimate progress
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.isdir(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('ppo_humanoid_steps_') and f.endswith('_steps.zip')]
            if checkpoint_files:
                # Find the file with the highest step count
                highest_step_file = max(checkpoint_files, key=lambda f: int(f.split('_steps_')[1].split('_steps.zip')[0]))
                try:
                    steps_str = highest_step_file.split('_steps_')[1].split('_steps.zip')[0]
                    start_steps = int(steps_str)
                    start_iteration = start_steps // args.timesteps_per_iteration
                    print(f"Estimated progress from checkpoint directory: step {start_steps} (iteration {start_iteration})")
                except (ValueError, IndexError):
                    print("Could not parse step count from checkpoint files, starting from iteration 0")
            else:
                print("No step information found in checkpoint directory, starting from iteration 0")
        else:
            print("Continuing from checkpoint, starting from iteration 0")

# If additional_timesteps is specified, adjust total_timesteps
if args.additional_timesteps is not None and args.continue_from:
    # Calculate new total including already completed steps
    total_timesteps = start_steps + args.additional_timesteps
    print(f"Training for {args.additional_timesteps} additional timesteps")
    print(f"Total timesteps (including already completed): {total_timesteps}")

# Create a unique run name with timestamp if not provided or reuse from checkpoint
if args.run_name is None:
    if args.continue_from:
        # Try to extract run name from checkpoint path
        checkpoint_dir = os.path.dirname(args.continue_from)
        if os.path.isdir(checkpoint_dir):
            # Use the directory name as the run name
            args.run_name = os.path.basename(checkpoint_dir)
            print(f"Reusing run name from checkpoint: {args.run_name}")
        
    # If still no run name, create a new one with timestamp
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
    
    if args.additional_timesteps is not None and args.continue_from:
        f.write(f"Additional timesteps: {args.additional_timesteps}\n")
        f.write(f"Timesteps per iteration (for logging): {args.timesteps_per_iteration}\n")
        f.write(f"Previously completed steps: {start_steps}\n")
        f.write(f"Previously completed iterations: {start_iteration}\n")
        f.write(f"Total timesteps (including already completed): {total_timesteps}\n")
    else:
        f.write(f"Total timesteps: {total_timesteps}\n")
        f.write(f"Timesteps per iteration (for logging): {args.timesteps_per_iteration}\n")
    
    f.write(f"Motion file: {args.motion_file}\n")
    f.write(f"Environments: {args.n_envs}\n")
    f.write(f"Eval frequency: {args.eval_freq}\n")
    f.write(f"Save frequency: {args.save_freq}\n")
    if args.continue_from:
        f.write(f"Continuing from checkpoint: {args.continue_from}\n")
        f.write(f"Starting from iteration: {start_iteration}\n")

# Create environment function for vectorized env
def make_env(render=False, rank=0, seed=0):
    def _init():
        # Create the environment and wrap it with our adapter
        env = HumanoidImitationWalkEnv(renders=render, motion_file=args.motion_file)
        env = GymAdapter(env)  # Add the adapter wrapper
        # Wrap environment with monitor
        env = Monitor(env, os.path.join(run_log_dir, f"train_{rank}"), 
                      allow_early_resets=True)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Create evaluation environment with our wrapper
eval_env_raw = HumanoidImitationWalkEnv(renders=args.render, motion_file=args.motion_file)
eval_env = GymAdapter(eval_env_raw)  # Add the adapter wrapper
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

# Create the iteration metrics callback with the appropriate starting iteration
iteration_metrics_callback = IterationMetricsCallback(
    verbose=1, 
    timesteps_per_iteration=args.timesteps_per_iteration,
    start_iteration=start_iteration
)

# Combine all callbacks
callbacks = [
    eval_callback, 
    checkpoint_callback, 
    timestep_checkpoint_callback, 
    iteration_metrics_callback
]

# Load or create the agent
if args.continue_from:
    print(f"Loading model from checkpoint: {args.continue_from}")
    model = PPO.load(
        args.continue_from,
        env=env,
        tensorboard_log=run_log_dir,
        verbose=1
    )
    
    # Search for a vec_normalize.pkl file in the same directory as the checkpoint
    checkpoint_dir = os.path.dirname(args.continue_from)
    vec_normalize_path = os.path.join(checkpoint_dir, "vec_normalize.pkl")
    
    if os.path.exists(vec_normalize_path):
        print(f"Loading normalization stats from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        # Don't update the normalization statistics during testing
        env.training = True
        
        # Also load normalization for eval env
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False
    else:
        print("Warning: No vec_normalize.pkl file found. Using new normalization stats.")
else:
    # Create the PPO model with optimized hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,        # Lowered learning rate for potentially better tuning
        n_steps=2048,              # Increase for stability
        batch_size=512,            # Increase for stability
        gamma=0.99,                # Future reward discount factor
        ent_coef=0.005,            # Standard entropy coefficient
        clip_range=0.2,            # Standard clip range
        vf_coef=0.5,
        max_grad_norm=0.5,         # Keep original
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 512], vf=[512, 512])], # Increased network size
            activation_fn=torch.nn.ReLU
        ),
        tensorboard_log=run_log_dir, # This will point to the new run's log dir
        verbose=1
    )
    
    # Log hyperparameters to training config
    with open(config_log, 'a') as f:
        f.write("\nPPO Hyperparameters (Set 3 - Larger Net, Pose Focus Rewards):\n") # Note the change
        f.write(f"  learning_rate: {1e-4}\n") # Keep lower LR
        f.write(f"  n_steps: {2048}\n")
        f.write(f"  batch_size: {512}\n")
        f.write(f"  gamma: {0.99}\n")
        f.write(f"  ent_coef: {0.005}\n") # Keep standard entropy
        f.write(f"  clip_range: {0.2}\n")
        f.write(f"  max_grad_norm: {0.5}\n")
        f.write(f"  net_arch: [512, 512]\n") # Log new architecture

# Train the model
print(f"Starting training run: {args.run_name}")
print(f"Logs will be saved to: {run_log_dir}")
print(f"Models will be saved to: {run_model_dir}")

if args.additional_timesteps is not None and args.continue_from:
    print(f"Training for {args.additional_timesteps} additional timesteps")
    print(f"Total timesteps (including already completed): {total_timesteps}")
else:
    print(f"Training for a total of {total_timesteps} timesteps")

model.learn(
    total_timesteps=total_timesteps,
    callback=callbacks,
    reset_num_timesteps=args.reset_num_timesteps
)

# Save the final model
final_model_path = os.path.join(run_model_dir, "ppo_humanoid_final.zip")
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")

# Save the normalization stats too
env.save(os.path.join(run_model_dir, "vec_normalize.pkl"))
print(f"Normalization statistics saved to {os.path.join(run_model_dir, 'vec_normalize.pkl')}")
print("Training complete!") 