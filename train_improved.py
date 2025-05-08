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

# Custom callback to track reward components and implement curriculum learning
class EnhancedTrainingCallback(BaseCallback):
    def __init__(self, verbose=0, timesteps_per_iteration=10000, start_iteration=0,
                curriculum_stages=None, env=None, vec_env=None):
        super(EnhancedTrainingCallback, self).__init__(verbose)
        self.timesteps_per_iteration = timesteps_per_iteration
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_iteration = start_iteration
        self.start_time = time.time()
        self.iteration_start_time = time.time()
        
        # For tracking reward components
        self.reward_comp_totals = {}
        
        # Curriculum learning
        self.curriculum_stages = curriculum_stages
        self.current_stage = 0
        self.env = env
        self.vec_env = vec_env
        
        # Track the progress metrics
        self.progress_metrics = []
        self.last_progress_update = 0
        
        # For adaptive learning rate
        self.performance_window = []
        self.lr_update_freq = 10  # Update learning rate every N iterations
        self.initial_lr = None
    
    def _on_training_start(self):
        # Store initial learning rate
        self.initial_lr = float(self.model.learning_rate)
        print(f"Initial learning rate: {self.initial_lr}")
        
        # Apply initial curriculum stage if any
        self._update_curriculum_stage(0)
    
    def _on_step(self):
        # Collect episode stats
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                # Debug: Print normalized and unnormalized rewards
                if 'episode_real_reward' in info:
                    print(f"Episode finished: Normalized reward={info['episode']['r']:.4f}, "
                          f"Unnormalized reward={info['episode_real_reward']:.4f}, "
                          f"Length={info['episode']['l']}")
            
            # Collect reward components
            if 'reward_components' in info:
                for comp_name, comp_value in info['reward_components'].items():
                    if comp_name not in self.reward_comp_totals:
                        self.reward_comp_totals[comp_name] = []
                    self.reward_comp_totals[comp_name].append(comp_value)
        
        # Update curriculum based on timesteps
        if self.curriculum_stages and self.current_stage < len(self.curriculum_stages) - 1:
            next_stage = self.current_stage + 1
            if self.n_calls >= self.curriculum_stages[next_stage]['start_at_step']:
                self._update_curriculum_stage(next_stage)
        
        # Check if we've completed an iteration
        if self.n_calls % self.timesteps_per_iteration == 0:
            self.current_iteration += 1
            iteration_time = time.time() - self.iteration_start_time
            total_time = time.time() - self.start_time
            
            # Calculate metrics for this iteration
            if len(self.episode_rewards) > 0:
                # Get rewards and lengths only for this iteration
                start_idx = max(0, len(self.episode_rewards) - 50)
                recent_rewards = self.episode_rewards[start_idx:]
                recent_lengths = self.episode_lengths[start_idx:]
                
                mean_reward = np.mean(recent_rewards) 
                mean_length = np.mean(recent_lengths)
                
                # Update performance window for adaptive learning rate
                self.performance_window.append(mean_reward)
                if len(self.performance_window) > 5:  # Keep only last 5 iterations
                    self.performance_window.pop(0)
                
                # Maybe update learning rate
                if self.current_iteration % self.lr_update_freq == 0 and len(self.performance_window) >= 3:
                    self._maybe_update_learning_rate()
                
                print(f"\n{'='*80}")
                print(f"Iteration {self.current_iteration} completed:")
                print(f"  Steps: {self.n_calls}")
                print(f"  Episodes in this iteration: {len(recent_rewards)}")
                print(f"  Mean reward: {mean_reward:.2f}")
                print(f"  Mean episode length: {mean_length:.2f}")
                
                # Print mean reward components
                if self.reward_comp_totals:
                    print("  Reward components:")
                    for comp_name, values in self.reward_comp_totals.items():
                        if values:  # Check if we have any values
                            mean_value = np.mean(values)
                            print(f"    {comp_name}: {mean_value:.4f}")
                    # Reset component accumulators for next iteration
                    self.reward_comp_totals = {}
                
                # Print curriculum stage
                if self.curriculum_stages:
                    print(f"  Current curriculum stage: {self.current_stage}")
                
                print(f"  Current learning rate: {self.model.learning_rate}")
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
    
    def _update_curriculum_stage(self, stage_idx):
        """Update parameters based on curriculum stage"""
        if not self.curriculum_stages or stage_idx >= len(self.curriculum_stages):
            return
        
        self.current_stage = stage_idx
        stage = self.curriculum_stages[stage_idx]
        
        print(f"\n{'#'*80}")
        print(f"Applying curriculum stage {stage_idx}:")
        
        # Update environment parameters if needed
        if 'env_params' in stage and self.vec_env:
            for param, value in stage['env_params'].items():
                try:
                    # Use env_method to call setattr on each sub-environment
                    self.vec_env.env_method('setattr', param, value) 
                    print(f"  Attempted to set env.{param} = {value} in all parallel environments via env_method('setattr', ...)")
                except Exception as e:
                    # Catch potential errors if setattr fails or attribute doesn't exist
                    print(f"  Warning: Could not set {param} on vec_env via env_method('setattr', ...): {e}")
        
        # Update PPO parameters if needed
        if 'ppo_params' in stage and self.model:
            for param, value in stage['ppo_params'].items():
                if hasattr(self.model, param):
                    # Special handling for parameters that could be callable
                    if param in ['clip_range', 'clip_range_vf', 'learning_rate']:
                        # Check if the current attribute is callable
                        curr_attr = getattr(self.model, param)
                        if callable(curr_attr):
                            # If it's callable, we need to replace it with a constant function
                            def make_const_func(val):
                                return lambda _: val
                            setattr(self.model, param, make_const_func(value))
                            print(f"  Set model.{param} = {value} (as callable)")
                        else:
                            # Otherwise, just set the value directly
                            setattr(self.model, param, value)
                            print(f"  Set model.{param} = {value}")
                        
                        # Directly update optimizer's learning rate if it's the learning_rate parameter
                        if param == 'learning_rate' and hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                            for param_group in self.model.policy.optimizer.param_groups:
                                param_group['lr'] = value
                            print(f"  Also updated optimizer's learning rate to {value}")
                    else:
                        # For other parameters, just set directly
                        setattr(self.model, param, value)
                        print(f"  Set model.{param} = {value}")
        
        print(f"{'#'*80}\n")
    
    def _maybe_update_learning_rate(self):
        """Adaptively adjust learning rate based on performance"""
        if len(self.performance_window) < 3:
            return
        
        # Calculate performance trend
        recent_avg = np.mean(self.performance_window[-2:])
        previous_avg = np.mean(self.performance_window[:-2])
        performance_change = recent_avg - previous_avg
        
        # Current learning rate
        current_lr = float(self.model.learning_rate)
        
        # Decision logic
        if performance_change < -0.5:  # Performance degrading
            # Reduce learning rate if it's not already too low
            if current_lr > self.initial_lr * 0.1:
                new_lr = current_lr * 0.7  # Reduce by 30%
                # Update model's learning rate attribute
                self.model.learning_rate = new_lr
                
                # Also update the optimizer's learning rate directly
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                    for param_group in self.model.policy.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"\nPerformance degrading. Reducing learning rate from {current_lr:.6f} to {new_lr:.6f} (updated in optimizer)")
                else:
                    print(f"\nPerformance degrading. Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
        elif performance_change < 0.1:  # Stagnating
            # Try slightly higher learning rate to escape plateau
            new_lr = current_lr * 1.2  # Increase by 20%
            if new_lr > self.initial_lr * 2:  # Cap at 2x initial
                new_lr = self.initial_lr * 2
                
            # Update model's learning rate attribute
            self.model.learning_rate = new_lr
            
            # Also update the optimizer's learning rate directly
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"\nPerformance stagnating. Increasing learning rate from {current_lr:.6f} to {new_lr:.6f} (updated in optimizer)")
            else:
                print(f"\nPerformance stagnating. Increasing learning rate from {current_lr:.6f} to {new_lr:.6f}")
        # If improving well, keep current learning rate

# Enhanced evaluation callback that syncs normalization stats properly
class SyncedEvalCallback(EvalCallback):
    def __init__(self, eval_env, train_env, *args, **kwargs):
        super(SyncedEvalCallback, self).__init__(eval_env=eval_env, *args, **kwargs)
        self.train_env = train_env
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync with the latest normalization statistics
            if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = self.train_env.obs_rms
                self.eval_env.ret_rms = self.train_env.ret_rms
                print("Synced normalization statistics before evaluation")
        return super()._on_step()

def make_env(render=False, rank=0, seed=0):
    """
    Create environment factory function for parallel environments
    """
    def _init():
        # Create the environment and wrap it with our adapter
        env = HumanoidImitationWalkEnv(renders=render)
        env = GymAdapter(env)
        
        # Apply wrappers as needed
        env = Monitor(env)
        
        # Set random seed
        env.seed(seed + rank)
        return env
    return _init

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=20000000, 
                        help='Total number of timesteps to train')
    parser.add_argument('--timesteps_per_iteration', type=int, default=10000,
                        help='Number of timesteps per iteration (for logging purposes)')
    parser.add_argument('--log_dir', type=str, default='logs', 
                        help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, default='models', 
                        help='Directory to save models')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom name for this training run (default: timestamp)')
    parser.add_argument('--n_envs', type=int, default=8, 
                        help='Number of environments to run in parallel')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='Path to model file to continue training from')
    parser.add_argument('--motion_file', type=str, default='data/Walking.json', 
                        help='Path to motion file to imitate')
    args = parser.parse_args()
    
    # Create a unique run name with timestamp if not provided
    if args.run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"improved_{timestamp}"

    # Create directories with subdirectories for this run
    run_log_dir = os.path.join(args.log_dir, args.run_name)
    run_model_dir = os.path.join(args.model_dir, args.run_name)

    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)
    
    # Define curriculum stages
    curriculum_stages = [
        {
            'name': 'Initial Stage',
            'start_at_step': 0,
            'ppo_params': {
                'learning_rate': 5e-4,
                'clip_range': 0.2,
                'ent_coef': 0.02,  # Increased entropy for more exploration
            },
            'env_params': {}  # No specific env params for initial stage
        },
        {
            'name': 'Early Intermediate Stage',
            'start_at_step': 1000000,  # After 1M steps
            'ppo_params': {
                'learning_rate': 3e-4,
                'clip_range': 0.18,
                'ent_coef': 0.015,
            },
            'env_params': {}
        },
        {
            'name': 'Intermediate Stage',
            'start_at_step': 3000000,  # After 3M steps
            'ppo_params': {
                'learning_rate': 1e-4,
                'clip_range': 0.15,
                'ent_coef': 0.01,
            },
            'env_params': {}
        },
        {
            'name': 'Late Intermediate Stage',
            'start_at_step': 6000000,  # After 6M steps
            'ppo_params': {
                'learning_rate': 7e-5,
                'clip_range': 0.125,
                'ent_coef': 0.008,
            },
            'env_params': {}
        },
        {
            'name': 'Advanced Stage',
            'start_at_step': 9000000,  # After 9M steps
            'ppo_params': {
                'learning_rate': 5e-5,
                'clip_range': 0.1,
                'ent_coef': 0.005,
            },
            'env_params': {}
        },
        {
            'name': 'Final Stage',
            'start_at_step': 15000000,  # After 15M steps
            'ppo_params': {
                'learning_rate': 1e-5,
                'clip_range': 0.05,
                'ent_coef': 0.003,
            },
            'env_params': {}
        }
    ]
    
    # Log the training configuration
    config_log = os.path.join(run_log_dir, "training_config.txt")
    with open(config_log, 'w') as f:
        f.write(f"Training run: {args.run_name}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Timesteps per iteration: {args.timesteps_per_iteration}\n")
        f.write(f"Motion file: {args.motion_file}\n")
        f.write(f"Number of parallel environments: {args.n_envs}\n")
        f.write(f"Continuing from: {args.continue_from if args.continue_from else 'None (fresh start)'}\n\n")
        f.write("Curriculum stages:\n")
        for i, stage in enumerate(curriculum_stages):
            f.write(f"  Stage {i}: {stage['name']}\n")
            f.write(f"    Start at step: {stage['start_at_step']}\n")
            for param, value in stage['ppo_params'].items():
                f.write(f"    {param}: {value}\n")
    
    # Set the environment variable for the motion file
    if args.motion_file:
        os.environ["PYBULLET_DEEP_MIMIC_MOTION_FILE"] = args.motion_file
    
    # Create environments (vectorized)
    env = DummyVecEnv([make_env(rank=i) for i in range(args.n_envs)])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99
    )
    
    # Store the unwrapped environment for callbacks to access
    unwrapped_env = env.envs[0].unwrapped
    
    # Model hyperparameters with improvements
    model_kwargs = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 256,  # Increased batch size from 64 to 256 for more stable gradients
        'n_epochs': 10,    # More optimization epochs
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': None,
        # 'normalize_advantage': True,
        'ent_coef': 0.02,   # Higher entropy coefficient for better exploration
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': 0.03,  # Slightly higher KL divergence to allow more policy updates
        'tensorboard_log': run_log_dir,
        'policy_kwargs': {
            # 'net_arch': [dict(pi=[256, 256, 128], vf=[256, 256, 128])],  # Deeper network with an additional layer
            'net_arch': [dict(pi=[256, 256], vf=[256, 256])], 
            'activation_fn': torch.nn.ReLU
        },
        'verbose': 1
    }
    
    # Continue from existing model or create new one
    if args.continue_from and os.path.exists(args.continue_from):
        print(f"Loading existing model from {args.continue_from}")
        # Create a copy of model_kwargs without 'env' parameter for loading
        load_kwargs = model_kwargs.copy()
        if 'env' in load_kwargs:
            del load_kwargs['env']
        model = PPO.load(args.continue_from, env=env, **load_kwargs)
        
        # Extract current steps from filename if possible
        start_steps = 0
        try:
            # Extract steps from a format like "ppo_humanoid_1000000_steps.zip"
            checkpoint_filename = os.path.basename(args.continue_from)
            if '_steps' in checkpoint_filename:
                steps_str = checkpoint_filename.split('_')[-2]
                start_steps = int(steps_str)
                print(f"Extracted start steps: {start_steps}")
        except Exception as e:
            print(f"Could not extract start steps from filename: {e}")
        
        # Try to load VecNormalize statistics if they exist
        vec_normalize_path = os.path.join(os.path.dirname(args.continue_from), "vec_normalize.pkl")
        if os.path.exists(vec_normalize_path):
            print(f"Loading normalization statistics from {vec_normalize_path}")
            env = VecNormalize.load(vec_normalize_path, env)
            # Ensure normalization is still enabled
            env.training = True
            env.norm_reward = True
            print("Successfully loaded normalization statistics")
        else:
            print("Warning: VecNormalize statistics file not found. Training may be unstable.")
        
        # Calculate the start iteration
        start_iteration = start_steps // args.timesteps_per_iteration
    else:
        print("Creating new model")
        model = PPO(**model_kwargs)
        start_steps = 0
        start_iteration = 0
    
    # Create callbacks
    # Custom checkpoint callback to also save VecNormalize statistics
    class EnhancedCheckpointCallback(CheckpointCallback):
        def __init__(self, save_freq, save_path, name_prefix, save_vec_normalize=True, verbose=0):
            super(EnhancedCheckpointCallback, self).__init__(
                save_freq=save_freq,
                save_path=save_path,
                name_prefix=name_prefix,
                verbose=verbose
            )
            self.save_vec_normalize = save_vec_normalize
            
        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                # Save VecNormalize statistics when saving the model
                if self.save_vec_normalize and isinstance(self.model.get_env(), VecNormalize):
                    self.model.get_env().save(os.path.join(self.save_path, "vec_normalize.pkl"))
                    if self.verbose > 1:
                        print(f"Saved VecNormalize statistics to {os.path.join(self.save_path, 'vec_normalize.pkl')}")
            return super()._on_step()
    
    # Use the enhanced checkpoint callback
    checkpoint_callback = EnhancedCheckpointCallback(
        save_freq=100000 // args.n_envs,
        save_path=run_model_dir,
        name_prefix="ppo_humanoid",
        verbose=1
    )
    
    eval_env = Monitor(GymAdapter(HumanoidImitationWalkEnv(renders=False)))
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99
    )
    
    # Replace standard EvalCallback with our synced version
    eval_callback = SyncedEvalCallback(
        eval_env=eval_env,
        train_env=env,
        n_eval_episodes=10,
        eval_freq=5000 // args.n_envs,
        log_path=run_log_dir,
        best_model_save_path=run_model_dir,
        deterministic=True
    )
    
    # Enhanced training callback with curriculum learning
    enhanced_callback = EnhancedTrainingCallback(
        verbose=1,
        timesteps_per_iteration=args.timesteps_per_iteration,
        start_iteration=start_iteration,
        curriculum_stages=curriculum_stages,
        env=unwrapped_env,
        vec_env=env
    )
    
    # Combine callbacks
    callbacks = [checkpoint_callback, eval_callback, enhanced_callback]
    
    # Ensure environment is reset before training
    env.reset()
    
    # Train the model
    print(f"Starting training for {args.timesteps} steps")
    model.learn(
        total_timesteps=args.timesteps, 
        callback=callbacks,
        tb_log_name=args.run_name,
        reset_num_timesteps=(args.continue_from is None)
    )
    
    # Save the final model
    final_model_path = os.path.join(run_model_dir, "ppo_humanoid_final.zip")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    # Save the normalization parameters
    env_stats_path = os.path.join(run_model_dir, "vec_normalize_final.pkl")
    env.save(env_stats_path)
    print(f"Environment normalization stats saved to {env_stats_path}")

if __name__ == "__main__":
    main() 