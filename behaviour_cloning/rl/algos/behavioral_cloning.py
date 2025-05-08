"""Behavioral Cloning Implementation for Imitation Learning."""

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import numpy as np
import time

import ray
from ray.exceptions import RayError, RayTaskError, GetTimeoutError

from rl.policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor
from rl.envs.normalize import get_normalization_params

class BehavioralCloning:
    def __init__(self, env_fn, args):
        self.lr             = args.lr
        self.eps            = args.eps
        self.minibatch_size = args.minibatch_size
        self.epochs         = args.epochs
        self.max_traj_len   = args.max_traj_len
        self.n_proc         = args.num_procs
        self.grad_clip      = args.max_grad_norm
        self.eval_freq      = args.eval_freq
        self.recurrent      = args.recurrent
        self.imitate_coeff  = args.imitate_coeff if hasattr(args, 'imitate_coeff') else 1.0

        # batch_size depends on number of parallel envs
        self.batch_size = self.n_proc * self.max_traj_len

        self.total_steps = 0
        self.highest_reward = -np.inf

        # counter for training iterations
        self.iteration_count = 0

        # directory logging and saving weights
        self.save_path = Path(args.logdir)
        Path.mkdir(self.save_path, parents=True, exist_ok=True)

        # create the summarywriter
        self.writer = SummaryWriter(log_dir=self.save_path, flush_secs=10)

        # create networks or load up pretrained
        obs_dim = env_fn().observation_space.shape[0]
        action_dim = env_fn().action_space.shape[0]
        
        if args.recurrent:
            policy = Gaussian_LSTM_Actor(obs_dim, action_dim, init_std=args.std_dev,
                                         learn_std=args.learn_std)
        else:
            policy = Gaussian_FF_Actor(obs_dim, action_dim, init_std=args.std_dev,
                                       learn_std=args.learn_std, bounded=False)

        if hasattr(env_fn(), 'obs_mean') and hasattr(env_fn(), 'obs_std'):
            obs_mean, obs_std = env_fn().obs_mean, env_fn().obs_std
        else:
            obs_mean, obs_std = get_normalization_params(iter=args.input_norm_steps,
                                                         noise_std=1,
                                                         policy=policy,
                                                         env_fn=env_fn,
                                                         procs=args.num_procs)
        with torch.no_grad():
            policy.obs_mean, policy.obs_std = map(torch.Tensor, (obs_mean, obs_std))

        # Load expert policy
        if not args.expert_model:
            raise ValueError("Expert model path is required for behavioral cloning")
            
        expert_policy = torch.load(args.expert_model, weights_only=False)
        
        self.policy = policy
        self.expert_policy = expert_policy
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)

    @staticmethod
    def save(nets, save_path, suffix=""):
        filetype = ".pt"
        for name, net in nets.items():
            path = Path(save_path, name + suffix + filetype)
            torch.save(net, path)
            print("Saved {} at {}".format(name, path))
        return

    @ray.remote
    @torch.no_grad()
    @staticmethod
    def collect_demonstrations(env_fn, expert_policy, max_steps, max_traj_len):
        """
        Collect demonstration data from the expert policy.
        """
        env = env_fn()
        
        states = []
        actions = []
        rewards = []
        total_collected = 0
        
        while total_collected < max_steps:
            state = torch.tensor(env.reset(), dtype=torch.float)
            done = False
            traj_len = 0
            
            if hasattr(expert_policy, 'init_hidden_state'):
                expert_policy.init_hidden_state()
                
            traj_states = []
            traj_actions = []
            traj_rewards = []
            
            while not done and traj_len < max_traj_len:
                action = expert_policy(state, deterministic=True)
                next_state, reward, done, _ = env.step(action.numpy().copy())
                
                traj_states.append(state)
                traj_actions.append(action)
                traj_rewards.append(reward)
                
                state = torch.tensor(next_state, dtype=torch.float)
                traj_len += 1
                total_collected += 1
                
                if total_collected >= max_steps:
                    break
            
            states.extend(traj_states)
            actions.extend(traj_actions)
            rewards.extend(traj_rewards)
            
        return {
            'states': torch.stack(states),
            'actions': torch.stack(actions),
            'rewards': torch.tensor(rewards, dtype=torch.float)
        }

    def collect_demonstrations_parallel(self, env_fn):
        max_steps = self.batch_size
        max_steps_per_proc = max_steps // self.n_proc
        
        # Create pool of workers
        worker = self.collect_demonstrations
        workers = []
        
        try:
            # Start workers with a timeout mechanism
            workers = [worker.remote(env_fn, self.expert_policy, max_steps_per_proc, self.max_traj_len) 
                      for _ in range(self.n_proc)]
            
            # Get results with timeout
            result = ray.get(workers, timeout=60.0)  # 60 second timeout
            
            # Aggregate results
            aggregated_data = {
                'states': torch.cat([r['states'] for r in result]),
                'actions': torch.cat([r['actions'] for r in result]),
                'rewards': torch.cat([r['rewards'] for r in result])
            }
            
            return aggregated_data
            
        except (GetTimeoutError, RayTaskError, RayError) as e:
            print(f"Ray parallel collection error: {e}")
            print("Terminating hanging Ray tasks and restarting Ray...")
            
            # Clean up Ray workers that might be hanging
            for worker_id in workers:
                try:
                    ray.cancel(worker_id, force=True)
                except:
                    pass
            
            # Restart Ray if needed
            try:
                ray.shutdown()
                time.sleep(1)
                ray.init(num_cpus=self.n_proc)
            except:
                pass
                
            # Return empty data with correct shapes to allow training to continue
            dummy_env = env_fn()
            obs_dim = dummy_env.observation_space.shape[0]
            action_dim = dummy_env.action_space.shape[0]
            dummy_env.close()
            
            # Create minimal dummy data (just enough to continue)
            min_size = self.minibatch_size * 2
            return {
                'states': torch.zeros((min_size, obs_dim)),
                'actions': torch.zeros((min_size, action_dim)),
                'rewards': torch.zeros(min_size)
            }

    def update_policy(self, states, expert_actions):
        """Update policy using behavioral cloning loss."""
        total_loss = 0
        
        # Create dataset from demonstrations
        dataset = torch.utils.data.TensorDataset(states, expert_actions)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.minibatch_size, shuffle=True
        )
        
        # Train for multiple epochs over the collected data
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_states, batch_actions in data_loader:
                # Get predicted actions from current policy
                pred_actions = self.policy(batch_states, deterministic=True)
                
                # Compute imitation loss (MSE between predicted and expert actions)
                loss = F.mse_loss(pred_actions, batch_actions)
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(data_loader)
            total_loss += avg_epoch_loss
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.6f}")
            
        return total_loss / self.epochs

    def evaluate(self, env_fn, itr, num_episodes=5):
        """Evaluate current policy."""
        env = env_fn()
        policy = self.policy
        policy.eval()
        
        returns = []
        
        try:
            for episode in range(num_episodes):
                state = torch.tensor(env.reset(), dtype=torch.float)
                done = False
                episode_return = 0
                
                if hasattr(policy, 'init_hidden_state'):
                    policy.init_hidden_state()
                
                # Set a step limit to avoid infinite loops
                step_count = 0
                max_steps = 1000  # Safety limit
                
                while not done and step_count < max_steps:
                    with torch.no_grad():
                        action = policy(state, deterministic=True)
                    next_state, reward, done, _ = env.step(action.numpy())
                    
                    episode_return += reward
                    state = torch.tensor(next_state, dtype=torch.float)
                    step_count += 1
                
                returns.append(episode_return)
                
            mean_return = np.mean(returns)
            
            # Log to TensorBoard
            self.writer.add_scalar('Evaluation/AverageReturn', mean_return, itr)
            print(f"Evaluation at iteration {itr}: Mean return = {mean_return:.2f}")
            
        except Exception as e:
            print(f"Error during policy evaluation: {e}")
            # Return a default value so training can continue
            mean_return = 0.0
            
        finally:
            try:
                env.close()
            except:
                pass
            policy.train()
            
        return mean_return

    def train(self, env_fn, n_itr):
        """Train the agent using behavioral cloning."""
        ray_initialized_here = False
        if not ray.is_initialized():
            ray.init(num_cpus=self.n_proc)
            ray_initialized_here = True
            
        try:    
            for itr in range(n_itr):
                self.iteration_count = itr
                print(f"\nIteration {itr+1}/{n_itr}")
                
                # Collect demonstrations from expert
                print("Collecting demonstrations from expert policy...")
                try:
                    demo_data = self.collect_demonstrations_parallel(env_fn)
                    
                    # Update policy using collected demonstrations
                    print("Updating policy...")
                    avg_loss = self.update_policy(demo_data['states'], demo_data['actions'])
                    
                    # Log training information
                    self.writer.add_scalar('Loss/ImitationLoss', avg_loss, itr)
                    
                    # Save policy periodically
                    nets = {'actor': self.policy}
                    self.save(nets, self.save_path, suffix=f"_{itr}")
                    print(f"Saved actor at {self.save_path}/actor_{itr}.pt")
                    
                    # Evaluate if needed, with error handling
                    if (itr + 1) % self.eval_freq == 0 or itr == 0:
                        try:
                            self.evaluate(env_fn, itr)
                        except Exception as e:
                            print(f"Evaluation error at iteration {itr}: {e}")
                            print("Continuing with training...")
                    
                    # Always save the latest policy
                    self.save(nets, self.save_path)
                    
                except Exception as e:
                    print(f"Error during iteration {itr}: {e}")
                    print("Continuing with next iteration...")
                    
                # Reset Ray if needed after each iteration to prevent resource leaks
                if (itr + 1) % 50 == 0:  # Restart Ray every 50 iterations
                    try:
                        if ray.is_initialized():
                            ray.shutdown()
                            time.sleep(1)
                        ray.init(num_cpus=self.n_proc)
                    except Exception as e:
                        print(f"Error restarting Ray: {e}")
                
            print("Training completed!")
            return self.policy
        finally:
            # Clean up Ray resources
            if ray_initialized_here and ray.is_initialized():
                try:
                    ray.shutdown()
                    print("Ray resources released.")
                except:
                    pass 