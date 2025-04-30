# Humanoid Imitation Learning

This project implements imitation learning for a humanoid character using motion capture data and PyBullet physics simulation. The agent learns to mimic walking motion using PPO (Proximal Policy Optimization) reinforcement learning algorithm.

## Setup

### Prerequisites

This project requires the following dependencies:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install numpy>=1.18.0 gym==0.21.0 pybullet>=3.0.9 stable-baselines3>=1.1.0,<2.0.0 tensorboard>=2.5.0
```

You'll also need PyBullet's Deep Mimic environment, which should be part of your existing PyBullet installation.

### Directory Structure

```
humanoid_imitation/
├── data/                     # Contains motion data
│   └── Walking.json          # Motion capture data for walking
├── logs/                     # TensorBoard logs root directory
│   └── run_YYYYMMDD_HHMMSS/  # Individual training run logs (auto-created)
│       ├── PPO_1/            # TensorBoard event files
│       ├── train_0/          # Monitor logs for env 0
│       └── training_config.txt # Training parameters log
├── models/                   # Saved models root directory
│   └── run_YYYYMMDD_HHMMSS/  # Individual training run models (auto-created)
│       ├── best_model.zip    # Best model during training
│       ├── ppo_humanoid_final.zip # Final trained model
│       ├── vec_normalize.pkl # Normalization statistics
│       └── ppo_humanoid_steps_X_steps.zip # Intermediate checkpoints
├── humanoid_env.py           # Environment implementation
├── train.py                  # Training script
├── test_model.py             # Testing script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Running the Training

To train the agent, run:

```bash
python3 train.py --iterations 1000 --n_envs 8 --save_freq 50000 --eval_freq 10000
```

For a quick test to verify everything works:

```bash
python3 train.py --iterations 1 --timesteps_per_iteration 500 --n_envs 2 --save_freq 200 --eval_freq 200
```

Each training run creates a timestamped directory (e.g., `run_20250424_153845`) within both the logs and models directories to keep training runs organized.

### Training Arguments

- `--iterations`: Number of iterations to train (default: 1000)
- `--timesteps_per_iteration`: Number of timesteps per iteration (default: 10000)
- `--log_dir`: Directory for TensorBoard logs (default: "logs")
- `--model_dir`: Directory to save final models (default: "models")
- `--checkpoint_dir`: Directory to save intermediate checkpoints (default: same as model_dir)
- `--motion_file`: Path to motion file (default: "data/Walking.json")
- `--render`: Render the environment during training (off by default)
- `--eval_freq`: Evaluation frequency in timesteps (default: 10000)
- `--save_freq`: Model save frequency in timesteps (default: 100000)
- `--n_envs`: Number of parallel environments (default: 8)
- `--run_name`: Custom name for this training run (default: timestamp-based name)

### Monitoring Training Progress

You can monitor the training progress using TensorBoard:

```bash
# For the most recent training run:
tensorboard --logdir=logs/$(ls -t logs | head -1)

# For all training runs:
tensorboard --logdir=logs
```

Then access http://localhost:6006 in your browser to view training metrics.

The training script also prints metrics to the console at regular intervals, including:
- Number of steps completed
- Number of episodes completed
- Mean reward
- Mean episode length
- Training speed (FPS)

Each training run's configuration is also saved in `logs/run_YYYYMMDD_HHMMSS/training_config.txt` for reference.

## Testing the Trained Agent

To evaluate a fully trained agent from the most recent run:

```bash
# Find latest run directory
latest_run=$(ls -t models | head -1)
python3 test_model.py --model_dir models/$latest_run --model_name ppo_humanoid_final --num_episodes 5
```

To test an intermediate checkpoint during training:

```bash
# Find latest run directory
latest_run=$(ls -t models | head -1)
python3 test_model.py --model_dir models/$latest_run --model_name ppo_humanoid_steps_50000_steps --num_episodes 3 --debug
```

### Testing Arguments

- `--model_dir`: Directory containing the trained model (default: "models")
- `--model_name`: Name of the model file without .zip extension (default: "ppo_humanoid_final")
- `--motion_file`: Path to motion file (default: "data/Walking.json")
- `--render`: Render the environment during testing (off by default)
- `--num_episodes`: Number of episodes to run (default: 5)
- `--max_steps`: Maximum steps per episode (default: 1000)
- `--debug`: Enable additional debug output (off by default)

## Training Workflow

The training process has the following features:

1. **Run-Specific Directories**: Each training run creates its own subdirectories within the logs and models directories, keeping experiments organized.

2. **Training Configuration Logging**: Parameters for each run are saved in a training_config.txt file for reference.

3. **Intermediate Checkpoints**: The agent saves checkpoints at regular intervals (as specified by `--save_freq`), allowing you to resume training or test intermediate models.

4. **Evaluation During Training**: The agent evaluates itself at regular intervals (as specified by `--eval_freq`) to track progress.

5. **TensorBoard Integration**: All training metrics are logged to TensorBoard for visualization.

6. **Console Metrics**: Training progress is reported in the console, showing metrics like mean reward, episode length, and training speed.

## How It Works

1. The environment loads a humanoid character and the Walking.json motion data.
2. During each step, the agent receives the current state of the humanoid.
3. The agent outputs joint actions to control the humanoid.
4. The reward is calculated based on how well the humanoid's motion matches the reference motion.
5. PPO algorithm optimizes the policy to maximize the imitation reward.

## Example Commands

#### Training with Custom Run Name
```bash
python3 train.py --iterations 500 --n_envs 4 --run_name walking_experiment_1
```

#### Training with Custom Timesteps per Iteration
```bash
python3 train.py --iterations 500 --timesteps_per_iteration 20000 --n_envs 4
```

#### Training with Custom Directories
```bash
python3 train.py --iterations 500 --n_envs 4 --log_dir custom_logs --model_dir final_models --run_name experiment_1
```

#### Resume Training from a Checkpoint
To resume training from a checkpoint, you'll need to:
1. Create a new run directory
2. Copy the checkpoint and normalization file into it
3. Start training with the new run name

```bash
# Identify the run and checkpoint you want to continue from
source_run="run_20250424_153845"
checkpoint="ppo_humanoid_steps_500000_steps"

# Create a new run directory with a descriptive name
new_run="continued_from_500k"
mkdir -p models/$new_run

# Copy the checkpoint files
cp models/$source_run/$checkpoint.zip models/$new_run/ppo_humanoid_final.zip
cp models/$source_run/vec_normalize.pkl models/$new_run/

# Start training with the same hyperparameters
python3 train.py --iterations 500 --n_envs 4 --run_name $new_run
```

#### Test an Intermediate Checkpoint from a Specific Run
```bash
python3 test_model.py --model_dir models/run_20250424_153845 --model_name ppo_humanoid_steps_500000_steps --num_episodes 3
```

#### Compare Multiple Training Runs in TensorBoard
```bash
tensorboard --logdir=logs/run_20250424_153845:run1,logs/run_20250424_160012:run2
```

## Credits

This project uses [PyBullet's Deep Mimic framework](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/deep_mimic) and [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for reinforcement learning.

## Running Test Scripts

There are several test scripts available for evaluating trained models:

### Basic Model Testing

To test a trained model:

```bash
python test.py --model_path models/ppo_humanoid_final.zip --vec_normalize_path models/vec_normalize.pkl --motion_file data/Walking.json
```

### Extended Model Testing

For more detailed testing with additional metrics:

```bash
python test_model.py --model_dir models --model_name ppo_humanoid_final --motion_file data/Walking.json
```

### GPU-Supported Testing

For optimal performance on systems with GPU hardware:

```bash
python test_gpu_model.py --model_path models/ppo_humanoid_final.zip --vec_normalize_path models/vec_normalize.pkl --motion_file data/Walking.json
```

The GPU test script includes:
- Automatic GPU detection for both PyTorch and TensorFlow
- Optimized memory usage with GPU memory growth control
- Performance metrics and hardware utilization reporting
- Support for both CUDA and CPU fallback

## Common Arguments

All test scripts support the following common arguments:

- `--model_path`: Path to the trained model file
- `--vec_normalize_path`: Path to saved vector normalization statistics
- `--motion_file`: Path to motion file to imitate (default: data/Walking.json)
- `--render`: Flag to enable environment rendering
- `--deterministic`: Flag to use deterministic actions (default: True)
- `--num_episodes` or `--episodes`: Number of test episodes to run (default: 5)
- `--max_steps`: Maximum steps per episode (default: 1000)
- `--debug`: Flag to enable additional debug information

## Vector Normalization

The test scripts require normalization statistics to correctly process observations. These statistics are stored in a file specified by `--vec_normalize_path`.

Vector normalization is important for stable training and evaluation. It:
1. Normalizes observations to have zero mean and unit variance
2. Optionally normalizes rewards (disabled during testing)
3. Maintains running statistics of observations during training

### Loading Vector Normalization

To correctly load vector normalization for testing:

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Wrap environment in DummyVecEnv
env = DummyVecEnv([lambda: env])

# Load normalization statistics
env = VecNormalize.load("models/vec_normalize.pkl", env)

# Disable training and reward normalization for testing
env.training = False
env.norm_reward = False
```

## Important Notes

- Use the same motion file for testing as was used during training
- Ensure the vector normalization file matches the model you're testing
- GPU acceleration is only available if CUDA-capable hardware is detected
- For best performance with the GPU test script, ensure you have the latest CUDA and cuDNN libraries installed

## Prerequisites

- Docker
- NVIDIA Docker support (for GPU acceleration)

## Quick Start with Docker

### Building the Docker Image

```bash
# Clone the repository
git clone <your-repository-url>
cd humanoid_imitation

# Build the Docker image
docker build -t humanoid-imitation .
```

### Running with Docker

```bash
# Basic run with default parameters
docker run --gpus all -it \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  humanoid-imitation python3 train.py --run_name my_run --n_envs 8
```

### Using Docker Compose

```bash
# Run with the default parameters in docker-compose.yml
docker-compose up

# Run with custom parameters
docker-compose run humanoid-imitation python3 train.py --run_name custom_run --n_envs 8 --timesteps 5000000
```

### Continuing Training from a Checkpoint

```bash
docker run --gpus all -it \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  humanoid-imitation python3 train.py \
  --continue_from models/train_run_2/ppo_humanoid_steps_54400000_steps.zip \
  --run_name continued_run \
  --additional_timesteps 1000000 \
  --n_envs 8
```

## Training Parameters

- `--run_name`: Name for the training run (default: timestamp)
- `--n_envs`: Number of parallel environments (default: 8)
- `--timesteps`: Total number of timesteps to train (default: 10000000)
- `--additional_timesteps`: Number of additional timesteps when continuing training
- `--continue_from`: Path to checkpoint to continue training from
- `--eval_freq`: Evaluate the model every n steps (default: 10000)
- `--save_freq`: Save the model every n steps (default: 100000)
- `--motion_file`: Path to motion file to imitate (default: data/Walking.json)

For more parameters, run:
```bash
docker run --rm humanoid-imitation python3 train.py --help
```

## Project Structure

```
humanoid_imitation/
├── train.py                  # Main training script
├── humanoid_env.py           # Environment implementation
├── data/                     # Motion files and reference data
├── models/                   # Saved model checkpoints
├── logs/                     # Training logs and tensorboard files
├── Dockerfile                # Docker configuration
└── docker-compose.yml        # Docker Compose configuration
```

## Viewing TensorBoard Logs

You can use TensorBoard to visualize training progress:

```bash
# On your host machine (not in Docker)
tensorboard --logdir=./logs
```

Then open your browser at http://localhost:6006/ to view the metrics. 