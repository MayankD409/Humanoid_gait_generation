# Humanoid Imitation Learning

This repository implements imitation learning for a humanoid character using motion capture data and PyBullet physics simulation.

## Setup

```bash
# Clone the repository
git clone https://github.com/your-username/humanoid_imitation.git
cd humanoid_imitation

# Install dependencies
pip install -r requirements.txt
```

## Running the Pre-trained Model

### 1. Download the pre-trained model

Download the `model` folder from [Google Drive](INSERT_GOOGLE_DRIVE_LINK_HERE) and place it in the root directory.


### 2. Run the model

```bash
python test_model.py --model_dir model --model_name best_model --render
```

Add `--slowmo 1.5` to slow down the animation for better visualization:

```bash
python test_model.py --model_dir model --model_name best_model --render --slowmo 1.5
```

To record a video of the simulation:

```bash
python test_model.py --model_dir model --model_name best_model --render --record --video_name enhanced_walk_demo
```

## Training Your Own Model

```bash
./run_improved_training.sh
```

### Monitor Training Progress

```bash
# View training metrics for the latest run
tensorboard --logdir=logs/$(ls -t logs | head -1)
```

Access http://localhost:6006 in your browser to view training metrics.

## Behaviour Cloning

This repository includes a behavior cloning implementation to learn from expert demonstrations using a neural network policy.

### Prerequisites

Ensure you've installed all the required dependencies:

```bash
pip install -r requirements.txt
```

### Running Behavior Cloning

#### Training Mode

To train a policy using behavior cloning:

```bash
python3 behaviour_cloning/run_imitation_learning.py train \
    --env jvrc_walk \
    --expert-model <path_to_expert_model>/actor.pt \
    --logdir <output_directory> \
    --n-itr 500 \
    --num-procs 12
```

Additional training options:
- `--env`: Environment type (`jvrc_walk` or `jvrc_step`)
- `--expert-model`: Path to the expert policy model to imitate
- `--logdir`: Directory to save weights and logs
- `--n-itr`: Number of iterations for training (default: 500)
- `--num-procs`: Number of parallel processes for training (default: 12)
- `--recurrent`: Use LSTM network instead of feedforward
- `--yaml`: Path to config file for the environment

#### Evaluation Mode

To evaluate a trained behavior cloning policy:

```bash
python3 behaviour_cloning/run_imitation_learning.py eval \
    --path <path_to_trained_model>/actor.pt \
    --out-dir model_evaluation \
    --ep-len 10
```

Evaluation options:
- `--path`: Path to the trained model (either directory containing actor.pt or direct path)
- `--out-dir`: Directory to save evaluation videos
- `--ep-len`: Episode length in seconds (default: 10)

### Example

Run the provided pre-trained model:

```bash
python3 behaviour_cloning/run_imitation_learning.py eval --path model/actor.pt --out-dir model_evaluation
```

### Alternative Evaluation Script

The repository also includes an alternative evaluation script that runs multiple episodes and reports average performance:

```bash
python3 behaviour_cloning/evaluate_imitation.py --model_path model/actor.pt
```

Options:
- `--model_path`: Path to the trained policy model file (required)
- `--num_episodes`: Number of episodes to evaluate (default: 5)
- `--no_render`: Disable rendering to run without visualization

Example with no rendering (useful for headless servers or performance testing):
```bash
python3 behaviour_cloning/evaluate_imitation.py --model_path model/actor.pt --no_render
```

References:
1. https://github.com/rohanpsingh/LearningHumanoidWalking
2. https://github.com/xbpeng/DeepMimic.git