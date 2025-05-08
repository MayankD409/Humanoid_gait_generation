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

Download the `best_model.zip` file from [Google Drive](INSERT_GOOGLE_DRIVE_LINK_HERE) and place it in the `models/enhanced_walk_2/` directory.

```bash
# Create directory if it doesn't exist
mkdir -p models/enhanced_walk_2

# Download the model (replace with actual download command)
# wget -O models/enhanced_walk_2/best_model.zip YOUR_GOOGLE_DRIVE_LINK
```

### 2. Run the model

```bash
python test_model.py --model_dir models --model_name best_model --render
```

Add `--slowmo 1.5` to slow down the animation for better visualization:

```bash
python test_model.py --model_dir models --model_name best_model --render --slowmo 1.5
```

To record a video of the simulation:

```bash
python test_model.py --model_dir models --model_name best_model --render --record --video_name enhanced_walk_demo
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