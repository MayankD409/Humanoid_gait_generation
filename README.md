# Humanoid Robot Imitation Learning from Human Walking Videos

## Project Overview
This project aims to train a simulated humanoid robot (Atlas model in PyBullet) to autonomously imitate human walking actions by analyzing video data. The robot will learn to replicate human-like walking motions with stable gait patterns based on visual inputs.

## Core Components
1. **Simulation**: PyBullet with Atlas humanoid robot model
2. **Computer Vision**: OpenPose/MediaPipe Pose for human pose extraction
3. **Machine Learning**: Reinforcement Learning (PPO) and Imitation Learning (GAIL, Behavioral Cloning)
4. **Programming Language**: Python

## Directory Structure
- `src/`: Source code for the project
- `data/`: Dataset storage (videos, extracted poses)
- `models/`: Trained models and weights
- `notebooks/`: Jupyter notebooks for experimentation and visualization
- `config/`: Configuration files

## Setup
```bash
# Clone the repository
git clone [repository-url]
cd humanoid_imitation

# Set up virtual environment
python -m venv venv

# Activate virtual environment
## On Windows
venv\Scripts\activate
## On Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
