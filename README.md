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
- `utils/`: Utility scripts including dependency checker
- `atlas_description/`: Atlas robot URDF and mesh files

## Installation & Setup

### Quick Setup (Linux/macOS)
For a quick setup on Linux or macOS, use the provided setup script:

```bash
# Clone the repository
git clone git@github.com:MayankD409/Humanoid_gait_generation.git
cd Humanoid_gait_generation

# Run the setup script
./setup.sh
```

### Manual Setup

```bash
# Clone the repository
git clone git@github.com:MayankD409/Humanoid_gait_generation.git
cd Humanoid_gait_generation

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

### Checking Dependencies
To verify all dependencies are correctly installed:

```bash
# Activate virtual environment first if not already active
python -m utils.check_dependencies
```

## Running the Simulation

### Basic Atlas Robot Simulation
To run the basic Atlas robot simulation:

```bash
python src/simulation_setup.py
```

This will launch a PyBullet window with the Atlas robot in a stable standing pose with animated elbow and knee joints.