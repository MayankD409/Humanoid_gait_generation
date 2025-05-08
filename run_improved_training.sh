#!/bin/bash

# This script runs the improved humanoid training with enhanced rewards, curriculum learning, and exploration

# Stop any existing training processes
stop_existing_training() {
    echo "Checking for existing training processes..."
    EXISTING_PROCS=$(pgrep -f "python3.*train_improved.py")
    if [ -n "$EXISTING_PROCS" ]; then
        echo "Found existing training processes, stopping: $EXISTING_PROCS"
        kill $EXISTING_PROCS
        sleep 2
        # Check if any processes are still running
        REMAINING=$(pgrep -f "python3.*train_improved.py")
        if [ -n "$REMAINING" ]; then
            echo "Some processes still running, forcing stop: $REMAINING"
            kill -9 $REMAINING
            sleep 1
        fi
        echo "All existing training processes stopped."
    else
        echo "No existing training processes found."
    fi
}

# Stop existing training before starting a new one
stop_existing_training

# Create a run name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="enhanced_walk_20250501_102201"
# "enhanced_walk_${TIMESTAMP}"

# You can continue from a previous model if needed
# CONTINUE_FROM="models/walk_reward_v2/best_model.zip"
CONTINUE_FROM="model/enhanced_walk_20250501_102201/ppo_humanoid_19600000_steps.zip"
# "models/train_run_2/ppo_humanoid_steps_27200000_steps.zip"

# Set the number of environments based on your CPU cores
NUM_ENVS=8

# Create the logs directory if it doesn't exist
mkdir -p logs

# Create a log file for the output
LOG_FILE="logs/${RUN_NAME}_output.log"

# Run the training script and tee output to log file
python3 train_improved.py \
    --timesteps 150000000 \
    --timesteps_per_iteration 10000 \
    --log_dir logs \
    --model_dir model \
    --run_name "${RUN_NAME}" \
    --n_envs ${NUM_ENVS} \
    --motion_file data/Walking.json \
    ${CONTINUE_FROM:+--continue_from "${CONTINUE_FROM}"} 2>&1 | tee "${LOG_FILE}"

echo "Enhanced training completed for ${RUN_NAME}" | tee -a "${LOG_FILE}" 