#!/bin/bash

# Setup script for Humanoid Imitation Learning Project

echo "Setting up Humanoid Imitation Learning environment..."

# Check if Python is installed
if ! [ -x "$(command -v python3)" ]; then
  echo 'Error: Python 3 is not installed.' >&2
  exit 1
fi

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data
fi

echo "Setup complete! The environment is ready."
echo "To activate the virtual environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To run the simulation, use:"
echo "python src/simulation_setup.py" 