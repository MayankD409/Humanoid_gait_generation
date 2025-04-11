#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dependency checker for humanoid imitation learning project.
This script checks if all required dependencies are installed
and offers to install missing ones.
"""

import os
import sys
import subprocess
import pkg_resources

def check_dependencies():
    """
    Check if all dependencies listed in requirements.txt are installed
    and offer to install missing ones.
    """
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    requirements_path = os.path.join(project_root, "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print("Error: requirements.txt not found at {}".format(requirements_path))
        return False
    
    # Parse requirements file
    with open(requirements_path, 'r') as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            requirements.append(line)
    
    # Check installed packages
    missing = []
    for requirement in requirements:
        try:
            # Extract package name (remove version specifiers)
            package_name = requirement.split('==')[0].split('>=')[0].split('<=')[0].strip()
            pkg_resources.get_distribution(package_name)
            print("✓ {} is installed".format(package_name))
        except pkg_resources.DistributionNotFound:
            missing.append(requirement)
            print("✗ {} is not installed".format(package_name))
    
    # Offer to install missing packages
    if missing:
        print("\n{} dependencies are missing.".format(len(missing)))
        answer = input("Do you want to install them? (y/n): ")
        
        if answer.lower() == 'y':
            for requirement in missing:
                print("\nInstalling {}...".format(requirement))
                subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print("\nAll dependencies have been installed.")
        else:
            print("\nPlease install the missing dependencies manually.")
            print("You can use 'pip install -r requirements.txt'")
    else:
        print("\nAll dependencies are installed correctly!")
    
    return True

if __name__ == "__main__":
    check_dependencies() 