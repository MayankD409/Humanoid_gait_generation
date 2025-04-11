#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation setup for humanoid imitation learning project.
This script initializes the PyBullet environment, loads a humanoid robot,
and demonstrates playback of dummy motion data.
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data

def main():
    # 1. Initialize PyBullet
    print("Initializing PyBullet...")
    physicsClient = p.connect(p.GUI)  # Connect to the PyBullet physics server with GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add pybullet_data path
    p.setGravity(0, 0, -9.81)  # Set gravity
    
    # Load the ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # 2. Load Humanoid Robot
    print("Loading humanoid robot...")
    
    # Set the initial position and orientation of the robot
    startPos = [0, 0, 1.0]  # Starting position
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # Starting orientation
    
    # Load the robot (using the built-in humanoid model)
    robotId = p.loadURDF("humanoid/humanoid.urdf", startPos, startOrientation)
    
    if robotId is None:
        print("Failed to load the humanoid robot. Exiting...")
        p.disconnect()
        return
    
    # 3. Configure Visualization
    print("Configuring visualization...")
    p.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 1.0]
    )
    
    # 4. Inspect Robot Joints
    print("Inspecting robot joints...")
    num_joints = p.getNumJoints(robotId)
    print("Number of joints: {}".format(num_joints))
    
    # Print information for the first 15 joints
    print("Joint information (showing first 15):")
    print("Index\tName\t\tType\tLower\tUpper")
    print("-" * 50)
    
    # Store key joint indices for motion control
    knee_joint_idx = None
    elbow_joint_idx = None
    
    for i in range(min(15, num_joints)):
        joint_info = p.getJointInfo(robotId, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        
        print("{}\t{}\t{}\t{:.2f}\t{:.2f}".format(i, joint_name, joint_type, lower_limit, upper_limit))
        
        # Identify joints for motion control (assuming typical joint names)
        if "knee" in joint_name.lower() and knee_joint_idx is None:
            knee_joint_idx = i
        elif "elbow" in joint_name.lower() and elbow_joint_idx is None:
            elbow_joint_idx = i
    
    # If we couldn't find specific joints by name, use some default indices
    # For the humanoid.urdf model, use appropriate joint indices based on inspection
    if knee_joint_idx is None:
        knee_joint_idx = 10  # right_knee joint (index 10)
    if elbow_joint_idx is None:
        elbow_joint_idx = 4  # right_elbow joint (index 4)
    
    print("Selected joints for motion control: Knee: {}, Elbow: {}".format(knee_joint_idx, elbow_joint_idx))
    
    # 5. Create Dummy Motion Data
    print("Creating dummy motion data...")
    num_steps = 150  # Number of simulation steps for the motion
    
    # Create sine wave motion data for the selected joints
    time_steps = np.linspace(0, 2*np.pi, num_steps)
    knee_angles = 0.5 * np.sin(time_steps) + 0.2  # Sine wave around 0.2 radians
    elbow_angles = 0.4 * np.sin(time_steps + np.pi/4) + 0.1  # Offset sine wave
    
    # Combine the joint angles into a single array
    motion_data = np.column_stack((knee_angles, elbow_angles))
    
    # Save the motion data to a file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    # Create directory if it doesn't exist (Python 2.7 compatible)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    motion_data_path = os.path.join(data_dir, "dummy_motion.npy")
    np.save(motion_data_path, motion_data)
    print("Saved dummy motion data to {}".format(motion_data_path))
    
    # 6. Load Dummy Motion Data
    print("Loading dummy motion data...")
    loaded_motion_data = np.load(motion_data_path)
    print("Loaded motion data with shape: {}".format(loaded_motion_data.shape))
    
    # 7. Implement Simulation Loop with Motion Playback
    print("Starting simulation loop with motion playback...")
    step = 0
    
    try:
        while p.isConnected():
            # Get current target joint angles
            if step < len(loaded_motion_data):
                knee_angle, elbow_angle = loaded_motion_data[step]
            else:
                # Loop back to the beginning when we reach the end
                step = 0
                knee_angle, elbow_angle = loaded_motion_data[step]
            
            # Apply the joint controls
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=knee_joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=knee_angle,
                force=500  # Adjust force as needed
            )
            
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=elbow_joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=elbow_angle,
                force=500  # Adjust force as needed
            )
            
            # Step the simulation
            p.stepSimulation()
            
            # Small delay to control simulation speed
            time.sleep(1./240.)
            
            # Increment step counter
            step += 1
            
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        # 8. Disconnect from the physics server
        p.disconnect()
        print("Disconnected from PyBullet")

if __name__ == "__main__":
    main() 