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
    
    # Configure debug visualizer
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    
    # Load the ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # 2. Load Humanoid Robot
    print("Loading Atlas robot...")
    
    # Set the initial position and orientation of the robot
    startPos = [0, 0, 1.3]  # Starting position (raised higher for Atlas)
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # Starting orientation
    
    # Define the path to the Atlas URDF file in the local atlas_description folder
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    atlas_urdf_path = os.path.join(current_dir, "atlas_description", "atlas_v4_with_multisense.urdf")
    
    print("Loading Atlas from: {}".format(atlas_urdf_path))
    
    # Load the robot
    robotId = p.loadURDF(atlas_urdf_path, startPos, startOrientation)
    
    if robotId is None:
        print("Failed to load the Atlas robot. Trying a fallback humanoid model...")
        # Try an alternative robot if Atlas is not available
        robotId = p.loadURDF("humanoid/humanoid.urdf", startPos, startOrientation)
    
    if robotId is None:
        print("Failed to load any humanoid robot. Exiting...")
        p.disconnect()
        return
    
    # Initialize Atlas to a stable standing position
    print("Initializing Atlas to a stable pose...")
    
    # Dictionary of joint names to initial angles for a stable pose
    # These values are approximate - you may need to adjust them
    initial_poses = {
        "back_bky": 0.0,      # Keep torso upright
        "back_bkx": 0.0,
        "back_bkz": 0.0,
        "l_leg_hpy": -0.1,    # Slight hip bend
        "r_leg_hpy": -0.1,
        "l_leg_kny": 0.2,     # Slight knee bend
        "r_leg_kny": 0.2,
        "l_leg_aky": 0.0,     # Neutral ankles
        "r_leg_aky": 0.0,
        "l_arm_shx": -0.1,    # Arms slightly out
        "r_arm_shx": 0.1,
        "l_arm_ely": 0.5,     # Slight elbow bend
        "r_arm_ely": 0.5,
    }
    
    # Set initial joint poses
    joint_name_to_index = {}
    for i in range(p.getNumJoints(robotId)):
        joint_info = p.getJointInfo(robotId, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_name_to_index[joint_name] = i
    
    # Apply initial poses
    for joint_name, angle in initial_poses.items():
        if joint_name in joint_name_to_index:
            p.resetJointState(robotId, joint_name_to_index[joint_name], angle)
    
    # Let the robot settle briefly in the starting pose
    for _ in range(50):
        p.stepSimulation()
    
    # Add a fixed base constraint to keep the robot stable during the demo
    # This can be removed later when implementing actual walking
    fixed_base = p.createConstraint(
        robotId, -1, -1, -1,
        p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1.3]
    )
    
    # 3. Configure Visualization
    print("Configuring visualization...")
    p.resetDebugVisualizerCamera(
        cameraDistance=5.0,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 1.0]
    )
    
    # 4. Inspect Robot Joints
    print("Inspecting robot joints...")
    num_joints = p.getNumJoints(robotId)
    print("Number of joints: {}".format(num_joints))
    
    # Print information for all joints and save to a file
    print("Joint information:")
    print("Index\tName\t\tType\tLower\tUpper")
    print("-" * 50)
    
    # Save joint info to a file for debugging
    joint_info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "joint_info.txt")
    with open(joint_info_path, 'w') as f:
        f.write("Joint information for Atlas robot:\n")
        f.write("Index\tName\t\tType\tLower\tUpper\n")
        f.write("-" * 50 + "\n")
    
    # Store key joint indices for motion control
    knee_joint_idx = None
    elbow_joint_idx = None
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robotId, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        
        # Print joint info
        print("{}\t{}\t{}\t{:.2f}\t{:.2f}".format(i, joint_name, joint_type, lower_limit, upper_limit))
        
        # Save joint info to file
        with open(joint_info_path, 'a') as f:
            f.write("{}\t{}\t{}\t{:.2f}\t{:.2f}\n".format(i, joint_name, joint_type, lower_limit, upper_limit))
        
        # Identify joints for motion control (Atlas joint naming convention)
        if ("knee" in joint_name.lower() or "l_leg_kny" in joint_name.lower()) and knee_joint_idx is None:
            knee_joint_idx = i
        elif ("elbow" in joint_name.lower() or "l_arm_ely" in joint_name.lower()) and elbow_joint_idx is None:
            elbow_joint_idx = i
    
    # If we couldn't find specific joints by name, use some default indices
    # For the Atlas robot, use the known indices based on the joint information
    if knee_joint_idx is None:
        knee_joint_idx = 21  # l_leg_kny (left knee joint)
    if elbow_joint_idx is None:
        elbow_joint_idx = 5  # l_arm_ely (left elbow joint)
    
    print("Selected joints for motion control: Knee: {}, Elbow: {}".format(knee_joint_idx, elbow_joint_idx))
    
    # 5. Create Dummy Motion Data
    print("Creating dummy motion data...")
    num_steps = 150  # Number of simulation steps for the motion
    
    # Create sine wave motion data for the selected joints
    time_steps = np.linspace(0, 2*np.pi, num_steps)
    knee_angles = 0.3 * np.sin(time_steps) + 0.5  # Sine wave centered around 0.5 radians
    elbow_angles = 0.4 * np.sin(time_steps + np.pi/4) + 1.0  # Offset sine wave centered around 1.0 radians
    
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
                force=100  # Lower force for smoother motion
            )
            
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=elbow_joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=elbow_angle,
                force=100  # Lower force for smoother motion
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