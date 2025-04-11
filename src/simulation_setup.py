#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation setup for humanoid imitation learning project.
This script initializes the PyBullet environment, loads the Atlas robot,
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
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Enable shadows for better visualization
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    
    # Load the ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # 2. Load Atlas Robot
    print("Loading Atlas robot...")
    
    # Set the initial position and orientation of the robot
    # Position the robot so feet are directly on the ground
    startPos = [0, 0, 0.93]  # Lower position to make feet touch the ground
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # Starting orientation
    
    # Define the path to the Atlas URDF file in the local atlas_description folder
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    atlas_urdf_path = os.path.join(current_dir, "atlas_description", "atlas_v4_with_multisense.urdf")
    
    print("Loading Atlas from: {}".format(atlas_urdf_path))
    
    # Load the robot with fixed base initially to set up the pose
    # This prevents the robot from falling during initialization
    robotId = p.loadURDF(atlas_urdf_path, startPos, startOrientation, useFixedBase=True)
    
    if robotId is None:
        print("Failed to load the Atlas robot. Trying a fallback humanoid model...")
        # Try an alternative robot if Atlas is not available
        robotId = p.loadURDF("humanoid/humanoid.urdf", startPos, startOrientation, useFixedBase=True)
    
    if robotId is None:
        print("Failed to load any humanoid robot. Exiting...")
        p.disconnect()
        return
    
    # 3. Configure Visualization
    print("Configuring visualization...")
    p.resetDebugVisualizerCamera(
        cameraDistance=5.0,
        cameraYaw=30,
        cameraPitch=-20,  # Better angle to see the robot
        cameraTargetPosition=[0, 0, 1.0]
    )
    
    # 4. Apply a stable standing pose
    print("Setting Atlas to a stable pose...")
    
    # Get all joint indices and names for easier reference
    joint_name_to_index = {}
    for i in range(p.getNumJoints(robotId)):
        joint_info = p.getJointInfo(robotId, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_name_to_index[joint_name] = i
    
    # Disable the default motor control for all joints
    for joint_name, joint_index in joint_name_to_index.items():
        p.setJointMotorControl2(robotId, joint_index, p.VELOCITY_CONTROL, force=0)
    
    # Dictionary of joint names to initial angles for a stable pose
    initial_poses = {
        # Torso
        "back_bkz": 0.0,
        "back_bky": 0.05,  # Slight forward tilt for balance
        "back_bkx": 0.0,
        
        # Left leg
        "l_leg_hpz": 0.0,
        "l_leg_hpx": 0.0,
        "l_leg_hpy": -0.2,  # Bend at hip for stability
        "l_leg_kny": 0.4,   # Bend knee for stability
        "l_leg_aky": -0.2,  # Ankle compensates for knee bend
        "l_leg_akx": 0.0,
        
        # Right leg
        "r_leg_hpz": 0.0,
        "r_leg_hpx": 0.0,
        "r_leg_hpy": -0.2,  # Bend at hip for stability  
        "r_leg_kny": 0.4,   # Bend knee for stability
        "r_leg_aky": -0.2,  # Ankle compensates for knee bend
        "r_leg_akx": 0.0,
        
        # Arms in neutral position
        "l_arm_shz": 0.0,
        "l_arm_shx": -0.1,
        "l_arm_ely": 1.5,   # Arm bent at elbow
        "l_arm_elx": 0.0,
        "l_arm_wry": 0.0,
        "l_arm_wrx": 0.0,
        "l_arm_wry2": 0.0,
        
        "r_arm_shz": 0.0,
        "r_arm_shx": 0.1,
        "r_arm_ely": 1.5,   # Arm bent at elbow
        "r_arm_elx": 0.0,
        "r_arm_wry": 0.0,
        "r_arm_wrx": 0.0,
        "r_arm_wry2": 0.0,
        
        # Neck
        "neck_ry": 0.0
    }
    
    # Apply initial poses to all joints
    for joint_name, angle in initial_poses.items():
        if joint_name in joint_name_to_index:
            joint_index = joint_name_to_index[joint_name]
            p.resetJointState(robotId, joint_index, angle)
    
    # Let the robot settle with fixed base
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)  # Slow down simulation for stability
    
    # Keep track of the original position
    basePos, baseOrn = p.getBasePositionAndOrientation(robotId)
    print("Base position after pose setup: {}".format(basePos))
    
    # Now apply strong motor control to hold the position
    for joint_name, angle in initial_poses.items():
        if joint_name in joint_name_to_index:
            joint_index = joint_name_to_index[joint_name]
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                positionGain=1.0,
                velocityGain=1.0,
                force=1000  # Strong force to maintain position
            )
    
    # Let the robot settle again with motor control
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)
    
    # Now that the robot is in a stable pose, we can switch to dynamic base
    # First, get current base pose
    basePos, baseOrn = p.getBasePositionAndOrientation(robotId)
    
    # Create a temporary robot with fixed base (needed because we can't change fixed base status)
    temp_robot_id = robotId
    
    # Reload the robot with the same pose but with dynamic base
    robotId = p.loadURDF(atlas_urdf_path, basePos, baseOrn, useFixedBase=False)
    
    # Remove the temporary robot
    p.removeBody(temp_robot_id)
    
    # Reapply the joint states to the new robot instance
    for joint_name, angle in initial_poses.items():
        if joint_name in joint_name_to_index:
            joint_index = joint_name_to_index[joint_name]
            p.resetJointState(robotId, joint_index, angle)
            
            # Apply strong position control to each joint
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                positionGain=1.0,
                velocityGain=1.0,
                force=1000  # Strong force to maintain position
            )
    
    # Let the robot settle with gravity but strong motor control
    print("Stabilizing the robot with dynamics...")
    for _ in range(200):
        p.stepSimulation()
        time.sleep(0.01)
    
    # 5. Inspect Robot Joints
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
    
    # 6. Create Dummy Motion Data
    print("Creating dummy motion data...")
    num_steps = 200  # More steps for smoother motion
    
    # Create smaller sine wave motion for stability (reduce amplitude)
    time_steps = np.linspace(0, 2*np.pi, num_steps)
    knee_angles = 0.1 * np.sin(time_steps) + initial_poses["l_leg_kny"]  # Small motion around initial pose
    elbow_angles = 0.2 * np.sin(time_steps + np.pi/4) + initial_poses["l_arm_ely"]  # Small motion around initial pose
    
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
    
    # 7. Load Dummy Motion Data
    print("Loading dummy motion data...")
    loaded_motion_data = np.load(motion_data_path)
    print("Loaded motion data with shape: {}".format(loaded_motion_data.shape))
    
    # 8. Implement Simulation Loop with Motion Playback
    print("Starting simulation loop with motion playback...")
    print("Press Ctrl+C to exit...")
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
            
            # Keep applying strong control to all joints except the ones we're animating
            for joint_name, angle in initial_poses.items():
                if joint_name in joint_name_to_index:
                    joint_index = joint_name_to_index[joint_name]
                    
                    # Skip the joints we're animating separately
                    if joint_index == knee_joint_idx or joint_index == elbow_joint_idx:
                        continue
                        
                    p.setJointMotorControl2(
                        bodyUniqueId=robotId,
                        jointIndex=joint_index,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=angle,
                        positionGain=1.0,
                        velocityGain=1.0,
                        force=1000  # Strong force to maintain position
                    )
            
            # Apply the joint controls with smoother motion
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=knee_joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=knee_angle,
                positionGain=0.5,  # Lower gain for smoother motion
                velocityGain=0.5,
                force=100  # Lower force for smoother motion
            )
            
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=elbow_joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=elbow_angle,
                positionGain=0.5,  # Lower gain for smoother motion
                velocityGain=0.5,
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
        # 9. Disconnect from the physics server
        p.disconnect()
        print("Disconnected from PyBullet")

if __name__ == "__main__":
    main() 