#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation setup for humanoid imitation learning project.
This script initializes the PyBullet environment, loads the Atlas robot,
and demonstrates a stable standing pose with basic joint animation.
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data

def main():
    # Initialize PyBullet
    print("Initializing PyBullet...")
    p.connect(p.GUI)  # Connect to the PyBullet physics server with GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Configure visualization
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    
    # Load the ground plane
    p.loadURDF("plane.urdf")
    
    # Load Atlas Robot
    print("Loading Atlas robot...")
    start_pos = [0, 0, 0.93]  # Position robot so feet touch the ground
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
    # Path to Atlas URDF
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    atlas_urdf_path = os.path.join(current_dir, "atlas_description", "atlas_v4_with_multisense.urdf")
    
    # Load robot with fixed base for initial pose setting
    robot_id = p.loadURDF(atlas_urdf_path, start_pos, start_orientation, useFixedBase=True)
    
    if robot_id is None:
        print("Failed to load Atlas robot. Exiting...")
        p.disconnect()
        return
    
    # Configure camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=5.0,
        cameraYaw=30,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 1.0]
    )
    
    # Create mapping of joint names to indices
    joint_name_to_index = {}
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_name_to_index[joint_name] = i
    
    # Disable default motor control for all joints
    for joint_index in joint_name_to_index.values():
        p.setJointMotorControl2(robot_id, joint_index, p.VELOCITY_CONTROL, force=0)
    
    # Define stable standing pose
    initial_poses = {
        # Torso
        "back_bkz": 0.0, "back_bky": 0.05, "back_bkx": 0.0,
        
        # Left leg
        "l_leg_hpz": 0.0, "l_leg_hpx": 0.0, "l_leg_hpy": -0.2,
        "l_leg_kny": 0.4, "l_leg_aky": -0.2, "l_leg_akx": 0.0,
        
        # Right leg
        "r_leg_hpz": 0.0, "r_leg_hpx": 0.0, "r_leg_hpy": -0.2,
        "r_leg_kny": 0.4, "r_leg_aky": -0.2, "r_leg_akx": 0.0,
        
        # Left arm
        "l_arm_shz": 0.0, "l_arm_shx": -0.1, "l_arm_ely": 1.5,
        "l_arm_elx": 0.0, "l_arm_wry": 0.0, "l_arm_wrx": 0.0, "l_arm_wry2": 0.0,
        
        # Right arm
        "r_arm_shz": 0.0, "r_arm_shx": 0.1, "r_arm_ely": 1.5,
        "r_arm_elx": 0.0, "r_arm_wry": 0.0, "r_arm_wrx": 0.0, "r_arm_wry2": 0.0,
        
        # Neck
        "neck_ry": 0.0
    }
    
    # Apply initial pose
    print("Setting Atlas to a stable pose...")
    for joint_name, angle in initial_poses.items():
        if joint_name in joint_name_to_index:
            joint_index = joint_name_to_index[joint_name]
            p.resetJointState(robot_id, joint_index, angle)
    
    # Let the robot settle with fixed base
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)
    
    # Apply position control to all joints
    for joint_name, angle in initial_poses.items():
        if joint_name in joint_name_to_index:
            joint_index = joint_name_to_index[joint_name]
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                positionGain=1.0,
                velocityGain=1.0,
                force=1000
            )
    
    # Let robot settle again with motor control
    for _ in range(50):
        p.stepSimulation()
        time.sleep(0.01)
    
    # Switch to dynamic base (reload robot)
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    temp_robot_id = robot_id
    robot_id = p.loadURDF(atlas_urdf_path, base_pos, base_orn, useFixedBase=False)
    p.removeBody(temp_robot_id)
    
    # Reapply joint states and control to the new robot instance
    for joint_name, angle in initial_poses.items():
        if joint_name in joint_name_to_index:
            joint_index = joint_name_to_index[joint_name]
            p.resetJointState(robot_id, joint_index, angle)
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=1000
            )
    
    # Let the robot stabilize with dynamics
    print("Stabilizing the robot...")
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)
    
    # Get joint indices for animation
    knee_joint_idx = joint_name_to_index.get("l_leg_kny", 21)  # Default to 21 if not found
    elbow_joint_idx = joint_name_to_index.get("l_arm_ely", 5)  # Default to 5 if not found
    
    # Create simple animation data
    print("Starting simulation with basic joint animation...")
    num_steps = 200
    time_steps = np.linspace(0, 2*np.pi, num_steps)
    knee_angles = 0.1 * np.sin(time_steps) + initial_poses["l_leg_kny"]
    elbow_angles = 0.2 * np.sin(time_steps + np.pi/4) + initial_poses["l_arm_ely"]
    motion_data = np.column_stack((knee_angles, elbow_angles))
    
    # Main simulation loop
    print("Press Ctrl+C to exit...")
    step = 0
    
    try:
        while p.isConnected():
            # Loop animation data
            if step >= len(motion_data):
                step = 0
            
            knee_angle, elbow_angle = motion_data[step]
            
            # Maintain position control on all joints except animated ones
            for joint_name, angle in initial_poses.items():
                if joint_name in joint_name_to_index:
                    joint_index = joint_name_to_index[joint_name]
                    
                    # Skip the joints we're animating
                    if joint_index == knee_joint_idx or joint_index == elbow_joint_idx:
                        continue
                    
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=joint_index,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=angle,
                        force=1000
                    )
            
            # Apply animation to selected joints
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=knee_joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=knee_angle,
                positionGain=0.5,
                velocityGain=0.5,
                force=100
            )
            
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=elbow_joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=elbow_angle,
                positionGain=0.5,
                velocityGain=0.5,
                force=100
            )
            
            p.stepSimulation()
            time.sleep(1./240.)
            step += 1
            
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        p.disconnect()
        print("Disconnected from PyBullet")

if __name__ == "__main__":
    main() 