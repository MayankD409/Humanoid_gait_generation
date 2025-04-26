import time
import os
import inspect
import pybullet as p
import pybullet_data
import numpy as np
import math
from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger
from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv, InitializationStrategy

def build_arg_parser():
    arg_parser = ArgParser()
    arg_file = "run_humanoid3d_walk_args.txt"
    path = pybullet_data.getDataPath() + "/args/" + arg_file
    succ = arg_parser.load_file(path)
    assert succ, Logger.print2('Failed to load args from: ' + arg_file)
    
    # Load additional arguments to ensure stable walking
    additional_args = [
        "--sync_char_root_pos", "true",  # Keep models aligned in position
        "--sync_char_root_rot", "true",  # Keep models aligned in rotation
        "--enable_draw", "true",
        "--fall_contact_bodies", "0", "1", "2", "3", "4", "6", "7", "8", "9", "10", "12", "13", "14"
    ]
    arg_parser.load_args(additional_args)
    
    return arg_parser

def main():
    # Create environment with synced character root position
    arg_parser = build_arg_parser()
    env = PyBulletDeepMimicEnv(arg_parser, enable_draw=True, init_strategy=InitializationStrategy.START)
    
    print("Motion file:", arg_parser.parse_string("motion_file"))
    print("Starting simulation. Press 'q' to quit, 'r' to reset, 's' to slow motion.")
    
    # Reset the environment
    env.reset()
    
    # Physics parameters optimized for stable walking
    p.setPhysicsEngineParameter(numSolverIterations=5)      # Simple solver for better stability
    p.setPhysicsEngineParameter(numSubSteps=1)              # Keep it simple, one step
    p.setTimeStep(1/240)                                    # Standard timestep
    p.setGravity(0, -9.8, 0)                                # Normal gravity
    timeStep = 1/240                                        # Display timestep
    
    # Ground parameters that work well with the humanoid model
    p.changeDynamics(env._planeId, -1, 
                    lateralFriction=0.8,                    # Good friction level for walking
                    rollingFriction=0.0,                    # No rolling friction (simplifies physics)
                    spinningFriction=0.0,                   # No spinning friction (simplifies physics)
                    restitution=0.0,                        # No bounce
                    contactStiffness=10000,                 # Firm ground
                    contactDamping=50)                      # Some damping
    
    # Get joint information for better debugging
    joint_info = {}
    for j in range(p.getNumJoints(env._humanoid._sim_model)):
        info = p.getJointInfo(env._humanoid._sim_model, j)
        joint_info[j] = {"name": info[1].decode('utf-8'), "type": info[2]}
        print(f"Joint {j}: {info[1].decode('utf-8')}, Type: {info[2]}")
    
    # Configure all joints with basic parameters
    for j in range(p.getNumJoints(env._humanoid._sim_model)):
        p.changeDynamics(
            env._humanoid._sim_model, 
            j, 
            lateralFriction=0.5,
            jointDamping=0.1,                  # Very low damping for more responsiveness
            jointLimitForce=200,               # Lower joint limits for natural motion
            maxJointVelocity=50.0,             # Allow faster joint movement
        )
    
    # Set joint limits specifically for the key walking joints
    joint_limits = {
        # Right leg
        3: [-0.4, 1.4],    # Right hip (limit extension but allow good flexion)
        4: [0.0, 2.0],     # Right knee (prevent hyperextension, limit flexion)
        5: [-0.2, 0.2],    # Right ankle (moderate range for stability)
        # Left leg
        8: [-0.4, 1.4],    # Left hip (same as right)
        9: [0.0, 2.0],     # Left knee (same as right)
        10: [-0.2, 0.2],   # Left ankle (same as right)
    }
    
    # Apply joint limits
    for joint_id, limits in joint_limits.items():
        p.changeDynamics(
            env._humanoid._sim_model, 
            joint_id, 
            jointLowerLimit=limits[0], 
            jointUpperLimit=limits[1]
        )
    
    # Make kinematic model semi-transparent
    alpha = 0.3
    p.changeVisualShape(env._humanoid._kin_model, -1, rgbaColor=[1, 1, 1, alpha])
    for j in range(p.getNumJoints(env._humanoid._kin_model)):
        p.changeVisualShape(env._humanoid._kin_model, j, rgbaColor=[1, 1, 1, alpha])
    
    # Initialize simulation
    simTime = 0
    slowMotion = False
    
    # Very careful initialization of the physical model
    print("Initializing physical model...")
    
    # Make sure we start from frame 0
    env._humanoid.setSimTime(0)
    
    # Get the frame 0 pose
    mocapPose = env._humanoid.computePose(env._humanoid._frameFraction)
    
    # Apply it to the kinematic model
    env._humanoid.initializePose(env._humanoid._poseInterpolator, env._humanoid._kin_model, initBase=True)
    
    # Get base position and orientation for perfect initialization
    basePos, baseOrn = p.getBasePositionAndOrientation(env._humanoid._kin_model)
    
    # Position the physical model exactly at the same spot
    p.resetBasePositionAndOrientation(env._humanoid._sim_model, basePos, baseOrn)
    
    # Zero velocities
    p.resetBaseVelocity(env._humanoid._sim_model, [0, 0, 0], [0, 0, 0])
    
    # Copy exact joint positions from kinematic to physical model
    for j in range(p.getNumJoints(env._humanoid._sim_model)):
        if j < p.getNumJoints(env._humanoid._kin_model):
            kin_state = p.getJointState(env._humanoid._kin_model, j)
            p.resetJointState(env._humanoid._sim_model, j, kin_state[0], 0.0)  # Zero velocity
    
    # Let the model settle to find stable starting position
    print("Letting physics model settle...")
    for _ in range(5):
        p.stepSimulation()
    
    # Perfectly tuned PD controller gains for all joints
    # These values are critical for stable walking
    pd_gains = {
        # Torso and head - moderate control for balance
        1: {"kp": 300, "kd": 30},  # Chest - strong enough for upright posture
        2: {"kp": 100, "kd": 10},  # Neck - looser control
        
        # Right leg - strong control
        3: {"kp": 500, "kd": 50},  # Right hip - very strong control for stability
        4: {"kp": 500, "kd": 50},  # Right knee - strong for stance phase
        5: {"kp": 400, "kd": 40},  # Right ankle - strong for push-off
        
        # Right arm - low control
        6: {"kp": 100, "kd": 10},  # Right shoulder
        7: {"kp": 100, "kd": 10},  # Right elbow
        
        # Left leg - strong control
        8: {"kp": 500, "kd": 50},  # Left hip
        9: {"kp": 500, "kd": 50},  # Left knee
        10: {"kp": 400, "kd": 40}, # Left ankle
        
        # Left arm - low control
        11: {"kp": 100, "kd": 10}, # Left shoulder
        12: {"kp": 100, "kd": 10}, # Left elbow
    }
    
    # Map joint indices to their ranges in the maxForces array
    # This mapping is critical for proper PD controller behavior
    joint_indices = {
        1: {"start": 7, "size": 4},     # Chest (4 values)
        2: {"start": 11, "size": 4},    # Neck (4 values)
        3: {"start": 15, "size": 4},    # Right hip (4 values)
        4: {"start": 19, "size": 1},    # Right knee (1 value)
        5: {"start": 20, "size": 4},    # Right ankle (4 values)
        6: {"start": 24, "size": 4},    # Right shoulder (4 values)
        7: {"start": 28, "size": 1},    # Right elbow (1 value)
        8: {"start": 29, "size": 4},    # Left hip (4 values)
        9: {"start": 33, "size": 1},    # Left knee (1 value)
        10: {"start": 34, "size": 4},   # Left ankle (4 values)
        11: {"start": 38, "size": 4},   # Left shoulder (4 values)
        12: {"start": 42, "size": 1},   # Left elbow (1 value)
    }
    
    # Main simulation loop
    print("Starting main simulation...")
    while p.isConnected():
        # Adjust timestep for slow motion if enabled
        effectiveTimeStep = timeStep * (0.2 if slowMotion else 1.0)
        
        # Update the simulation time
        simTime += effectiveTimeStep
        env._humanoid.setSimTime(simTime)
        
        # Compute the kinematic pose for this time step
        mocapPose = env._humanoid.computePose(env._humanoid._frameFraction)
        
        # Apply the pose to the kinematic model
        env._humanoid.initializePose(env._humanoid._poseInterpolator, 
                                    env._humanoid._kin_model, 
                                    initBase=True)
        
        # Create force array for PD controller
        # This array defines how strongly each DOF will be controlled
        maxForces = [0] * 43  # Initialize with zeros
        
        # Root joints are not directly controlled
        maxForces[0:7] = [0, 0, 0, 0, 0, 0, 0]
        
        # Fill the array with our carefully tuned PD gains
        for joint_id, indices in joint_indices.items():
            if joint_id in pd_gains:
                start = indices["start"]
                size = indices["size"]
                kp_value = pd_gains[joint_id]["kp"]
                for i in range(size):
                    maxForces[start + i] = kp_value
        
        # Apply PD forces to make the physical model track the kinematic one
        env._humanoid.computeAndApplyPDForces(mocapPose, maxForces=maxForces)
        
        # Step the simulation
        p.stepSimulation()
        
        # Sleep to maintain real-time performance
        time.sleep(effectiveTimeStep)
        
        # Loop the walk cycle
        if simTime >= env._humanoid.getCycleTime():
            simTime = 0
            print("Cycle complete - restarting walk cycle")
        
        # Get keyboard events
        keys = env.getKeyboardEvents()
        
        # 'r' key to reset
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            print("Resetting simulation...")
            env.reset()
            simTime = 0
            
            # Re-initialize the physical model completely
            env._humanoid.setSimTime(0)
            mocapPose = env._humanoid.computePose(env._humanoid._frameFraction)
            env._humanoid.initializePose(env._humanoid._poseInterpolator, env._humanoid._kin_model, initBase=True)
            basePos, baseOrn = p.getBasePositionAndOrientation(env._humanoid._kin_model)
            p.resetBasePositionAndOrientation(env._humanoid._sim_model, basePos, baseOrn)
            p.resetBaseVelocity(env._humanoid._sim_model, [0, 0, 0], [0, 0, 0])
            
            # Reset all joint states
            for j in range(p.getNumJoints(env._humanoid._sim_model)):
                if j < p.getNumJoints(env._humanoid._kin_model):
                    kin_state = p.getJointState(env._humanoid._kin_model, j)
                    p.resetJointState(env._humanoid._sim_model, j, kin_state[0], 0.0)
            
            # Let physics settle
            for _ in range(5):
                p.stepSimulation()
        
        # 's' key to toggle slow motion
        if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
            slowMotion = not slowMotion
            print("Slow motion:", "ON" if slowMotion else "OFF")
        
        # 'q' key to exit
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break

if __name__ == "__main__":
    main()