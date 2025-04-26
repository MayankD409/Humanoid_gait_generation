import os
import gym
import numpy as np
import pybullet as p
from gym import spaces

# Import HumanoidDeepMimicWalkBulletEnv with error handling
try:
    from pybullet_envs.deep_mimic.gym_env import HumanoidDeepMimicWalkBulletEnv
except ImportError:
    try:
        # Try alternative class names if the exact one isn't found
        from pybullet_envs.deep_mimic.gym_env import HumanoidDeepMimicEnv as HumanoidDeepMimicWalkBulletEnv
        print("Warning: Using HumanoidDeepMimicEnv as the base environment. This may need configuration for walking tasks.")
    except ImportError:
        print("Error: Could not import HumanoidDeepMimicWalkBulletEnv or an alternative. Please check your pybullet_envs installation.")
        raise

class HumanoidImitationWalkEnv(HumanoidDeepMimicWalkBulletEnv):
    """
    A custom environment for humanoid walking imitation learning, directly inheriting
    from PyBullet's HumanoidDeepMimicWalkBulletEnv.
    """
    
    def __init__(self, renders=False, motion_file='data/Walking.json'):
        """
        Initialize the humanoid walking imitation environment.
        
        Args:
            renders (bool): Whether to render the environment
            motion_file (str): Path to the motion file to imitate
        """
        # Set the environment variable for the motion file before creating the env
        if motion_file:
            os.environ["PYBULLET_DEEP_MIMIC_MOTION_FILE"] = motion_file
        
        # Call the parent class constructor
        super().__init__(renders=renders)
        
        # Identify and store link indices for key body parts
        # These names might need to be adjusted based on the exact URDF structure
        try:
            # Try to get body part indices from the humanoid URDF
            self.left_foot_id = self._find_link_index('left_foot')
            self.right_foot_id = self._find_link_index('right_foot')
            self.left_hand_id = self._find_link_index('left_hand')
            self.right_hand_id = self._find_link_index('right_hand')
            self.torso_link_id = self._find_link_index('torso')
            
            print(f"Found link IDs: left_foot={self.left_foot_id}, right_foot={self.right_foot_id}, "
                  f"left_hand={self.left_hand_id}, right_hand={self.right_hand_id}, torso={self.torso_link_id}")
        except Exception as e:
            print(f"Warning: Could not find exact link names, using estimated indices: {e}")
            # Fallback to estimated link indices if names don't match
            # These might need tuning based on the actual URDF structure
            self.left_foot_id = 3  
            self.right_foot_id = 7
            self.left_hand_id = 11
            self.right_hand_id = 15
            self.torso_link_id = 1
        
        # Initialize an empty dictionary for reward components
        self.reward_components = {}

    def _find_link_index(self, link_name):
        """
        Find the index of a link by name in the URDF.
        
        Args:
            link_name (str): Name of the link to find
            
        Returns:
            int: The index of the link
        """
        # Try different ways to find the link index
        # Method 1: Try direct access to body parts if available
        if hasattr(self, 'humanoid') and hasattr(self.humanoid, 'parts'):
            if link_name in self.humanoid.parts:
                return self.humanoid.parts[link_name].bodyPartIndex
        
        # Method 2: Use PyBullet's internal methods if we have a body ID
        if hasattr(self, 'humanoid_id'):
            num_joints = p.getNumJoints(self.humanoid_id)
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.humanoid_id, i)
                if joint_info[12].decode("utf-8") == link_name:
                    return i
        
        # Method 3: Look for sim_id or robot_id attributes if humanoid_id is not available
        robot_id = getattr(self, 'sim_id', getattr(self, 'robot_id', None))
        if robot_id is not None:
            num_joints = p.getNumJoints(robot_id)
            for i in range(num_joints):
                joint_info = p.getJointInfo(robot_id, i)
                if joint_info[12].decode("utf-8") == link_name:
                    return i
        
        # If none of the methods worked, raise an exception
        raise ValueError(f"Could not find link '{link_name}' in the humanoid model")

    def _rewards(self):
        """
        Calculate rewards for the humanoid walking imitation task.
        
        Returns:
            float: The total reward for the current state
        """
        # Define weights for different reward components - NEEDS CAREFUL TUNING
        w_pose = 0.5       # Weight for pose matching
        w_vel = 0.1        # Weight for velocity matching
        w_end_eff = 0.15   # Weight for end-effector position matching
        w_com = 0.1        # Weight for center of mass tracking
        w_upright = 0.05   # Weight for keeping torso upright
        alive_bonus = 0.1  # Small constant reward for staying alive
        
        # Define scales for different reward components - NEEDS CAREFUL TUNING
        pose_scale = 2.0    # Scale for pose reward
        vel_scale = 0.1     # Scale for velocity reward
        end_eff_scale = 40  # Scale for end-effector reward
        com_scale = 10      # Scale for center of mass reward
        upright_scale = 5   # Scale for upright reward
        
        # 1. Pose Reward: Match joint angles with reference motion
        try:
            # Try using the base class's pose reward calculation if available
            if hasattr(self, '_calc_pose_reward'):
                pose_reward = self._calc_pose_reward() * pose_scale
            else:
                # Placeholder for manual pose reward calculation
                pose_reward = 0.0  # This should be replaced with actual calculation
                print("Warning: No _calc_pose_reward method found, using placeholder.")
        except Exception as e:
            print(f"Error calculating pose reward: {e}")
            pose_reward = 0.0
        
        # 2. Velocity Reward: Match joint velocities with reference motion
        try:
            # Try using the base class's velocity reward calculation if available
            if hasattr(self, '_calc_velocity_reward'):
                velocity_reward = self._calc_velocity_reward() * vel_scale
            else:
                # Placeholder for manual velocity reward calculation
                velocity_reward = 0.0  # This should be replaced with actual calculation
                print("Warning: No _calc_velocity_reward method found, using placeholder.")
        except Exception as e:
            print(f"Error calculating velocity reward: {e}")
            velocity_reward = 0.0
        
        # 3. End-Effector Reward: Match positions of hands and feet
        try:
            end_effector_reward = 0.0
            end_effector_count = this_error = 0
            
            # Calculate error for each end-effector if methods are available
            if hasattr(self, '_get_link_pos') and hasattr(self, '_get_ref_link_pos'):
                for link_id in [self.left_foot_id, self.right_foot_id, self.left_hand_id, self.right_hand_id]:
                    curr_pos = np.array(self._get_link_pos(link_id))
                    ref_pos = np.array(self._get_ref_link_pos(link_id))
                    this_error += np.sum((curr_pos - ref_pos) ** 2)
                    end_effector_count += 1
                
                if end_effector_count > 0:
                    avg_error = this_error / end_effector_count
                    end_effector_reward = np.exp(-end_eff_scale * avg_error)
            else:
                print("Warning: Methods for end-effector tracking not found, using placeholder.")
        except Exception as e:
            print(f"Error calculating end-effector reward: {e}")
            end_effector_reward = 0.0
        
        # 4. Center of Mass (CoM) Reward: Match CoM position and velocity
        try:
            com_reward = 0.0
            
            if hasattr(self, '_get_com_pos') and hasattr(self, '_get_ref_com_pos'):
                # Get current and reference CoM positions
                com_pos = np.array(self._get_com_pos())
                ref_com_pos = np.array(self._get_ref_com_pos())
                
                # Calculate error in CoM position (emphasize Z-axis height)
                com_pos_error = np.sum((com_pos - ref_com_pos) ** 2)
                
                # If velocity methods are available, also match CoM velocity
                if hasattr(self, '_get_com_vel') and hasattr(self, '_get_ref_com_vel'):
                    com_vel = np.array(self._get_com_vel())
                    ref_com_vel = np.array(self._get_ref_com_vel())
                    
                    # Calculate error in CoM velocity (emphasize X-axis forward motion)
                    # Penalize Y-axis (sideways) velocity more
                    com_vel_error = (com_vel[0] - ref_com_vel[0])**2 + \
                                   3.0 * (com_vel[1] - ref_com_vel[1])**2 + \
                                   (com_vel[2] - ref_com_vel[2])**2
                    
                    # Combine position and velocity errors
                    com_error = com_pos_error + com_vel_error
                else:
                    com_error = com_pos_error
                
                com_reward = np.exp(-com_scale * com_error)
            else:
                print("Warning: Methods for CoM tracking not found, using placeholder.")
        except Exception as e:
            print(f"Error calculating CoM reward: {e}")
            com_reward = 0.0
        
        # 5. Upright Reward: Keep the torso upright
        try:
            upright_reward = 0.0
            
            if hasattr(self, '_get_link_orientation'):
                # Get torso orientation
                torso_quat = self._get_link_orientation(self.torso_link_id)
                
                # Convert quaternion to rotation matrix
                # This is a simplified method for extracting the up vector (Z-axis)
                # in world coordinates from the quaternion
                qw, qx, qy, qz = torso_quat
                
                # Calculate the dot product with world up vector [0,0,1]
                # This gives the cosine of the angle between torso up and world up
                up_vector_z = 2.0 * (qx*qz + qw*qy)
                
                # Apply reward function that peaks at 1 (when perfectly upright)
                # and falls off exponentially as the torso tilts
                upright_reward = np.exp(-upright_scale * (1.0 - up_vector_z)**2)
            else:
                print("Warning: Method for getting link orientation not found, using placeholder.")
        except Exception as e:
            print(f"Error calculating upright reward: {e}")
            upright_reward = 0.0
        
        # Combine all reward components
        total_reward = (
            w_pose * pose_reward +
            w_vel * velocity_reward +
            w_end_eff * end_effector_reward +
            w_com * com_reward +
            w_upright * upright_reward +
            alive_bonus
        )
        
        # Store individual reward components
        self.reward_components = {
            'pose_reward': pose_reward,
            'velocity_reward': velocity_reward,
            'end_effector_reward': end_effector_reward,
            'com_reward': com_reward,
            'upright_reward': upright_reward,
            'alive_bonus': alive_bonus,
            'total_reward': total_reward
        }
        
        return total_reward

# Custom wrapper to handle the gym 0.26.2 API with stable-baselines3 1.2.0
class GymAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        # Handle the gym 0.26.2 reset which returns (obs, info)
        # but stable-baselines3 1.2.0 expects just obs
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0]  # Just return the observation
        return result
        
    def step(self, action):
        # Handle the gym 0.26.2 step which returns (obs, reward, terminated, truncated, info)
        # but stable-baselines3 1.2.0 expects (obs, reward, done, info)
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            
            # Pass through reward components if available
            if hasattr(self.env, 'reward_components'):
                info['reward_components'] = self.env.reward_components
                
            return obs, reward, done, info
        elif len(result) == 4:
            obs, reward, done, info = result
            
            # Pass through reward components if available
            if hasattr(self.env, 'reward_components'):
                info['reward_components'] = self.env.reward_components
                
            return result
        
        return result 