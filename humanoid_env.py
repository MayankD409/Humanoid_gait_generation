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
    def __init__(self, renders=False, motion_file='data/Walking.json'):
        if motion_file:
            os.environ["PYBULLET_DEEP_MIMIC_MOTION_FILE"] = motion_file

        # ***** STEP 1: Call the parent constructor FIRST *****
        # This will load the URDF and initialize self.humanoid
        super().__init__(renders=renders)

        # Store the PyBullet client ID 
        self.p_client = self._p
        # Initialize reward components dictionary
        self.reward_components = {}

        # ***** STEP 2: Now that self.humanoid exists, assign the link indices DIRECTLY *****
        # Indices based on the default PyBullet humanoid structure inferred from examples
        # Note: PyBullet often maps joint index i to link index i.
        try:
            # Indices corresponding to ankle joints (for feet), elbow joints (for hands), and chest joint (for torso)
            # Updated based on provided joint info
            self.left_foot_id = 11  # Index of left_ankle joint/link (Corrected from 10)
            self.right_foot_id = 5   # Index of right_ankle joint/link
            self.left_hand_id = 13  # Index of left_elbow joint/link (Corrected from 12)
            self.right_hand_id = 7   # Index of right_elbow joint/link
            self.torso_link_id = 1   # Index of chest joint/link

            print(f"Using hardcoded link IDs: left_foot={self.left_foot_id}, right_foot={self.right_foot_id}, "
                  f"left_hand={self.left_hand_id}, right_hand={self.right_hand_id}, torso/pelvis={self.torso_link_id}")

            # # Basic check if humanoid has enough links for these indices (REMOVED due to initialization order issue)
            # num_joints = self.p_client.getNumJoints(self.humanoid.model_id) 
            # max_expected_index = max(self.left_foot_id, self.right_foot_id, self.left_hand_id, self.right_hand_id, self.torso_link_id)
            # if num_joints <= max_expected_index:
            #     print(f"Error: Humanoid model has only {num_joints} joints/links, but index {max_expected_index} is expected. Check indices.")
            #     # Reset IDs to -1 to prevent errors later
            #     self.left_foot_id = -1
            #     self.right_foot_id = -1
            #     self.left_hand_id = -1
            #     self.right_hand_id = -1
            #     self.torso_link_id = -1

        except Exception as e:
            print(f"Error during link index assignment: {e}")
            self.left_foot_id = -1
            self.right_foot_id = -1
            self.left_hand_id = -1
            self.right_hand_id = -1
            self.torso_link_id = -1

    def _find_link_index(self, link_name):
        """
        Find the index of a link by name in the URDF.
        
        Args:
            link_name (str): Name of the link to find
            
        Returns:
            int: The index of the link, or -1 if not found
        """
        # Ensure we have a valid PyBullet client and humanoid ID
        if not hasattr(self, 'humanoid') or not hasattr(self.humanoid, 'model_id'):
            print(f"Warning: Cannot find link '{link_name}', humanoid model not fully initialized.")
            return -1 # Indicate not found or error

        humanoid_id = self.humanoid.model_id

        # Iterate through joints to find the link name
        num_joints = self.p_client.getNumJoints(humanoid_id)
        for i in range(num_joints):
            try:
                joint_info = self.p_client.getJointInfo(humanoid_id, i)
                # Link name is typically stored in joint_info[12]
                if joint_info[12].decode("utf-8") == link_name:
                    return i # Return the joint index which corresponds to the link index in PyBullet
            except Exception as e:
                print(f"Error querying joint info for index {i}: {e}")
                continue # Skip problematic joints if necessary

        # Handle base link case - simplified check
        print(f"Warning: Link '{link_name}' not found.")
        return -1 # Indicate not found

    def _rewards(self):
        """
        Calculate rewards for the humanoid walking imitation task. Includes pose,
        velocity, end-effector, COM matching, uprightness, forward progress,
        and survival bonus.
        
        Returns:
            float: The total reward for the current state
        """
        # --- Weights (Tunable) ---
        w_pose = 0.15        # Decreased pose matching to reduce over-constraint
        w_vel = 0.10         # Velocity matching (Same)
        w_end_eff = 0.20     # Increased end-effector position matching for better foot placement
        w_com = 0.15         # Increased center of mass tracking for better balance
        w_upright = 0.20     # Increased for stability
        w_forward = 0.30     # Further increased forward progress to encourage walking
        w_balance = 0.15     # New: reward for maintaining balance
        alive_bonus = 0.10   # Increased survival bonus

        # --- Scales (Tunable) ---
        pose_scale = 1.0     # Reduced to be less punishing for minor deviations
        vel_scale = 0.1      # Keep the same
        end_eff_scale = 20   # Reduced to be less punishing
        com_scale = 5        # Reduced to be less punishing
        upright_scale = 3    # Same
        forward_scale = 5    # Increased to encourage forward motion
        balance_scale = 10   # Scale for balance reward

        total_reward = 0.0
        self.reward_components = {} # Reset components dict

        # --- Get Base/Root State ---
        # Ensure self.humanoid.model_id is the correct body index
        try:
            root_pos, root_orn = self.p_client.getBasePositionAndOrientation(self.humanoid.model_id)
            root_lin_vel, root_ang_vel = self.p_client.getBaseVelocity(self.humanoid.model_id)
        except Exception as e:
             print(f"Error getting base state: {e}")
             # Handle error: maybe return 0 reward or raise an exception
             return 0.0 # Early exit if base state fails

        # --- Calculate Reward Components ---

        # 1. Pose Reward
        try:
            if hasattr(self, '_calc_pose_reward'):
                pose_reward = self._calc_pose_reward() # Base class method might return scaled value
                # If not scaled by base method, apply scale: pose_reward = np.exp(-pose_scale * pose_error)
                self.reward_components['pose'] = w_pose * pose_reward # Apply weight
                total_reward += self.reward_components['pose']
            else:
                print("Warning: _calc_pose_reward method not found.")
                self.reward_components['pose'] = 0.0
        except Exception as e:
            print(f"Error calculating pose reward: {e}")
            self.reward_components['pose'] = 0.0

        # 2. Velocity Reward
        try:
            if hasattr(self, '_calc_velocity_reward'):
                velocity_reward = self._calc_velocity_reward() # Base class method might return scaled value
                # If not scaled by base method, apply scale: velocity_reward = np.exp(-vel_scale * vel_error)
                self.reward_components['velocity'] = w_vel * velocity_reward # Apply weight
                total_reward += self.reward_components['velocity']
            else:
                print("Warning: _calc_velocity_reward method not found.")
                self.reward_components['velocity'] = 0.0
        except Exception as e:
            print(f"Error calculating velocity reward: {e}")
            self.reward_components['velocity'] = 0.0

        # 3. End-Effector Reward (Feet + Hands)
        try:
            end_effector_reward = 0.0
            if hasattr(self, '_calc_end_effector_reward'): # Check for specific base method
                 end_effector_reward = self._calc_end_effector_reward()
                 # If base method doesn't exist, manually calculate if possible
            elif hasattr(self, '_get_link_pos') and hasattr(self, '_get_ref_link_pos'):
                this_error = 0
                count = 0
                # Ensure link IDs are valid (not -1)
                link_ids = [self.left_foot_id, self.right_foot_id, self.left_hand_id, self.right_hand_id]
                valid_link_ids = [lid for lid in link_ids if lid != -1]

                for link_id in valid_link_ids:
                    curr_pos = np.array(self._get_link_pos(link_id))
                    ref_pos = np.array(self._get_ref_link_pos(link_id)) # Assumes this gets ref pos for current phase
                    this_error += np.sum((curr_pos - ref_pos) ** 2)
                    count += 1

                if count > 0:
                    avg_error = this_error / count
                    end_effector_reward = np.exp(-end_eff_scale * avg_error)
            else:
                 print("Warning: End-effector tracking methods not found.")

            self.reward_components['end_effector'] = w_end_eff * end_effector_reward # Apply weight
            total_reward += self.reward_components['end_effector']
        except Exception as e:
            print(f"Error calculating end-effector reward: {e}")
            self.reward_components['end_effector'] = 0.0

        # 4. Center of Mass (CoM) Reward
        try:
            com_reward = 0.0
            if hasattr(self, '_calc_com_reward'): # Check for specific base method
                 com_reward = self._calc_com_reward()
                 # Add manual calculation here if base method doesn't exist and helpers do
            elif hasattr(self, '_get_com_pos') and hasattr(self, '_get_ref_com_pos'):
                 com_pos = np.array(self._get_com_pos())
                 ref_com_pos = np.array(self._get_ref_com_pos()) # Assumes ref for current phase
                 com_pos_error = np.sum((com_pos - ref_com_pos) ** 2)
                 # Optionally include velocity error if methods exist
                 com_vel_error = 0.0
                 if hasattr(self, '_get_com_vel') and hasattr(self, '_get_ref_com_vel'):
                    com_vel = np.array(self._get_com_vel())
                    ref_com_vel = np.array(self._get_ref_com_vel())
                    com_vel_error = np.sum((com_vel - ref_com_vel) ** 2)

                 com_error = com_pos_error + 0.1 * com_vel_error # Example weighting
                 com_reward = np.exp(-com_scale * com_error)
            else:
                 print("Warning: CoM tracking methods not found.")

            self.reward_components['com'] = w_com * com_reward # Apply weight
            total_reward += self.reward_components['com']
        except Exception as e:
            print(f"Error calculating CoM reward: {e}")
            self.reward_components['com'] = 0.0

        # 5. Upright Reward (Based on Torso Orientation)
        try:
            upright_reward = 0.0
            # Ensure torso_link_id is valid
            if self.torso_link_id != -1 and hasattr(self, '_get_link_orientation'):
                torso_quat = self._get_link_orientation(self.torso_link_id)
            # If torso_link_id is invalid or method missing, try getting base orientation
            elif self.torso_link_id == -1 : # Use base orientation if torso link is base
                 torso_quat = root_orn
            else:
                 print("Warning: Cannot get torso orientation for upright reward.")
                 torso_quat = None # Indicate orientation couldn't be obtained

            if torso_quat is not None:
                # Calculate up vector's Z component from quaternion [x, y, z, w]
                qx, qy, qz, qw = torso_quat[0], torso_quat[1], torso_quat[2], torso_quat[3]
                # Simplified Z component of the rotated Z-axis (assumes passive rotation)
                # For active rotation (frame rotates): rot_mat = self.p_client.getMatrixFromQuaternion(torso_quat); up_z = rot_mat[8]
                # For passive rotation (vector rotates):
                up_z = 1.0 - 2.0 * (qx*qx + qy*qy) # Z component of [0,0,1] rotated by torso_quat

                # Reward measure based on Z component being close to 1
                # More punishment for tilting: (1 - up_z) is error, then squared
                upright_error = (1.0 - up_z) ** 2 
                upright_reward = np.exp(-upright_scale * upright_error)
            else:
                upright_reward = 0.0 # Default if no orientation available

            self.reward_components['upright'] = w_upright * upright_reward # Apply weight
            total_reward += self.reward_components['upright']
        except Exception as e:
            print(f"Error calculating upright reward: {e}")
            self.reward_components['upright'] = 0.0

        # 6. NEW: Balance Reward (Based on Angular Velocity)
        try:
            balance_reward = 0.0
            # Use root angular velocity to measure balance
            if root_ang_vel is not None:
                # Lower angular velocity = better balance
                ang_vel_magnitude = np.linalg.norm(root_ang_vel)
                balance_error = ang_vel_magnitude ** 2  # Square to emphasize large angular velocities
                balance_reward = np.exp(-balance_scale * balance_error)
            else:
                balance_reward = 0.0
                
            self.reward_components['balance'] = w_balance * balance_reward
            total_reward += self.reward_components['balance']
        except Exception as e:
            print(f"Error calculating balance reward: {e}")
            self.reward_components['balance'] = 0.0

        # 7. Forward Progress Reward
        try:
            forward_reward = 0.0
            if hasattr(self, '_calc_forward_reward'): # Check for method in base class
                forward_reward = self._calc_forward_reward()
            # Otherwise, use simple heuristic based on forward (X) component of velocity
            elif root_lin_vel is not None:
                # Reward for positive forward velocity
                forward_vel = root_lin_vel[0] # Assuming X is forward
                # Non-linear reward: higher reward for maintaining a good walking pace
                # Too slow or too fast are both suboptimal
                target_vel = 1.0  # Target velocity (m/s)
                vel_error = (forward_vel - target_vel) ** 2
                forward_reward = np.exp(-forward_scale * vel_error)
                # Add bonus for positive forward movement
                if forward_vel > 0:
                    forward_reward += 0.2 * forward_vel
            else:
                forward_reward = 0.0

            self.reward_components['forward'] = w_forward * forward_reward
            total_reward += self.reward_components['forward']
        except Exception as e:
            print(f"Error calculating forward reward: {e}")
            self.reward_components['forward'] = 0.0

        # 8. Alive Bonus (Constant reward for not falling)
        self.reward_components['alive'] = alive_bonus
        total_reward += alive_bonus

        # Apply dynamic reward scaling based on performance
        # If the model is doing well already, make rewards more targeted
        if total_reward > 0.75:  
            # Increase forward progress weight as performance improves
            total_reward = 0.7 * total_reward + 0.3 * self.reward_components['forward']

        return total_reward

    def step(self, action):
        """
        Override the step method to include reward components in the info dict.
        """
        # Call parent step method
        obs, reward, done, info = super().step(action)
        
        # Add reward components to info
        if hasattr(self, 'reward_components'):
            info['reward_components'] = self.reward_components
            
        return obs, reward, done, info

# Custom wrapper to handle the gym 0.26.2 API with stable-baselines3 1.2.0
class GymAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Adapt observation space if needed (example: Dict -> Box)
        if isinstance(env.observation_space, spaces.Dict):
            # Flatten the Dict space into a Box space
            # This requires knowing the keys and shapes within the Dict
            # Example: Assuming 'obs' and 'achieved_goal' keys
            # Adjust based on your actual Dict structure
            low = np.concatenate([env.observation_space['obs'].low, env.observation_space['achieved_goal'].low])
            high = np.concatenate([env.observation_space['obs'].high, env.observation_space['achieved_goal'].high])
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

            # Store the original observation keys for processing
            self._obs_keys = list(env.observation_space.spaces.keys())
        else:
            # Keep the original space if it's not a Dict
            self.observation_space = env.observation_space

        self.action_space = env.action_space

    def step(self, action):
        # Call the environment's step method
        # Handle potential difference in return values (4 vs 5)
        try:
            step_result = self.env.step(action)
            
            if len(step_result) == 5:  # If step returns obs, reward, done, truncated, info
                obs, reward, done, truncated, info = step_result
                
                # Add reward components if available in the base environment
                if hasattr(self.env, 'reward_components'):
                    info['reward_components'] = self.env.reward_components
                
                # Store true reward (before potential normalization in wrapper)
                info['episode_real_reward'] = reward
                
                # Convert the observation if it's a dictionary
                if isinstance(obs, dict):
                    obs = self._flatten_obs(obs)
                
                return obs, reward, done, truncated, info
            else:  # Traditional 4-element return
                obs, reward, done, info = step_result
                
                # Add reward components if available in the base environment
                if hasattr(self.env, 'reward_components'):
                    info['reward_components'] = self.env.reward_components
                
                # Store true reward (before potential normalization in wrapper)
                info['episode_real_reward'] = reward
                
                # Convert the observation if it's a dictionary
                if isinstance(obs, dict):
                    obs = self._flatten_obs(obs)
                
                return obs, reward, done, info
        except Exception as e:
            print(f"Error in GymAdapter.step: {e}")
            # Return safe defaults in case of error
            zero_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return zero_obs, 0.0, True, {'error': str(e)}

    def reset(self, **kwargs):
        # Call the environment's reset method
        # Handle potential difference in return values (obs vs obs, info)
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result # Newer gym API
        else:
            obs = reset_result # Older gym API or base env specific
            info = {} # Provide an empty info dict

        # Process observation if it was a Dict
        if isinstance(self.env.observation_space, spaces.Dict):
            obs = self._flatten_obs(obs)

        # Return observation (and info if needed, though SB3 v1.x primarily uses obs)
        # Return 1 item as expected by older reset API used by SB3 v1.x
        # If using SB3 v2+, return (obs, info)
        return obs # Return 1 item

    def _flatten_obs(self, obs_dict):
        # Flatten the dictionary observation into a single numpy array
        # Ensure the order matches the order used to define the Box space
        return np.concatenate([obs_dict[key] for key in self._obs_keys])

    # Forward other methods if necessary
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        # Pass the seed call to the underlying environment
        # Adjust based on how the base env handles seeding
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        else:
            # Handle cases where the base env might use np_random directly
            # This might involve setting env.np_random.seed(seed)
            print("Warning: Base environment might not have a seed method. Seeding might not be fully applied.")
            pass # Or implement more specific seeding logic if needed