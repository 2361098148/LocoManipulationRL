from abc import abstractmethod
# from tasks.base.rl_task import RLTask
import torch

from utils.math import transform_vectors, inverse_transform_vectors, rand_quaternions

from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_conjugate, quat_mul

class UnitedQuadrupedPoseControl():

    # Robot configs
    ## Observation
    ground_position_scale = 5
    ground_position_scale = 5
    ground_quaternion_scale = 1
    ground_linear_vel_scale = 2
    ground_angular_vel_scale = 0.25
    base_tip_position_scale = 1
    joint_position_scale = 0.3
    joint_velocity_scale = 0.3

    clip_reward = False
    ## Task related
    min_roll = -0.4
    max_roll = 0.4 
    min_pitch = -0.4
    max_pitch = 0.4 
    min_yaw = -3.14
    max_yaw = 3.14 

    # Reward scales
    ## Orientation scales
    quaternion_scale = 1
    rot_eps = 0.1
    ## Translation scale
    translation_scale = -2.5
    ## Joint movement scale
    joint_acc_scale = -0.001 # max about 0.5 for each dimension
    ## Action rate scale
    action_rate_scale = -0.03

    ## Reset threshold
    baseline_knee_height = 0.04
    fall_penalty = -200
    baseline_height = 0.05
    baseline_corner_height = 0.01

    ## Goal threshold
    success_thresh = 0.1
    success_bonus = 5 # per residual step
    max_consecutive_successes = 30

    ## Joint limit penalty
    joint_limit_penalty = -5
    min_joint_23_diff = 0.43 # 25 degree
    max_joint_23_diff = 2.53 # 145 degree
    reset_min_joint_23_diff = 0.384 # 22 degree
    reset_max_joint_23_diff = 2.61 # 150 degree
    min_joint_1_pos = -2.35 # For a1, 135 degree
    max_joint_1_pos = 0.78 # For a1, 45 degree
    reset_min_joint_1_pos = -2.44 # For a1, 140 degree
    reset_max_joint_1_pos = 0.87 # For a1, 50 degree

    def __init__(self) -> None:
        self.robot = self.quadruped_robot

        self._num_actions = 12
        self._num_observations = 57
        self._num_states = 57

        # super().__init__(name, env, offset)

        self.last_actions = torch.zeros((self._num_envs, self._num_actions), dtype=torch.float32, device=self._device)
        self.current_actions = torch.zeros((self._num_envs, self._num_actions), dtype=torch.float32, device=self._device)

        self.last_base_tip_positions = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.default_base_tip_positions = torch.tensor([[-0.0937,  0.1223, -0.1774],
            [ 0.0937,  0.1408, -0.1773],
            [-0.0937, -0.1408, -0.1773],
            [ 0.0937, -0.1223, -0.1774]], dtype=torch.float32, device=self._device).repeat((self._num_envs, 1, 1))

        self.__corner_pos_robot = torch.cat((
            torch.tensor([0.075, 0.1835, -0.04]).repeat(self._num_envs, 1),
            torch.tensor([-0.075, 0.1835, -0.04]).repeat(self._num_envs, 1),
            torch.tensor([0.075, -0.1835, -0.04]).repeat(self._num_envs, 1),
            torch.tensor([-0.075, -0.1835, -0.04]).repeat(self._num_envs, 1)
        ), dim=-1).view(self._num_envs, 4, 3).to(torch.float32).to(self._device)

        self.goal_quaternions = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        self.successes = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self.consecutive_successes = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self.goal_reset_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # Metrics
        self.max_reset_counts = torch.tensor(2048, dtype=torch.long, device=self._device)
        self.num_successes = torch.tensor(0, dtype=torch.long, device=self._device)
        self.num_resets = torch.tensor(0, dtype=torch.long, device=self._device)
        self.success_rate = torch.tensor(0.0, dtype=torch.float32, device=self._device)

    def pre_physics_step(self, actions):
        actions = actions.to(self._device)
        self.current_actions = actions.clone()

        # Check if the environment needes to be reset
        reset_buf = self.reset_buf.clone()
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # actions[:] = 0.0
        self.robot.take_action(actions)

    def reset_idx(self, env_ids):
        num_reset = len(env_ids)

        # Get the ids where the goal reached
        goal_reset_ids = self.goal_reset_buf.nonzero().flatten()

        # Reset base position if needed
        position_reset_indices = (self.progress_buf[env_ids] < self._max_episode_length - 1).nonzero().flatten()
        position_reset_ids = env_ids[position_reset_indices]

        # Get rid of goal_reset_indices
        if len(goal_reset_ids) > 0:
            position_goal_sumup = torch.cat((
                position_reset_ids,
                goal_reset_ids
            ), dim=0)
            # Find ids ends early but not because of goal reached
            unique_ids, counts = torch.unique(position_goal_sumup, return_counts=True)
            repeat_once_indices = (counts == 1).nonzero()
            position_reset_ids = unique_ids[repeat_once_indices].flatten()

        # Reset robot base positions and rotations
        num_pos_reset = len(position_reset_ids)
        if num_pos_reset > 0:
            # Reset joint positions/velocities to default
            self.reset_robot_joint_positions(position_reset_ids)
            self.reset_robot_joint_velocities(position_reset_ids)
            self.reset_robot_poses(position_reset_ids)
            self.reset_robot_velocities(position_reset_ids)

            # Reset last base tip positions
            self.last_base_tip_positions[position_reset_ids] = self.default_base_tip_positions[position_reset_ids]

            # Reset history buffers for env_ids
            self.clear_robot_last_joint_velocities(position_reset_ids)
            self.last_actions[position_reset_ids, :] = torch.zeros(num_pos_reset, self._num_actions, dtype=torch.float32, device=self._device)

        # Reset goal orientation
        new_rand_quaternions = rand_quaternions(num_reset, 
                                                self.min_roll,
                                                self.max_roll,
                                                self.min_pitch,
                                                self.max_pitch,
                                                self.min_yaw,
                                                self.max_yaw,
                                                self._device)
        
        # The goal quaternions will be the inverse quaternion of robot base
        self.goal_quaternions[env_ids, :] = quat_conjugate(new_rand_quaternions)

        self.reset_goal_indicator_pose(new_rand_quaternions, env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # Reset buffers
        self.successes[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)
        self.consecutive_successes[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)
        self.goal_reset_buf[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)

    def get_observations(self):
        self.robot.update_all_states()

        self.joint_positions = self.quadruped_robot.joint_positions
        self.joint_velocities = self.quadruped_robot.joint_velocities
        self.joint_accelerations = self.quadruped_robot.joint_accelerations

        self.base_quaternions = self.quadruped_robot.base_quaternions
        self.base_positions = self.quadruped_robot.base_positions
        self.base_linear_velocities = self.quadruped_robot.base_linear_velocities
        self.base_angular_velocities = self.quadruped_robot.base_angular_velocities

        self.tip_positions = self.quadruped_robot.tip_positions
        self.knee_positions = self.quadruped_robot.knee_positions

        # Calculate the position of the virtual moving obj
        self.ground_positions = quat_rotate_inverse(self.base_quaternions, -self.base_positions)
        scaled_ground_positions = self.ground_position_scale * self.ground_positions

        # Calculate the quaternion of the virtual moving obj
        self.ground_quaternion = quat_conjugate(self.base_quaternions)
        scaled_ground_quaternion = self.ground_quaternion_scale * self.ground_quaternion
        scaled_goal_ground_quaternion = self.ground_quaternion_scale * self.goal_quaternions

        # Calculate the linear velocity of the virtual moving obj
        self.ground_linear_vel = quat_rotate_inverse(self.base_quaternions, -self.base_linear_velocities)
        scaled_ground_linear_vel = self.ground_linear_vel_scale * self.ground_linear_vel

        # Calculate the angular velocity of the virtual moving obj
        self.ground_angular_vel = quat_rotate_inverse(self.base_quaternions, -self.base_angular_velocities)
        scaled_ground_angular_vel = self.ground_angular_vel_scale * self.ground_angular_vel
        
        # Calculate the tip positions in robot base frame
        self.base_tip_positions = inverse_transform_vectors(self.base_quaternions, self.base_positions, self.tip_positions, device=self._device)
        flat_base_tip_positions = self.base_tip_positions.reshape(self._num_envs, 12)
        scaled_base_tip_positions = self.base_tip_position_scale * flat_base_tip_positions
        flat_last_base_tip_positions = self.last_base_tip_positions.reshape(self._num_envs, 12)
        scaled_last_base_tip_positions = self.base_tip_position_scale * flat_last_base_tip_positions

        scaled_joint_positions = self.joint_position_scale * self.joint_positions
        scaled_joint_velocities = self.joint_velocity_scale * self.joint_velocities

        quat_diff = quat_mul(self.ground_quaternion, quat_conjugate(self.goal_quaternions))
        # self.obs_buf = torch.cat((
        #     scaled_ground_positions,
        #     scaled_ground_quaternion,
        #     scaled_goal_ground_quaternion,
        #     scaled_ground_linear_vel,
        #     scaled_ground_angular_vel,
        #     scaled_base_tip_positions,
        #     scaled_last_base_tip_positions,
        #     scaled_joint_positions,
        #     scaled_joint_velocities,
        #     self.current_actions,
        #     self.last_actions
        # ), dim=-1)

        self.obs_buf = torch.cat((
            scaled_ground_positions,
            scaled_ground_linear_vel,
            scaled_ground_quaternion,
            scaled_ground_angular_vel,

            scaled_joint_positions,
            scaled_joint_velocities,

            scaled_goal_ground_quaternion,
            quat_diff,

            self.current_actions
        ), dim=-1)

        self.states_buf = self.obs_buf

        # Update last tip positions
        self.last_base_tip_positions = self.base_tip_positions

    def calculate_metrics(self) -> None:
        # Calculate base x and y locations
        base_x = self.base_positions[:, 0]
        base_y = self.base_positions[:, 1]

        '''Pose error penalty'''
        quat_diff = quat_mul(self.ground_quaternion, quat_conjugate(self.goal_quaternions))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)) # changed quat convention
        rot_rew = 1.0/(torch.abs(rot_dist) + self.rot_eps) * self.quaternion_scale

        '''Translation penalty'''
        # Punish any translation from origin in xy axis
        sum_xy = torch.abs(base_x) + torch.abs(base_y)
        translation_penalty = sum_xy * self.translation_scale

        '''joint acceleration penalty'''
        joint_acc_penalty = torch.sum(torch.abs(self.joint_accelerations)*self.joint_acc_scale, dim=1)

        '''action rate penalty'''
        action_rate_penalty = torch.sum(torch.abs(self.last_actions - self.current_actions), dim=1)*self.action_rate_scale

        '''consecutive successes bonus reward'''
        consecutive_goal_reset = torch.where(self.consecutive_successes > self.max_consecutive_successes, 
                                                torch.ones_like(self.consecutive_successes), 
                                                torch.zeros_like(self.consecutive_successes))
        # consecutive_successes_rew = self._success_bonus * consecutive_goal_reset.to(torch.float32)
        # Give bonus to every left steps; earlier success gets higher reward
        rest_episode_steps = (self._max_episode_length - self.progress_buf).to(torch.float32)
        per_step_rew = self.success_bonus * consecutive_goal_reset.to(torch.float32)
        consecutive_successes_rew = torch.mul(per_step_rew, rest_episode_steps)
        
        # Check if success
        successes = torch.where(torch.abs(rot_dist) <= self.success_thresh, 
                            torch.ones_like(consecutive_goal_reset), 
                            torch.zeros_like(consecutive_goal_reset))

        '''joint limit penalty'''
        a14_dof1_positions = self.joint_positions[:, [0,3]]
        a23_dof1_positions = self.joint_positions[:, [1,2]]
        dof2_positions = self.joint_positions[:, [4,6,8,10]]
        dof3_positions = self.joint_positions[:, [5,7,9,11]]
        # Determines if position difference between DoF2&3 breaks the limit
        joint23_pos_diffs = torch.abs(dof3_positions - dof2_positions)
        joint23_pos_diff_too_low = joint23_pos_diffs < self.min_joint_23_diff
        joint23_pos_diff_too_high = joint23_pos_diffs > self.max_joint_23_diff
        joint23_pos_break_limit_sumup = joint23_pos_diff_too_low + joint23_pos_diff_too_high # shape (N, 4)
        joint23_pos_break_limit = torch.sum(joint23_pos_break_limit_sumup, dim=-1) # Shape (N, 1)
        # Check if break the reset limit
        joint23_pos_diff_low_reset = joint23_pos_diffs < self.reset_min_joint_23_diff
        joint23_pos_diff_high_reset = joint23_pos_diffs > self.reset_max_joint_23_diff
        joint23_pos_reset_sumup = joint23_pos_diff_low_reset + joint23_pos_diff_high_reset # shape (N, 4)
        self.joint23_pos_reset = torch.sum(joint23_pos_reset_sumup, dim=-1) # Shape (N, 1)
        # Determines if position of DoF 1 breaks the limit
        a14_dof1_min_pos = self.min_joint_1_pos
        a14_dof1_max_pos = self.max_joint_1_pos
        a23_dof1_min_pos = -self.max_joint_1_pos
        a23_dof1_max_pos = -self.min_joint_1_pos
        a14_dof_pos_too_low = a14_dof1_positions < a14_dof1_min_pos
        a14_dof_pos_too_high = a14_dof1_positions > a14_dof1_max_pos
        a23_dof_pos_too_low = a23_dof1_positions < a23_dof1_min_pos
        a23_dof_pos_too_high = a23_dof1_positions > a23_dof1_max_pos
        joint1_pos_break_limit_sumup = a14_dof_pos_too_low + a14_dof_pos_too_high + \
                                 a23_dof_pos_too_low + a23_dof_pos_too_high # Shape (N, 2)
        joint1_pos_break_limit = torch.sum(joint1_pos_break_limit_sumup, dim=-1) # Shape (N, 1)
        # Check if DoF 1 break the reset limit
        a14_dof1_reset_min_pos = self.reset_min_joint_1_pos
        a14_dof1_reset_max_pos = self.reset_max_joint_1_pos
        a23_dof1_reset_min_pos = -self.reset_max_joint_1_pos
        a23_dof1_reset_max_pos = -self.reset_min_joint_1_pos
        a14_dof_pos_low_reset = a14_dof1_positions < a14_dof1_reset_min_pos
        a14_dof_pos_high_reset = a14_dof1_positions > a14_dof1_reset_max_pos
        a23_dof_pos_low_reset = a23_dof1_positions < a23_dof1_reset_min_pos
        a23_dof_pos_high_reset = a23_dof1_positions > a23_dof1_reset_max_pos
        joint1_pos_reset_sumup = a14_dof_pos_low_reset + a14_dof_pos_high_reset + \
                                 a23_dof_pos_low_reset + a23_dof_pos_high_reset # Shape (N, 2)
        self.joint1_pos_reset = torch.sum(joint1_pos_reset_sumup, dim=-1) # Shape (N, 1)

        joint_break_limit_num = (joint1_pos_break_limit + joint23_pos_break_limit).flatten() # Shape (N)
        joint_break_limit = torch.where(joint_break_limit_num > 0, torch.ones_like(joint_break_limit_num), torch.zeros_like(joint_break_limit_num))
        joint_limit_penalty = (self.joint_limit_penalty * joint_break_limit).to(torch.float32)

        total_rew = rot_rew + \
                    translation_penalty + \
                    joint_acc_penalty + \
                    action_rate_penalty + \
                    consecutive_successes_rew + \
                    joint_limit_penalty

        if self.clip_reward:
            total_rew = torch.clip(total_rew, 0.0, None)

        log_dict = {}
        log_dict['orientation_rew'] = rot_rew
        log_dict['translation_penalty'] = translation_penalty
        log_dict['joint_acc_penalty'] = joint_acc_penalty
        log_dict['action_rate_penalty'] = action_rate_penalty
        log_dict['consecutive_successes_rew'] = consecutive_successes_rew
        log_dict["joint_limit_panelty"] = joint_limit_penalty

        # Update rew_buf, goal_reset_buf
        self.rew_buf[:] = total_rew
        self.goal_reset_buf = consecutive_goal_reset

        # Check if consecutive success
        ## 1 if and only current success and last success is 1; otherwise 0
        this_consecutive_success = torch.logical_and(successes, self.successes)
        # Reset consecutive success buf if 0 
        self.consecutive_successes[:] = torch.where(this_consecutive_success == 0, torch.zeros_like(self.consecutive_successes), self.consecutive_successes)
        # Add one success to consecutive success buf if 1 (both last and current step are successful)
        self.consecutive_successes[:] = torch.where(this_consecutive_success == 1, self.consecutive_successes+1, self.consecutive_successes)
        # If last not successful, but current successful, set buf to 1 (first success)
        last_fail = torch.where(self.successes == 0, torch.ones_like(self.successes), torch.zeros_like(self.successes))
        current_success = torch.where(successes == 1, torch.ones_like(successes), torch.zeros_like(successes))
        last_fail_current_success = torch.logical_and(last_fail, current_success)
        self.consecutive_successes[:] = torch.where(last_fail_current_success == 1, torch.ones_like(self.consecutive_successes), self.consecutive_successes)
        # Update successes buf
        self.successes[:] = successes

        # Update last actions
        self.last_actions = self.current_actions.clone()

        _rew_is_nan = torch.isnan(self.rew_buf)
        _rew_nan_sum = torch.sum(_rew_is_nan)
        if _rew_nan_sum != torch.tensor(0.0):
            print("NaN Value found in reward")
            print(self.rew_buf)

        rew_below_thresh = self.rew_buf < -50.0
        is_any_rew_neg = rew_below_thresh.nonzero()
        assert len(is_any_rew_neg) == 0

        self.extras.update({"env/rewards/"+k: v.mean() for k, v in log_dict.items()})

    def is_done(self):
        # Fall down reset
        body_heights = torch.flatten(self.base_positions[:, 2])
        self.reset_buf[:] = torch.where(body_heights <= self.baseline_height, torch.ones_like(self.reset_buf), self.reset_buf)

        # Corner fall down reset
        corner_heights =  self._get_corner_heights()
        corner_below_baseline = (corner_heights < self.baseline_corner_height)
        corner_below_baseline_sumup = torch.sum(corner_below_baseline, dim=-1)
        self.reset_buf[:] = torch.where(corner_below_baseline_sumup > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        # Knee touching ground reset; reset if anyone of the knees touches the ground
        knee_positions = self.knee_positions
        knee_z = knee_positions[:, :, 2]
        knee_z_below_baseline = knee_z - self.baseline_knee_height <= 0
        knee_z_below_sum = torch.sum(knee_z_below_baseline, dim=1)
        self.reset_buf[:] = torch.where(knee_z_below_sum > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        # Tip position larger than allowed (a circle of 0.25m radius)
        ## Approximately determine if tip leaves the ground
        tip_z_distance = self.tip_positions[:, :, 2].view(self._num_envs, 4)
        tip_z_above_thresh = tip_z_distance < 0.05
        tip_xy_distance = self.tip_positions[:, :, 0:2]
        tip_planar_distance = torch.norm(tip_xy_distance, dim=-1).view(self._num_envs, 4)
        tip_planar_above_thresh = tip_planar_distance > 0.25
        tip_z_and_planar_above_thresh = torch.logical_and(tip_z_above_thresh, tip_planar_above_thresh)
        tip_pos_reset_sumup = torch.sum(tip_z_and_planar_above_thresh, dim=-1)
        self.reset_buf[:] = torch.where(tip_pos_reset_sumup > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        # Joint limit reset
        joint_reset_num = self.joint1_pos_reset + self.joint23_pos_reset
        self.reset_buf[:] = torch.where(joint_reset_num > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        # Add falling penalty
        fall_penalty = (self.fall_penalty * self.reset_buf).to(torch.float32)
        self.rew_buf += fall_penalty
        self.extras.update({"env/rewards/fall_penalty": fall_penalty.mean()})

        # Goal reset
        self.reset_buf[:] = torch.where(self.goal_reset_buf == 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # Check if maximum progress reached
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        
        # Calculate the success rate
        if self.num_resets > self.max_reset_counts:
            # Update the success rate
            if self.num_resets != 0:
                self.success_rate = self.num_successes/self.num_resets
            else:
                self.success_rate = torch.tensor(0.0)

            # Zero reset counts periodically
            self.num_resets = torch.tensor(0, dtype=torch.long, device=self._device)
            self.num_successes = torch.tensor(0, dtype=torch.long, device=self._device)

        self.num_successes += torch.sum(self.goal_reset_buf)
        self.num_resets += torch.sum(self.reset_buf)

        self.extras.update({"env/success_rate": self.success_rate})

    def _get_corner_heights(self):
        world_corner_positions = transform_vectors(self.base_quaternions,
                                                   self.base_positions,
                                                   self.__corner_pos_robot,
                                                   self._device)

        corner_heights = world_corner_positions[:, :, 2].view(self._num_envs, 4)
        
        return corner_heights

    @abstractmethod
    def clear_robot_last_joint_velocities(self, indices):
        raise NotImplementedError

    @abstractmethod
    def reset_robot_joint_positions(self, indices):
        raise NotImplementedError

    @abstractmethod
    def reset_robot_joint_velocities(self, indices):
        raise NotImplementedError

    @abstractmethod
    def reset_robot_poses(self, indices):
        raise NotImplementedError

    @abstractmethod
    def reset_robot_velocities(self, indices):
        raise NotImplementedError

    @abstractmethod
    def reset_goal_indicator_pose(self, goal_quaternions, indices):
        pass

    # @property
    # def _max_episode_length(self):
    #     return None

    @property
    def quadruped_robot(self):
        """Implement this method with return the robot"""
        raise NotImplementedError