import torch
from abc import abstractmethod

from omni.isaac.core.utils.torch.rotations import quat_from_euler_xyz, quat_rotate_inverse, quat_conjugate, quat_mul

from utils.math import rand_quaternions, inverse_transform_vectors, inverse_rotate_orientations, transform_vectors

class UnitedQuadrupedManipulatePlateInverted:
    # Obs scale
    plate_position_scale = 5
    plate_quaternion_scale = 1
    plate_linear_vel_scale = 2
    plate_angular_vel_scale = 0.25
    base_tip_position_scale = 1
    joint_position_scale = 0.3
    joint_velocity_scale = 0.3

    # Common reward/penalty
    clip_reward = False
    rot_eps = 0.1
    translation_scale = -2.5
    joint_acc_scale = -0.001
    action_rate_scale = -0.03
    fall_penalty = -200
    baseline_height = 0.05
    baseline_knee_height = 0.04
    baseline_corner_height = 0.01
    joint_limit_penalty = -5
    min_joint_23_diff = 0.43 # 25 degree
    max_joint_23_diff = 2.53 # 145 degree
    reset_min_joint_23_diff = 0.384 # 22 degree
    reset_max_joint_23_diff = 2.61 # 150 degree
    min_joint_1_pos = -2.35 # For a1
    max_joint_1_pos = 0.78 # For a1
    reset_min_joint_1_pos = -2.44 # For a1, 140 degree
    reset_max_joint_1_pos = 0.87 # For a1, 50 degree

    curriculum1 = {
        # Task Completion Rewards
        "success_tolerance": 0.15,
        "success_duration": 15,
        "success_reward": 6,
        "completion_reward": 5.0,
        
        # Orientation rewards
        "k_orientation_reward": 2.0,

        # Goal orientation ranges
        "min_roll": -0.25,
        "max_roll": 0.25,
        "min_pitch": -0.25,
        "max_pitch": 0.25,
        "min_yaw": -1.57,
        "max_yaw": 1.57,
    }

    curriculum2 = {
        # Task Completion Rewards
        "success_tolerance": 0.1,
        "success_duration": 30,
        "success_reward": 3,
        "completion_reward": 5.0,
        
        # Orientation rewards
        "k_orientation_reward": 1.0,

        # Goal orientation ranges
        "min_roll": -0.4,
        "max_roll": 0.4,
        "min_pitch": -0.4,
        "max_pitch": 0.4,
        "min_yaw": -3.14,
        "max_yaw": 3.14,
    }

    curriculum3 = {
        # Task Completion Rewards
        "success_tolerance": 0.1,
        "success_duration": 30,
        "success_reward": 0,
        "completion_reward": 5.0,
        
        # Orientation rewards
        "k_orientation_reward": 1.5,

        # Goal orientation ranges
        "min_roll": -0.4,
        "max_roll": 0.4,
        "min_pitch": -0.4,
        "max_pitch": 0.4,
        "min_yaw": -3.14,
        "max_yaw": 3.14,
    }

    def __init__(self) -> None:
        self.robot = self.quadruped_robot

        self._num_actions = 12
        self._num_observations = 89
        self._num_states = 89

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

        self.__corner_pos_world = transform_vectors(self.fixed_robot_base_quaternions,
                                                    self.fixed_robot_base_positions,
                                                    self.__corner_pos_robot,
                                                    self._device)

        self.goal_quaternions = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        self.successes = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self.consecutive_successes = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self.goal_reset_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # Metrics
        self.max_reset_counts = torch.tensor(2048, dtype=torch.long, device=self._device)
        self.num_successes = torch.tensor(0, dtype=torch.long, device=self._device)
        self.num_resets = torch.tensor(0, dtype=torch.long, device=self._device)
        self.success_rate = torch.tensor(0.0, dtype=torch.float32, device=self._device)

        self.curriculums = [self.curriculum1, self.curriculum2, self.curriculum3]
        self.current_curriculum_id = 1

    def pre_physics_step(self, actions):
        actions = actions.to(self._device)
        self.current_actions = actions.clone()

        # Check if the environment needes to be reset
        reset_buf = self.reset_buf.clone()
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.robot.take_action(actions)

    def reset_idx(self, env_ids):
        num_reset = len(env_ids)

        # Get the ids where the goal reached
        not_goal_reset_buf = torch.logical_not(self.goal_reset_buf)

        # Reset the robot and plate pose if it is in env_ids
        # but not in goal_reset_ids
        position_reset_buf = torch.logical_and(not_goal_reset_buf, self.reset_buf)
        position_reset_ids = position_reset_buf.nonzero().flatten()

        num_pos_reset = len(position_reset_ids)
        if num_pos_reset > 0:
            # Reset joint positions/velocities to default
            self.reset_robot_joint_positions(position_reset_ids)
            self.reset_robot_joint_velocities(position_reset_ids)

            # Reset plate poses
            self.reset_plate_poses(position_reset_ids)
            self.reset_plate_velocities(position_reset_ids)

            # Reset last base tip positions
            self.last_base_tip_positions[position_reset_ids] = self.default_base_tip_positions[position_reset_ids]

            # Reset history buffers for env_ids
            self.clear_robot_last_joint_velocities(position_reset_ids)
            self.last_actions[position_reset_ids, :] = torch.zeros(num_pos_reset, self._num_actions, dtype=torch.float32, device=self._device)

        # Reset goal orientation
        min_roll = self.current_curriculum["min_roll"]
        max_roll = self.current_curriculum["max_roll"]
        min_pitch = self.current_curriculum["min_pitch"]
        max_pitch = self.current_curriculum["max_pitch"]
        min_yaw = self.current_curriculum["min_yaw"]
        max_yaw = self.current_curriculum["max_yaw"]
        new_rand_quaternions = rand_quaternions(num_reset, 
                                                min_roll,
                                                max_roll,
                                                min_pitch,
                                                max_pitch,
                                                min_yaw,
                                                max_yaw,
                                                self._device)
        
        self.goal_quaternions[env_ids, :] = new_rand_quaternions
        self.reset_goal_indicator_pose(new_rand_quaternions, env_ids)
        
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # Reset buffers
        self.successes[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)
        self.consecutive_successes[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)
        self.goal_reset_buf[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)

    def get_observations(self):
        self.robot.update_joint_states()
        self.robot.update_tip_states()
        self.robot.update_knee_positions()

        self.joint_positions = self.quadruped_robot.joint_positions
        self.joint_velocities = self.quadruped_robot.joint_velocities
        self.joint_accelerations = self.quadruped_robot.joint_accelerations
        self.tip_positions = self.quadruped_robot.tip_positions
        self.knee_positions = self.quadruped_robot.knee_positions

        # Get plate position and quaternion relative to the ground
        self.plate_pos_ground, self.plate_quat_ground = self.plate.get_object_poses()
        # Plate positions relative to the robot base
        self.plate_pos_robot = inverse_transform_vectors(self.fixed_robot_base_quaternions, 
                                                         self.fixed_robot_base_positions, 
                                                         self.plate_pos_ground.view(self._num_envs, 1, 3),
                                                         device=self._device).view(self._num_envs, 3)
        scaled_plate_pos_robot = self.plate_position_scale * self.plate_pos_robot
        # Plate quaternions relative to the robot base
        self.plate_quat_robot = inverse_rotate_orientations(self.fixed_robot_base_quaternions, self.plate_quat_ground, device=self._device)
        scaled_plate_quat_robot = self.plate_quaternion_scale * self.plate_quat_robot

        # Goal plate quaternions
        scaled_goal_plate_quat_robot = self.plate_quaternion_scale * self.goal_quaternions

        # Calculate plate linear&angular velocities under the robot base frame
        plate_linear_vel_ground, plate_angular_vel_ground = self.plate.get_object_velocities()
        self.plate_linear_vel_robot = quat_rotate_inverse(self.fixed_robot_base_quaternions, plate_linear_vel_ground)
        self.plate_angular_vel_robot = quat_rotate_inverse(self.fixed_robot_base_quaternions, plate_angular_vel_ground)
        scaled_plate_linear_vel_robot = self.plate_linear_vel_scale * self.plate_linear_vel_robot
        scaled_plate_angular_vel_robot = self.plate_angular_vel_scale * self.plate_angular_vel_robot

        # Get tip positions relative to the robot base
        self.base_tip_positions = inverse_transform_vectors(self.fixed_robot_base_quaternions, self.fixed_robot_base_positions, self.tip_positions, device=self._device)
        flat_base_tip_positions = self.base_tip_positions.reshape(self._num_envs, self.robot.num_modules*3)
        flat_last_base_tip_positions = self.last_base_tip_positions.reshape(self._num_envs, self.robot.num_modules*3)
        scaled_base_tip_positions = self.base_tip_position_scale * flat_base_tip_positions
        scaled_last_base_tip_positions = self.base_tip_position_scale * flat_last_base_tip_positions

        # joint positions
        scaled_joint_positions = self.joint_position_scale * self.joint_positions

        # joint velocities
        scaled_joint_velocities = self.joint_velocity_scale * self.joint_velocities

        self.obs_buf = torch.cat((
            scaled_plate_pos_robot,
            scaled_plate_quat_robot,
            scaled_goal_plate_quat_robot,
            scaled_plate_linear_vel_robot,
            scaled_plate_angular_vel_robot,
            scaled_base_tip_positions,
            scaled_last_base_tip_positions,
            scaled_joint_positions,
            scaled_joint_velocities,
            self.current_actions,
            self.last_actions
        ), dim=-1)

        self.states_buf = self.obs_buf

        # Update last tip positions
        self.last_base_tip_positions = self.base_tip_positions

    def calculate_metrics(self) -> None:
        # Get rewards/scales
        # Task completion rewards
        success_tolerance = self.current_curriculum["success_tolerance"]
        success_duration = self.current_curriculum["success_duration"]
        success_reward = self.current_curriculum["success_reward"]
        completion_reward = self.current_curriculum["completion_reward"]
        # Orientaion rewards
        k_orien_reward = self.current_curriculum["k_orientation_reward"]

        '''Pose error penalty'''
        quat_diff = quat_mul(self.plate_quat_robot, quat_conjugate(self.goal_quaternions))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)) # changed quat convention
        rot_rew = 1.0/(torch.abs(rot_dist) + self.rot_eps) * k_orien_reward

        '''Translation penalty'''
        # Punish any translation from origin in xy axis
        base_x = self.plate_pos_robot[:, 0]
        base_y = self.plate_pos_robot[:, 1]
        sum_xy = torch.abs(base_x) + torch.abs(base_y)
        translation_penalty = sum_xy * self.translation_scale

        '''joint velocity and joint acceleration penalty'''
        joint_acc_penalty = torch.sum(torch.abs(self.joint_accelerations)*self.joint_acc_scale, dim=1)
        
        '''action rate penalty'''
        action_rate_penalty = torch.sum(torch.abs(self.last_actions - self.current_actions), dim=1)*self.action_rate_scale

        '''consecutive successes bonus reward'''
        consecutive_goal_reset = torch.where(self.consecutive_successes > success_duration, 
                                                torch.ones_like(self.consecutive_successes), 
                                                torch.zeros_like(self.consecutive_successes))
        # consecutive_successes_rew = self._success_bonus * consecutive_goal_reset.to(torch.float32)
        # Give bonus to every left steps; earlier success gets higher reward
        rest_episode_steps = (self._max_episode_length - self.progress_buf).to(torch.float32)
        per_step_rew = completion_reward * consecutive_goal_reset.to(torch.float32)
        if self.current_curriculum_id != 0:
            # No consecutive success reward for phase1
            consecutive_successes_rew = torch.mul(per_step_rew, rest_episode_steps).to(self._device)
        else:
            consecutive_successes_rew = torch.zeros_like(self.reset_buf, dtype=torch.float32, device=self._device)
        
        '''success reward'''
        successes = torch.where(torch.abs(rot_dist) <= success_tolerance, 
                            torch.ones_like(consecutive_goal_reset), 
                            torch.zeros_like(consecutive_goal_reset))
        non_consecut_success_rew = (success_reward * successes).to(torch.float32).to(self._device)

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
                    joint_acc_penalty + \
                    action_rate_penalty + \
                    consecutive_successes_rew + \
                    non_consecut_success_rew + \
                    joint_limit_penalty + \
                    translation_penalty

        if self.clip_reward:
            total_rew = torch.clip(total_rew, 0.0, None)

        log_dict = {}
        log_dict['orientation_rew'] = rot_rew
        log_dict['joint_acc_penalty'] = joint_acc_penalty
        log_dict['action_rate_penalty'] = action_rate_penalty
        log_dict['consecutive_successes_rew'] = consecutive_successes_rew
        log_dict['non_consecutive_success_rew'] = non_consecut_success_rew
        log_dict['joint_limit_penalty'] = joint_limit_penalty
        log_dict['translation_penalty'] = translation_penalty

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
        assert _rew_nan_sum == torch.tensor(0.0)

        rew_below_thresh = self.rew_buf < -50.0
        is_any_rew_neg = rew_below_thresh.nonzero()
        assert len(is_any_rew_neg) == 0

        self.extras.update({"env/rewards/"+k: v.mean() for k, v in log_dict.items()})

    def is_done(self):
        # Plate down reset
        plate_z_pos_robot = self.plate_pos_robot[:, 2]
        self.reset_buf[:] = torch.where(plate_z_pos_robot > 0.0, torch.ones_like(self.reset_buf), self.reset_buf)
        
        # Body down reset
        ## Get the body position relative to the plate frame
        body_position_plate = inverse_transform_vectors(self.plate_quat_ground,
                                                        self.plate_pos_ground,
                                                        self.fixed_robot_base_positions.view(self._num_envs,1,3),
                                                        self._device)
        body_heights = body_position_plate.view(self._num_envs, 3)[:, 2]
        self.reset_buf[:] = torch.where(body_heights < self.baseline_height, torch.ones_like(self.reset_buf), self.reset_buf)

        # Corner down reset
        ## Get the corner position in plate frame
        corner_pos_plate = inverse_transform_vectors(self.plate_quat_ground,
                                                     self.plate_pos_ground,
                                                     self.__corner_pos_world,
                                                     self._device)
        corner_heights =  corner_pos_plate[:, :, 2].view(self._num_envs, 4)
        corner_below_baseline = (corner_heights < self.baseline_corner_height)
        corner_below_baseline_sumup = torch.sum(corner_below_baseline, dim=-1)
        self.reset_buf[:] = torch.where(corner_below_baseline_sumup > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        # Knee position reset
        ## Get the knee position relative to the plate frame
        knee_position_world = self.knee_positions
        knee_position_plate = inverse_transform_vectors(self.plate_quat_ground, 
                                                        self.plate_pos_ground, 
                                                        knee_position_world,
                                                        self._device)
        knee_z_position_plate = knee_position_plate[:, :, 2].view(self._num_envs, 8)
        knee_z_below_thresh = knee_z_position_plate < self.baseline_knee_height
        knee_z_below_sum = torch.sum(knee_z_below_thresh, dim=1)
        self.reset_buf[:] = torch.where(knee_z_below_sum > 0, torch.ones_like(self.reset_buf), self.reset_buf)
        
        # Punish if finish without success
        # if self.current_curriculum_id != 0:
        #     # No failure penalty for phase1
        #     self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # Joint limit reset
        joint_reset_num = self.joint1_pos_reset + self.joint23_pos_reset
        self.reset_buf[:] = torch.where(joint_reset_num > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        # Add falling penalty
        fall_penalty = (self.fall_penalty * self.reset_buf).to(torch.float32)
        self.rew_buf += fall_penalty
        self.extras.update({"env/rewards/fall_penalty": fall_penalty.mean()})

        # if self.current_curriculum_id == 0:
        #    self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # Goal reset
        # if self.current_curriculum_id != 0:
        #     # No goal reset for phase 1
        #     self.reset_buf[:] = torch.where(self.goal_reset_buf == 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf[:] = torch.where(self.goal_reset_buf == 1, torch.ones_like(self.reset_buf), self.reset_buf)
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

    @abstractmethod
    def reset_robot_joint_positions(self, indices):
        raise NotImplementedError

    @abstractmethod
    def reset_robot_joint_velocities(self, indices):
        raise NotImplementedError

    @abstractmethod
    def reset_plate_poses(self, indices):
        raise NotImplementedError

    @abstractmethod
    def reset_plate_velocities(self, indices):
        raise NotImplementedError

    @abstractmethod
    def reset_goal_indicator_pose(self, goal_quaternions, indices):
        pass

    @abstractmethod
    def clear_robot_last_joint_velocities(self, indices):
        raise NotImplementedError

    @property
    def current_curriculum(self):
        return self.curriculums[self.current_curriculum_id]

    @property
    def quadruped_robot(self):
        """Implement this method with return the robot"""
        raise NotImplementedError

    @property
    def plate(self):
        """Implement this method with return the plate object"""
        raise NotImplementedError

    @property
    def fixed_robot_base_positions(self):
        """Implement this method with return the fixed position
        of the robot base
        """
        raise NotImplementedError

    @property
    def fixed_robot_base_quaternions(self):
        """Implement this method with return the fixed orientation
        (in quaternion) of the robot base
        """
        raise NotImplementedError