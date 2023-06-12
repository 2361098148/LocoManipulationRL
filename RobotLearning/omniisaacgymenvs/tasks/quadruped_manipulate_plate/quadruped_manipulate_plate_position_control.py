from abc import abstractmethod
# from tasks.base.rl_task import RLTask
import torch

from utils.math import transform_vectors, inverse_transform_vectors, rand_quaternions, inverse_rotate_orientations, rotate_orientations

from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_conjugate, quat_mul, quat_axis, quat_apply

import torch
import numpy as np

from robot.quadruped_robot import QuadrupedRobotOVOmni, QuadrupedRobotOVFixedBaseOmni
from tasks.base.rl_task import RLTask
from objects.pose_indicator import PoseIndicator
from objects.plate import Plate50cmOmni

from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.replicator.isaac as dr

class QuadrupedManipulatePlatePositionControl(RLTask):
    '''
        The task is to train manipulation
    '''
    max_effort = 1.5 # Nm
    control_kp = 4.5
    control_kd = 0.2
    joint_damping = 0.008
    joint_friction = 0.007
    action_scale = 0.1
    control_decimal = 4 # 40Hz, 0.025s/step
    min_joint_pos = [-2.35, -0.78, -0.78, -2.35,
                     -1.57, -3.14,
                     1.57, 0.0,
                     1.57, 0.0,
                     -1.57, -3.14]
    max_joint_pos = [0.78,  2.35,  2.35,  0.78,
                     0.0,   -1.57,
                     3.14, 1.57,
                     1.57, 0.0,
                     0.0, -1.57]
    min_joint_pos_swing_ext = [-2.35, -0.78, -0.78, -2.35,
                                -2.09, 0.52, # Swing, extension
                                1.05, 0.52,
                                1.05, 0.52,
                                -2.09, 0.52]
    max_joint_pos_swing_ext = [0.78,  2.35,  2.35,  0.78,
                                -1.05,   2.09, # Swing, extension
                                2.09, 2.09,
                                2.09, 2.09,
                                -1.05, 2.09]

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
    # Task related
    min_roll = -0.5
    max_roll = 0.5
    min_pitch = -0.5
    max_pitch = 0.5
    min_yaw = -3.14
    max_yaw = 3.14

    # min_roll = -0.0
    # max_roll = 0.0
    # min_pitch = -0.0
    # max_pitch = 0.0
    # min_yaw = np.pi
    # max_yaw = np.pi

    # Reward scales
    ## Orientation scales
    quaternion_scale = 0.5
    rot_eps = 0.1
    ## Translation scale
    translation_scale = -2.5
    ## Joint movement scale
    joint_acc_scale = -0.0005 # max about 0.5 for each dimension
    ## Action rate scale
    action_rate_scale = -0.02

    ## Reset threshold
    baseline_knee_height = 0.04
    fall_penalty = 0.0 # -200
    baseline_height = 0.05
    baseline_corner_height = 0.01

    # Goal threshold
    # success_thresh = 0.1
    # success_bonus = 5 # per residual step
    # max_consecutive_successes = 30

    success_thresh = 0.1
    success_bonus = 3 # per residual step
    max_consecutive_successes = 15

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

    def __init__(self, sim_config, name, env, offset=None) -> None:
        self.robot_manipulation = QuadrupedRobotOVFixedBaseOmni()
        self.robot_manipulation.robot_description.default_quaternion = [0.0, 1.0, 0.0, 0.0]
        self.robot_manipulation.robot_description.default_position = [0.0, 0.0, 0.3] # [0.0, 0.0, 0.0]
        self.robot_manipulation.robot_description.init_joint_pos = [-1.57, 1.57, 1.57, -1.57,
                                                                    -1.04, -2.09, 
                                                                    2.09, 1.04, 
                                                                    2.09, 1.04, 
                                                                    -1.04, -2.09, 
                                                                    1.37, -1.37, 
                                                                    1.37, -1.37, 
                                                                    1.37, -1.37, 
                                                                    1.37, -1.37]
        self.robot_manipulation.robot_description.control_mode = "effort"
        # self.robot_manipulation.robot_description.control_mode = "position"
        # self.robot_manipulation.robot_description.joint_kps = [8.0, 8.0, 8.0] * 4
        # self.robot_manipulation.robot_description.joint_kds = [1.0, 1.0, 1.0] * 4
        self.obj = Plate50cmOmni()
        self.pose_indicator_mani = PoseIndicator(object_name="pose_indicator_mani",
                                                 object_prim_name="/pose_indicator")
        
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._dt = self._task_cfg['sim']['dt']
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg['sim']['max_episode_length']

        self._num_actions = 12
        self._num_observations = 64
        self._num_states = 64

        super().__init__(name, env, offset)

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

        # Default joint positions
        self.default_joint_positions_mani = torch.tensor(self.robot_manipulation.robot_description.init_joint_pos, dtype=torch.float32, device=self._device).repeat((self._num_envs,1))
        self.default_robot_positions_mani = torch.tensor(self.robot_manipulation.robot_description.default_position, dtype=torch.float32, device=self._device).repeat((self._num_envs,1))
        self.default_robot_quaternions_mani = torch.tensor(self.robot_manipulation.robot_description.default_quaternion, dtype=torch.float32, device=self._device).repeat((self._num_envs,1))
        # self.default_obj_positions_mani = torch.tensor([0.0, 0.0, 0.14], dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))
        self.default_obj_positions_mani = torch.tensor([0.0, 0.0, 0.44], dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))
        self.default_obj_quaternions_mani = torch.tensor(self.robot_manipulation.robot_description.default_quaternion, dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))
        # self.default_pose_indicator_mani_positions = torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))
        self.default_pose_indicator_mani_positions = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))
        self.joint_positions_mani = self.default_joint_positions_mani[:, 0:12].clone()
        self.joint_velocities_mani = torch.zeros(self._num_envs, 12, dtype=torch.float32, device=self._device)
        self.current_joint_position_targets = self.default_joint_positions_mani[:, 0:12].clone()
        self.last_joint_position_targets = self.current_joint_position_targets.clone()
        self.current_joint_position_targets_se = torch.tensor([-1.57, 1.57, 1.57, -1.57,
                                                               -1.57, 1.05,
                                                               1.57, 1.05,
                                                               1.57, 1.05,
                                                               -1.57, 1.05], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        self.joint_position_target_upper = torch.tensor(self.max_joint_pos, dtype=torch.float32, device=self._device)# .repeat(self._num_envs, 1)
        self.joint_position_target_lower = torch.tensor(self.min_joint_pos, dtype=torch.float32, device=self._device)# .repeat(self._num_envs, 1)
        self.joint_position_target_se_upper = torch.tensor(self.max_joint_pos_swing_ext, dtype=torch.float32, device=self._device)# .repeat(self._num_envs, 1)
        self.joint_position_target_se_lower = torch.tensor(self.min_joint_pos_swing_ext, dtype=torch.float32, device=self._device)# .repeat(self._num_envs, 1)

        # Metrics
        # Success for both envs
        self.max_reset_counts = torch.tensor(2048, dtype=torch.long, device=self._device)
        self.num_successes = torch.tensor(0, dtype=torch.long, device=self._device)
        self.num_resets = torch.tensor(0, dtype=torch.long, device=self._device)
        self.success_rate = torch.tensor(0.0, dtype=torch.float32, device=self._device)

        self._time_inv = self._dt * (self.control_decimal + 1)

        self.__corner_pos_world = transform_vectors(self.default_robot_quaternions_mani,
                                                    self.default_robot_positions_mani,
                                                    self.__corner_pos_robot,
                                                    self._device)

        # Domain randomization buffers
        self.randomization_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

    def set_up_scene(self, scene) -> None:
        self.robot_manipulation.init_omniverse_robot(self.default_zero_env_path)
        self._sim_config.apply_articulation_settings(self.robot_manipulation.robot_name, get_prim_at_path(self.robot_manipulation._omniverse_robot.prim_path), self._sim_config.parse_actor_config(self.robot_manipulation.robot_name))
        
        self.pose_indicator_mani.init_stage_object(self.default_zero_env_path, self._sim_config)

        self.obj.init_stage_object(self.default_zero_env_path, self._sim_config)

        super().set_up_scene(scene)

        # print(scene.stage.GetPrimAtPath(self.default_zero_env_path).GetAllChildren())
        # print(scene.stage.GetPrimAtPath(self.default_zero_env_path + "/" + self.robot_locomotion.robot_name).GetChildren())
        # print(scene.stage.GetPrimAtPath(self.default_zero_env_path + "/" + self.robot_manipulation.robot_name).GetChildren())
        # print(scene.stage.GetPrimAtPath(self.default_zero_env_path + "/" + self.pose_indicator_mani.object_name).GetChildren())

        # Init robot views
        self.robot_manipulation.init_robot_views(scene)

        # Init pose indicator views
        self.pose_indicator_mani.init_object_view(scene)

        # Init obj views
        self.obj.init_object_view(scene)

        # Apply on startup domain randomization
        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)
            print("Applying on startup domain randomization...")

        print("Finished setting up scenes! ")
        return

    def post_reset(self):
        self.robot_manipulation.post_reset_robot(self._env_pos, self._num_envs, self._dt, self._device)

        # set joint damping
        damping = torch.tensor(self.joint_damping, dtype=torch.float32, device=self._device).repeat(self._num_envs, 12)
        self.robot_manipulation._robot_articulation.set_gains(kds=damping, joint_indices=self.robot_manipulation._omni_dof_indices)
        # Set joint friction
        friction = torch.tensor(self.joint_friction, dtype=torch.float32, device=self._device).repeat(self._num_envs, 12)
        self.robot_manipulation._robot_articulation.set_friction_coefficients(values=friction, joint_indices=self.robot_manipulation._omni_dof_indices)

        self.pose_indicator_mani.set_env_pos(self._env_pos)

        self.obj.set_env_pos(self._env_pos)
        
        # Apply domain randomization
        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

    def pre_physics_step(self, actions):
        actions = actions.to(self._device)
        self.current_actions = actions.clone()

        # Check if the environment needes to be reset
        reset_buf = self.reset_buf.clone()
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions[:] = 0.0

        # # Get joint position targets
        # delta_joint_position_targets = actions * self.action_scale
        # self.current_joint_position_targets = self.current_joint_position_targets + delta_joint_position_targets
        # # Clamp the joint position targets
        # self.current_joint_position_targets = torch.clamp(self.current_joint_position_targets, min=-3.14, max=3.14)
        # Get joint target positions in swing extension form
        delta_joint_position_targets = actions * self.action_scale
        self.current_joint_position_targets_se = self.current_joint_position_targets_se + delta_joint_position_targets
        self.current_joint_position_targets_se = torch.clamp(self.current_joint_position_targets_se, 
                                                          min=self.joint_position_target_se_lower, 
                                                          max=self.joint_position_target_se_upper)
        # Convert swing extension angles to dof2 and dof3 angles
        joint_swing = self.current_joint_position_targets_se[:, [4,6,8,10]]
        joint_extension = self.current_joint_position_targets_se[:, [5,7,9,11]]
        dof2_joint_positions = joint_swing + joint_extension/2.0
        dof3_joint_positions = joint_swing - joint_extension/2.0
        self.current_joint_position_targets[:, 0:4] = self.current_joint_position_targets_se[:, 0:4] # For dof1, they are the same
        self.current_joint_position_targets[:, [4,6,8,10]] = dof2_joint_positions # For dof2 joint positions
        self.current_joint_position_targets[:, [5,7,9,11]] = dof3_joint_positions # For dof3 joint positions
        
        torque_sum = 0.0

        for i in range(self.control_decimal):
            # apply torque to the joints
            torque = self.control_kp * (self.current_joint_position_targets-self.joint_positions_mani) - self.control_kd * self.joint_velocities_mani
            torque = torch.clamp(torque, min=-self.max_effort, max=self.max_effort)
            self.robot_manipulation._robot_articulation.set_joint_efforts(torque, joint_indices=self.robot_manipulation._omni_dof_indices)
            torque_sum += torque
            from omni.isaac.core.simulation_context import SimulationContext
            SimulationContext.step(self._env._world, render=False)
            self.robot_manipulation.update_joint_states()
            self.joint_positions_mani = self.robot_manipulation.joint_positions
            self.joint_velocities_mani = self.robot_manipulation.joint_velocities

        # apply torque to the joints
        torque = self.control_kp * (self.current_joint_position_targets-self.joint_positions_mani) - self.control_kd * self.joint_velocities_mani
        torque = torch.clamp(torque, min=-self.max_effort, max=self.max_effort)
        self.robot_manipulation._robot_articulation.set_joint_efforts(torque, joint_indices=self.robot_manipulation._omni_dof_indices)
        torque_sum += torque

        self.torque = torque_sum/self.control_decimal

        reset_buf = self.reset_buf.clone()
        # Apply on reset and on interval domain randomization
        if self._dr_randomizer.randomize:
            rand_envs = torch.where(self.randomization_buf >= self._dr_randomizer.min_frequency, torch.ones_like(self.randomization_buf), torch.zeros_like(self.randomization_buf))
            rand_env_ids = torch.nonzero(torch.logical_and(rand_envs, reset_buf))
            dr.physics_view.step_randomization(rand_env_ids)
            self.randomization_buf[rand_env_ids] = 0

    def reset_idx(self, env_ids):
        num_reset = len(env_ids)

        rand_quaternions_mani = rand_quaternions(num_reset, 
                                                 self.min_roll,
                                                 self.max_roll,
                                                 self.min_pitch,
                                                 self.max_pitch,
                                                 self.min_yaw,
                                                 self.max_yaw,
                                                 self._device)

        '''Reset all envs regardless of success or not'''
        # Reset joint positions and velocities
        self.robot_manipulation.set_joint_positions(self.default_joint_positions_mani[env_ids], indices=env_ids, full_joint_indices=True)
        # Reset joint velocities
        zero_velocities_mani = torch.zeros(num_reset, 20, dtype=torch.float32, device=self._device)
        self.robot_manipulation.set_joint_velocities(zero_velocities_mani, indices=env_ids, full_joint_indices=True)

        self.obj.set_object_poses(positions=self.default_obj_positions_mani[env_ids], 
                                quaternions=self.default_obj_quaternions_mani[env_ids], # obj_quaternions, # self.default_obj_quaternions_mani[env_ids],
                                indices=env_ids)
        
        # Reset obj velocities
        zero_obj_velocities = torch.zeros(num_reset, 6, dtype=torch.float32, device=self._device)
        self.obj.set_object_velocities(velocities=zero_obj_velocities,
                                        indices=env_ids)
        
        # Clear robot last velocities
        self.robot_manipulation.last_joint_velcoties[env_ids, :] = torch.zeros(num_reset, 12, dtype=torch.float32, device=self._device)

        # Reset goal quaternions
        # Transform goal quaternions from world frame to robot frame
        self.goal_quaternions[env_ids, :] = rand_quaternions_mani
        indicator_quaternion = rotate_orientations(self.default_robot_quaternions_mani[env_ids], rand_quaternions_mani, device=self._device)
        self.pose_indicator_mani.set_object_poses(quaternions=indicator_quaternion,
                                                    positions=self.default_pose_indicator_mani_positions[env_ids],
                                                    indices=env_ids)

        # Reset last base tip positions
        self.last_base_tip_positions[env_ids] = self.default_base_tip_positions[env_ids]

        # Reset history buffers for env_ids
        self.last_actions[env_ids] = torch.zeros(num_reset, self._num_actions, dtype=torch.float32, device=self._device)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # Reset buffers
        self.successes[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)
        self.consecutive_successes[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)
        self.goal_reset_buf[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)

        self.joint_positions_mani[env_ids, :] = self.default_joint_positions_mani[env_ids, 0:12]
        self.joint_velocities_mani[env_ids, :] = torch.zeros(num_reset, 12, dtype=torch.float32, device=self._device)
        self.current_joint_position_targets[env_ids, :] = self.default_joint_positions_mani[env_ids, 0:12]
        self.last_joint_position_targets[env_ids, :] = self.current_joint_position_targets[env_ids, :].clone()
        self.current_joint_position_targets_se[env_ids, :] = torch.tensor([-1.57, 1.57, 1.57, -1.57,
                                                                            -1.57, 1.05,
                                                                            1.57, 1.05,
                                                                            1.57, 1.05,
                                                                            -1.57, 1.05], dtype=torch.float32, device=self._device).repeat(num_reset, 1)

    def get_observations(self):
        self.robot_manipulation.update_joint_states()
        self.robot_manipulation.update_tip_states()
        self.robot_manipulation.update_knee_positions()

        '''Observations for manipulation tasks'''
        self.joint_positions_mani = self.robot_manipulation.joint_positions
        self.joint_velocities_mani = self.robot_manipulation.joint_velocities
        self.joint_accelerations_mani = self.robot_manipulation.joint_accelerations
        self.tip_positions_mani = self.robot_manipulation.tip_positions
        self.knee_positions_mani = self.robot_manipulation.knee_positions
        '''Scaled observations'''
        # Get plate position and quaternion relative to the ground
        self.plate_pos_ground, self.plate_quat_ground = self.obj.get_object_poses()
        # Plate positions relative to the robot base
        self.plate_pos_robot = inverse_transform_vectors(self.default_robot_quaternions_mani, 
                                                         self.default_robot_positions_mani, 
                                                         self.plate_pos_ground.view(self._num_envs, 1, 3),
                                                         device=self._device).view(self._num_envs, 3)
        scaled_plate_pos_robot = self.ground_position_scale * self.plate_pos_robot
        # Plate quaternions relative to the robot base
        self.plate_quat_robot = inverse_rotate_orientations(self.default_robot_quaternions_mani, self.plate_quat_ground, device=self._device)
        scaled_plate_quat_robot = self.ground_quaternion_scale * self.plate_quat_robot
        # Goal plate quaternions
        scaled_goal_plate_quat_robot = self.ground_quaternion_scale * self.goal_quaternions
        # Calculate plate linear&angular velocities under the robot base frame
        plate_linear_vel_ground, plate_angular_vel_ground = self.obj.get_object_velocities()
        self.plate_linear_vel_robot = quat_rotate_inverse(self.default_robot_quaternions_mani, plate_linear_vel_ground)
        self.plate_angular_vel_robot = quat_rotate_inverse(self.default_robot_quaternions_mani, plate_angular_vel_ground)
        scaled_plate_linear_vel_robot = self.ground_linear_vel_scale * self.plate_linear_vel_robot
        scaled_plate_angular_vel_robot = self.ground_angular_vel_scale * self.plate_angular_vel_robot
        # Get tip positions relative to the robot base
        base_tip_positions_mani = inverse_transform_vectors(self.default_robot_quaternions_mani, self.default_robot_positions_mani, self.tip_positions_mani, device=self._device)
        flat_base_tip_positions_mani = base_tip_positions_mani.reshape(self._num_envs, self.robot_manipulation.num_modules*3)
        last_tip_positions_mani = self.last_base_tip_positions.clone()
        flat_last_base_tip_positions_mani = last_tip_positions_mani.reshape(self._num_envs, self.robot_manipulation.num_modules*3)
        scaled_base_tip_positions = self.base_tip_position_scale * flat_base_tip_positions_mani
        scaled_last_base_tip_positions = self.base_tip_position_scale * flat_last_base_tip_positions_mani
        # joint positions
        scaled_joint_positions = self.joint_position_scale * self.joint_positions_mani
        # joint velocities
        scaled_joint_velocities = self.joint_velocity_scale * self.joint_velocities_mani

        quat_diff_mani = quat_mul(self.plate_quat_robot, quat_conjugate(self.goal_quaternions))
        # Make the real part of the quaternion positive
        quat_diff_mani = torch.where((quat_diff_mani[:, 0] < 0.0).view(-1, 1),
                                     -quat_diff_mani,
                                     quat_diff_mani)
        plate_up_vec = quat_axis(self.plate_quat_robot, 2)

        scaled_joint_position_targets = self.current_joint_position_targets * 0.3
        scaled_last_joint_position_targets = self.last_joint_position_targets * 0.3

        self.obs_buf[:] = torch.cat((scaled_plate_pos_robot,
                                     plate_up_vec,
                                     quat_diff_mani,

                                     scaled_plate_linear_vel_robot,
                                     scaled_plate_angular_vel_robot,

                                     scaled_joint_positions,
                                     scaled_joint_velocities,

                                     scaled_joint_position_targets,
                                     scaled_last_joint_position_targets
                                    ), dim=-1)

        self.states_buf[:] = self.obs_buf[:]

        # Update last tip positions
        self.last_base_tip_positions = base_tip_positions_mani

    def calculate_metrics(self) -> None:
        '''Pose error penalty'''
        quat_diff = quat_mul(self.plate_quat_robot, quat_conjugate(self.goal_quaternions))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)) # changed quat convention
        rot_rew = 1.0/(torch.abs(rot_dist) + self.rot_eps) * self.quaternion_scale

        '''Translation penalty'''
        # Get xy positions
        deviation_distance = torch.norm(self.plate_pos_robot[:, 0:2], dim=-1)
        # Punish any translation from origin in xy axis
        translation_penalty = deviation_distance * self.translation_scale

        '''joint acceleration penalty'''
        joint_acc_penalty = torch.sum(torch.abs(self.joint_accelerations_mani)*self.joint_acc_scale, dim=1)

        '''action rate penalty'''
        action_rate_penalty = torch.sum(torch.abs(self.last_actions - self.current_actions), dim=1)*self.action_rate_scale

        '''consecutive successes bonus reward'''
        consecutive_goal_reset = torch.where(self.consecutive_successes > self.max_consecutive_successes, 
                                                torch.ones_like(self.consecutive_successes), 
                                                torch.zeros_like(self.consecutive_successes))
        consecutive_successes_rew = 600.0*consecutive_goal_reset.to(torch.float32)

        # Check if success
        successes = torch.where(torch.abs(rot_dist) <= self.success_thresh, 
                            torch.ones_like(consecutive_goal_reset), 
                            torch.zeros_like(consecutive_goal_reset))

        '''joint limit penalty'''
        joint_positions = self.joint_positions_mani
        a14_dof1_positions = joint_positions[:, [0,3]]
        a23_dof1_positions = joint_positions[:, [1,2]]
        dof2_positions = joint_positions[:, [4,6,8,10]]
        dof3_positions = joint_positions[:, [5,7,9,11]]
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
        # Ground/Plate above robot reset
        plate_z_pos_robot = self.plate_pos_robot[:, 2]
        self.reset_buf[:] = torch.where(plate_z_pos_robot > 0.0, torch.ones_like(self.reset_buf), self.reset_buf)

        # Fall down reset
        ## Body heights for manipulation
        body_position_plate = inverse_transform_vectors(self.plate_quat_ground,
                                                        self.plate_pos_ground,
                                                        self.default_robot_positions_mani.view(self._num_envs,1,3),
                                                        self._device)
        body_heights_mani = body_position_plate.view(self._num_envs, 3)[:, 2]
        self.reset_buf[:] = torch.where(body_heights_mani <= self.baseline_height, torch.ones_like(self.reset_buf), self.reset_buf)

        # Corner fall down reset
        ## Corner heights for manipulation
        corner_pos_plate = inverse_transform_vectors(self.plate_quat_ground,
                                                     self.plate_pos_ground,
                                                     self.__corner_pos_world,
                                                     self._device)
        corner_heights_mani =  corner_pos_plate[:, :, 2].view(self._num_envs, 4)
        corner_below_baseline = (corner_heights_mani < self.baseline_corner_height)
        corner_below_baseline_sumup = torch.sum(corner_below_baseline, dim=-1)
        self.reset_buf[:] = torch.where(corner_below_baseline_sumup > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        # Knee touching ground reset; reset if anyone of the knees touches the ground
        ## Knee position for manipulation
        knee_position_plate = inverse_transform_vectors(self.plate_quat_ground, 
                                                        self.plate_pos_ground, 
                                                        self.knee_positions_mani,
                                                        self._device)
        knee_z_mani = knee_position_plate[:, :, 2].view(self._num_envs, 8)
        knee_z_below_baseline = knee_z_mani - self.baseline_knee_height <= 0
        knee_z_below_sum = torch.sum(knee_z_below_baseline, dim=1)
        self.reset_buf[:] = torch.where(knee_z_below_sum > 0, torch.ones_like(self.reset_buf), self.reset_buf)

        # # Tip position larger than allowed (a circle of 0.25m radius)
        # ## Get tip positions in plate frame for manipulation envs
        # tip_positions_plate_mani = inverse_transform_vectors(self.plate_quat_ground, 
        #                                                self.plate_pos_ground, 
        #                                                self.tip_positions_mani, 
        #                                                device=self._device)
        # tip_positions = tip_positions_plate_mani
        # ## Approximately determine if tip leaves the ground
        # tip_z_distance = tip_positions[:, :, 2].view(self._num_envs, 4)
        # tip_z_above_thresh = tip_z_distance < 0.05
        # tip_xy_distance = tip_positions[:, :, 0:2]
        # tip_planar_distance = torch.norm(tip_xy_distance, dim=-1).view(self._num_envs, 4)
        # tip_planar_above_thresh = tip_planar_distance > 0.25
        # tip_z_and_planar_above_thresh = torch.logical_and(tip_z_above_thresh, tip_planar_above_thresh)
        # tip_pos_reset_sumup = torch.sum(tip_z_and_planar_above_thresh, dim=-1)
        # self.reset_buf[0:] = torch.where(tip_pos_reset_sumup > 0, torch.ones_like(self.reset_buf), self.reset_buf)

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