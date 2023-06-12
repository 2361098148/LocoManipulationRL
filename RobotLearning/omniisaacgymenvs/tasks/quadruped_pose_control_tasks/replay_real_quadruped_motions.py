import os
datalogger_dir = "/home/bionicdl/SHR/LocomanipulationTransfer/RobotLearning/real_experiment"
import sys
sys.path.append(datalogger_dir)

import pickle
import torch

from tasks.quadruped_pose_control_tasks.quadruped_pose_control_custom_controller_dr import QuadrupedPoseControlCustomControllerDR

from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_conjugate, quat_mul, quat_axis

class RealQuadrupedMotionReplay(QuadrupedPoseControlCustomControllerDR):
    def __init__(self, sim_config, name, env, offset=None) -> None:
        super().__init__(sim_config, name, env, offset)

        motion_dir = self._task_cfg["env"]["motion_dir"]
        motion_file_name = self._task_cfg["env"]["motion_file_name"]

        file_path = os.path.join(motion_dir, motion_file_name)
        with open(file_path, "rb") as f:
            # load the pickle object
            self.motion_data = pickle.load(file=f)

        self.max_step = len(self.motion_data.progress_step) - 1
        self.current_step = 0

    def set_up_scene(self, scene) -> None:
        return super().set_up_scene(scene)
    
    def post_reset(self):
        super().post_reset()
        # self.robot_locomotion._tip_view.disable_rigid_body_physics()

        # Get base positons
        base_positions = torch.tensor(self.motion_data.base_positions[0], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        base_quaternions = torch.tensor(self.motion_data.base_quaternions[0], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        joint_positions = self.default_joint_positions_loco
        # torch.tensor(self.motion_data.joint_positions[0], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        joint_velocities = torch.tensor(self.motion_data.joint_velocities[0], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)

        goal_quaternion = torch.tensor(self.motion_data.goal_quaternions[0], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)

        # Set robot pose and joints
        self.robot_locomotion.set_robot_pose(positions=base_positions,
                                             quaternions=base_quaternions)
        self.robot_locomotion.set_joint_positions(positions=joint_positions, full_joint_indices=True)
        # Set robot velocities
        zero_vels = torch.zeros(self._num_envs, 6, dtype=torch.float32, device=self._device)
        self.robot_locomotion.set_robot_velocities(zero_vels)
        # self.robot_locomotion.set_joint_velocities(joint_velocities, full_joint_indices=False)

        # Set marker
        marker_positions = torch.tensor([0.0, 0.0, 0.3], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        self.pose_indicator_loco.set_object_poses(positions=marker_positions,
                                                  quaternions=goal_quaternion)
        


    def pre_physics_step(self, actions):
        # replay motions

        # Get base positons
        base_positions = torch.tensor(self.motion_data.base_positions[self.current_step], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        base_quaternions = torch.tensor(self.motion_data.base_quaternions[self.current_step], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        joint_positions = torch.tensor(self.motion_data.joint_positions[self.current_step], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        joint_positions_full = torch.zeros(self._num_envs, 20, dtype=torch.float32, device=self._device)
        joint_positions_full[:, 0:12] = joint_positions
        joint_positions_full[:, 12:] = self.default_joint_positions_loco[:, 12:]
        joint_velocities = torch.tensor(self.motion_data.joint_velocities[self.current_step], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)

        goal_quaternion = torch.tensor([0.707, 0.0, 0.0, 0.707], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        # torch.tensor(self.motion_data.goal_quaternions[self.current_step], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)

        # Set robot pose and joints
        self.robot_locomotion.set_robot_pose(positions=base_positions,
                                             quaternions=base_quaternions)
        self.robot_locomotion.set_joint_positions(positions=joint_positions_full, full_joint_indices=True)
        # Set robot velocities
        zero_vels = torch.zeros(self._num_envs, 6, dtype=torch.float32, device=self._device)
        self.robot_locomotion.set_robot_velocities(zero_vels)
        # self.robot_locomotion.set_joint_velocities(joint_velocities, full_joint_indices=False)

        # Set marker
        marker_positions = torch.tensor([0.0, 0.0, 0.3], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        self.pose_indicator_loco.set_object_poses(positions=marker_positions,
                                                  quaternions=goal_quaternion)
        
        # zero_torques = torch.zeros_like(actions)
        # self.robot_locomotion._robot_articulation.set_joint_efforts(zero_torques, joint_indices=self.robot_locomotion._omni_dof_indices)

    def calculate_metrics(self) -> None:
        pass

    def is_done(self):

        # Check if max len reached
        if self.current_step > self.max_step-1:
            goal_quaternion = torch.tensor([0.707, 0.0, 0.0, 0.707], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
            quat_diff = quat_mul(self.base_quaternions_loco, quat_conjugate(goal_quaternion))
            rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)) # changed quat convention
            print("Rotation distance: ", rot_dist)
            # Return to the first step
            self.current_step = 0
        else:
            self.current_step += 1