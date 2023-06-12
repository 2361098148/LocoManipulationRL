import torch
import numpy as np
import csv
import pandas
import os
from collections import namedtuple

from tasks.base.rl_task import RLTask
from robot.base.robot import OmniverseRobot
from utils.path import *

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import get_current_stage

NUM_ENVS = 1

class TrajectoryRecorder(object):
    def __init__(self, trajectory_folder, trajectory_file_name) -> None:
        self.trajectory_folder = trajectory_folder
        self.trajectory_file_name = trajectory_file_name
        self.trajectory_file_path = os.path.join(self.trajectory_folder, trajectory_file_name)

        self.csv_file = open(self.trajectory_file_path, "w", newline='')
        self.csv_writer = csv.writer(self.csv_file, delimiter=',',
                                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Write column names
        self.csv_writer.writerow(["time", "position", "velocity"])

    def write_trajectory(self, time, position, velocity):
        self.csv_writer.writerow([time, position, velocity])

    def close(self):
        self.csv_file.close()

class FreeFallLoadExperimentOmni(RLTask):
    def __init__(self, sim_config, name, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._dt = self._task_cfg['sim']['dt']
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg['sim']['max_episode_length']

        self._num_actions = 1
        self._num_observations = 1
        self._num_states = 1
        
        self.robot = None
        self.robot_name = "robot"
        self.robot_default_position = [0, 0, 0.5]
        self.robot_articulation = None

        self.joint_friction = self._task_cfg["friction"]["joint_friction"]
        self.damping = self._task_cfg["friction"]["damping"]

        self.traj_amplitude = self._task_cfg["trajectory"]["amplitude"]
        self.traj_time_span = self._task_cfg["trajectory"]["time_span"]
        self.traj_file_name = self._task_cfg["trajectory"]["trajectory_file_name"]

        self.trajectory_file_folder = os.path.join(env_dir, "runs", "FreeFallLoad")

        super().__init__(name, env, offset)

        self.time_interval = self._dt * self.control_frequency_inv
        self.time_interval_ms = self.time_interval * 1000.0
        self.current_record_episode_len = round(self.traj_time_span/self.time_interval)

        self.robot_usd = os.path.join(root_ws_dir, "Design", "RobotUSD", "load_experiment_medium_arm", "load_experiment_medium_arm.usd")

        self.joint_positions = torch.zeros(self._num_envs, 1, dtype=torch.float32, device=self._device)
        self.joint_velocities = torch.zeros(self._num_envs, 1, dtype=torch.float32, device=self._device)

        self.trajectory_error_list = []

        self.current_time_ms = 0.0

    def set_up_scene(self, scene) -> None:
        prim_path = self.default_zero_env_path + "/" + self.robot_name
        self.robot = OmniverseRobot(prim_path,
                                    self.robot_name,
                                    self.robot_usd,
                                    self.robot_default_position)
        
                
        self._sim_config.apply_articulation_settings(self.robot_name, get_prim_at_path(self.robot.prim_path), self._sim_config.parse_actor_config(self.robot_name))
        stage = get_current_stage()
        prim = self.robot.prim

        from pxr import PhysxSchema
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64/np.pi*180)

        super().set_up_scene(scene)

        print(scene.stage.GetPrimAtPath(self.default_zero_env_path).GetAllChildren())
        print(scene.stage.GetPrimAtPath(self.default_zero_env_path + "/" + self.robot_name).GetChildren())

        arti_root_name = "/World/envs/.*/" + self.robot_name + "/load"
        self.robot_articulation = ArticulationView(prim_paths_expr=arti_root_name, name="robot_view", enable_dof_force_sensors=True)
        scene.add(self.robot_articulation)

    def post_reset(self):
        self.robot_articulation.switch_control_mode("effort")

        # Set joint damping anyway
        # if self.custom_controller:
        joint_damping = torch.tensor([self.damping], dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        self.robot_articulation.set_gains(kds=joint_damping)

        # Set joint friction
        joint_friction = torch.tensor([self.joint_friction], dtype=torch.float32, device=self._device)
        self.robot_articulation.set_friction_coefficients(joint_friction)

        # Print out the parameters
        print("<Control Parameters>")

        # Get actual joint gains
        stiffness, damping = self.robot_articulation.get_gains()
        print("Kp: ", stiffness)
        print("Kd: ", damping)

        # Get actual max efforts
        max_efforts = self.robot_articulation.get_max_efforts()
        print("Max effort: ", max_efforts)

        # Get actual max velocities
        max_velocities = self.robot_articulation._physics_view.get_dof_max_velocities()
        print("Max velocity: ", max_velocities)

        # Get actual friction coefficient
        friction_coeff = self.robot_articulation.get_friction_coefficients()
        print("Friction coefficient: ", friction_coeff)

    def reset_idx(self, env_ids):
        # prepare csv file writers
        self.traj_recorder = TrajectoryRecorder(self.trajectory_file_folder, self.traj_file_name)
        self.traj_recorder.write_trajectory(0.0, self.traj_amplitude, 0.0)

        # Reset trajectory and time buffer
        self.time_list = []
        self.trajectory_list = []

        # Set to init position
        init_pos_tensor = torch.tensor([self.traj_amplitude], dtype=torch.float32, device=self._device).repeat(NUM_ENVS, 1)
        self.robot_articulation.set_joint_positions(init_pos_tensor)

        # Set zero velocities
        zero_velocity = torch.zeros(NUM_ENVS, 1, dtype=torch.float32, device=self._device)
        self.robot_articulation.set_joint_velocities(zero_velocity)

        # Reset progress buf
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # Check if the environment needes to be reset
        reset_buf = self.reset_buf.clone()
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Do nothing, let it fall freely
        # for i in range(4):
        #     from omni.isaac.core.simulation_context import SimulationContext
        #     # joint_vel = self.robot_articulation.get_joint_velocities()
        #     # joint_viscous_friction = 0.0 # torch.tensor([1.2]) # joint_vel * -0.0
        #     # self.robot_articulation.set_joint_efforts(joint_viscous_friction)
        #     SimulationContext.step(self._env._world, render=False)

    def get_observations(self):
        # Record only when preparation is finished
        self.joint_positions = self.robot_articulation.get_joint_positions()
        self.joint_velocities = self.robot_articulation.get_joint_velocities()

        self.obs_buf[:] = self.joint_positions
        self.states_buf[:] = self.joint_positions

        current_time = round((self.progress_buf.to('cpu').numpy()[0]) * self._dt * self.control_frequency_inv * 1000.0)
        current_joint_position = self.joint_positions.to('cpu').numpy()[0][0]
        current_joint_velocity = self.joint_velocities.to('cpu').numpy()[0][0]

        # Save time and position to buffer
        self.time_list.append(current_time)
        self.trajectory_list.append(current_joint_position)

        self.traj_recorder.write_trajectory(current_time, current_joint_position, current_joint_velocity)

        # Increment time
        self.current_time_ms = 20 + current_time

    def calculate_metrics(self):
        # Print infos
        # print("<<<")
        # print("Joint Position: ", self.joint_positions)
        # print("Joint Velocity: ", self.joint_velocities)
        pass

    def is_done(self):
        if self.progress_buf[0] > self.current_record_episode_len - 1:
            # Current curriculum is done
            self.traj_recorder.close()
            exit(0)
            self.reset_buf[:] = 1