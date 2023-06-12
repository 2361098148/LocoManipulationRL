import torch
import numpy as np

from tasks.quadruped_manipulate_plate.united_quadruped_manipulate_plate import UnitedQuadrupedManipulatePlate
from robot.quadruped_robot import QuadrupedRobotFixedBaseOmni, QuadrupedRobotOVFixedBaseOmni
from tasks.base.rl_task import RLTask
from objects.pose_indicator import PoseIndicator
from objects.plate import Plate50cmOmni

from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.replicator.isaac as dr

class UnitedQuadrupedManipulatePlateOmni(RLTask, UnitedQuadrupedManipulatePlate):
    def __init__(self, sim_config, name, env, offset=None) -> None:
        self.training = False

        self.robot = QuadrupedRobotOVFixedBaseOmni() # QuadrupedRobotFixedBaseOmni()
        self._plate = Plate50cmOmni()

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._dt = self._task_cfg['sim']['dt']
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg['sim']['max_episode_length']
        
        self._num_actions = 12
        self._num_observations = 89
        self._num_states = 89

        super().__init__(name, env, offset)
        UnitedQuadrupedManipulatePlate.__init__(self)

        self.current_curriculum_id = 2

        self._time_inv = self._dt * self.control_frequency_inv

        # Domain randomization buffers
        self.randomization_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        '''For friction randomization'''
        self._default_material_prim_path = "/physicsScene/defaultMaterial"
        self._default_material = None
        self._friction_reset_buf = 0
        # Friction reset parameters
        self._friciton_reset_range = [1.0, 1.0] # Value from 0.6-0.9
        self._friction_reset_interval = 1000

        if not self.training:
            self.pose_indicator = PoseIndicator()

    def set_up_scene(self, scene) -> None:
        self.robot.init_omniverse_robot(self.default_zero_env_path)
        self._sim_config.apply_articulation_settings(self.robot.robot_name, get_prim_at_path(self.robot._omniverse_robot.prim_path), self._sim_config.parse_actor_config(self.robot.robot_name))
        
        if not self.training:
            self.pose_indicator.init_stage_object(self.default_zero_env_path, self._sim_config)
        
        self._plate.init_stage_object(self.default_zero_env_path, self._sim_config)

        super().set_up_scene(scene)

        print(scene.stage.GetPrimAtPath(self.default_zero_env_path).GetAllChildren())
        print(scene.stage.GetPrimAtPath(self.default_zero_env_path + "/" + self.robot.robot_name).GetChildren())

        # self.robot.init_robot_articulation(scene)
        self.robot.init_robot_views(scene)
        if not self.training:
            self.pose_indicator.init_object_view(scene)
        self._plate.init_object_view(scene)

        # Apply on startup domain randomization
        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)
            print("Applying on startup domain randomization...")

        print("Finished setting up scenes! ")
        return

    def post_reset(self):
        self.robot.post_reset_robot(self._env_pos, self._num_envs, self._time_inv, self._device)
        if not self.training:
            self.pose_indicator.set_env_pos(self._env_pos)
        self._plate.set_env_pos(self._env_pos)
        
        # Apply domain randomization
        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

        self._default_material = PhysicsMaterial(prim_path=self._default_material_prim_path)

    def pre_physics_step(self, actions):
        UnitedQuadrupedManipulatePlate.pre_physics_step(self, actions)

        reset_buf = self.reset_buf.clone()
        # Apply on reset and on interval domain randomization
        if self._dr_randomizer.randomize:
            rand_envs = torch.where(self.randomization_buf >= self._dr_randomizer.min_frequency, torch.ones_like(self.randomization_buf), torch.zeros_like(self.randomization_buf))
            rand_env_ids = torch.nonzero(torch.logical_and(rand_envs, reset_buf))
            dr.physics_view.step_randomization(rand_env_ids)
            self.randomization_buf[rand_env_ids] = 0

    def get_observations(self):
        return UnitedQuadrupedManipulatePlate.get_observations(self)

    def calculate_metrics(self) -> None:
        return UnitedQuadrupedManipulatePlate.calculate_metrics(self)

    def is_done(self):
        UnitedQuadrupedManipulatePlate.is_done(self)

        # Reset friction
        if self._dr_randomizer.randomize:
            if self._friction_reset_buf > self._friction_reset_interval:
                # Randomize the friciton value
                static_friction, dynamic_friction = self._generate_random_friction()
                self._default_material.set_dynamic_friction(dynamic_friction)
                self._default_material.set_static_friction(static_friction)
                self._friction_reset_buf = 0
            else:
                self._friction_reset_buf += 1

    def clear_robot_last_joint_velocities(self, indices):
        num_reset = len(indices)
        self.robot.last_joint_velcoties[indices, :] = torch.zeros(num_reset, 12, dtype=torch.float32, device=self._device)

    def reset_robot_joint_positions(self, indices):
        num_reset = len(indices)
        default_joint_positions = torch.tensor(self.robot.robot_description.init_joint_pos, dtype=torch.float32, device=self._device).repeat((num_reset,1))
        self.robot.set_joint_positions(default_joint_positions, indices, True)

    def reset_robot_joint_velocities(self, indices):
        num_reset = len(indices)
        zero_velocities = torch.zeros(num_reset, 20, dtype=torch.float32, device=self._device)
        self.robot.set_joint_velocities(zero_velocities, indices, True)

    def reset_goal_indicator_pose(self, goal_quaternions, indices):
        if not self.training:
            num_reset = len(indices)
            default_pose_indicator_positons = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32, device=self._device).repeat((num_reset, 1))
            self.pose_indicator.set_object_poses(positions=default_pose_indicator_positons, quaternions=goal_quaternions, indices=indices)

    def reset_plate_poses(self, indices):
        num_reset = len(indices)
        default_plate_positions = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float32, device=self._device).repeat((num_reset, 1))
        default_plate_quaternions = torch.tensor(self.robot.robot_description.default_quaternion, dtype=torch.float32, device=self._device).repeat((num_reset, 1))
        self._plate.set_object_poses(default_plate_positions, 
                                     quaternions=default_plate_quaternions, 
                                     indices=indices)

    def reset_plate_velocities(self, indices):
        num_reset = len(indices)
        zeros_velocities = torch.zeros((num_reset, 6), dtype=torch.float32, device=self._device)
        self._plate.set_object_velocities(zeros_velocities, indices)

    @property
    def quadruped_robot(self):
        return self.robot

    @property
    def plate(self):
        return self._plate

    def _generate_random_friction(self):
        rand_float = np.random.rand(2)

        rand_friction = (self._friciton_reset_range[1] - self._friciton_reset_range[0]) * rand_float + self._friciton_reset_range[0]

        return rand_friction

    @property
    def fixed_robot_base_positions(self):
        return torch.tensor(self.robot.robot_description.default_position, dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))

    @property
    def fixed_robot_base_quaternions(self):
        return torch.tensor(self.robot.robot_description.default_quaternion, dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))