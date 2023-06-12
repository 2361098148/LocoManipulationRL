from abc import abstractmethod
# from tasks.base.rl_task import RLTask
import torch

from utils.math import transform_vectors, inverse_transform_vectors, rand_quaternions, inverse_rotate_orientations, rotate_orientations

from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_conjugate, quat_mul, quat_axis

import torch
import numpy as np

from robot.quadruped_robot import QuadrupedRobotOVOmni
from tasks.quadruped_pose_control_tasks.quadruped_pose_control import QuadrupedPoseControl
from tasks.base.rl_task import RLTask
from objects.pose_indicator import PoseIndicator

from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.replicator.isaac as dr

class QuadrupedPoseControlBehavior(QuadrupedPoseControl):
    def __init__(self, sim_config, name, env, offset=None) -> None:
        super().__init__(sim_config, name, env, offset)

    def set_up_scene(self, scene) -> None:
        self.robot_locomotion.init_omniverse_robot(self.default_zero_env_path)
        self._sim_config.apply_articulation_settings(self.robot_locomotion.robot_name, get_prim_at_path(self.robot_locomotion._omniverse_robot.prim_path), self._sim_config.parse_actor_config(self.robot_locomotion.robot_name))
        self.pose_indicator_loco.init_stage_object(self.default_zero_env_path, self._sim_config)
        
        RLTask.set_up_scene(self, scene)

        # Init robot views
        self.robot_locomotion.init_robot_views(scene)

        # Init pose indicator views
        self.pose_indicator_loco.init_object_view(scene)

        # Apply on startup domain randomization
        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)
            print("Applying on startup domain randomization...")

        print("Finished setting up scenes! ")
        return