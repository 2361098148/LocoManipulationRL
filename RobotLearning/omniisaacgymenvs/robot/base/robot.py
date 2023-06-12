from typing import Optional
import numpy as np
import torch

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.torch.maths import unscale_transform

from pxr import PhysxSchema

class OmniverseRobot(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Module",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

    def set_rigid_body_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64/np.pi*180)
    
    def prepare_tip_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                if "fingertip_frame" in str(link_prim.GetPrimPath()):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)

class RobotOmni(object):
    def __init__(self, robot_description) -> None:
        self.num_envs = 1 # Default value
        self.device = "cuda" # Default to cuda
        self.time_interval = 0.033 # In milliseconds

        self.num_dof_per_module = 3
        
        self.robot_description = robot_description
        self.robot_name = self.robot_description.robot_name
        self.num_modules = self.robot_description.num_modules
        self.num_dofs = self.num_modules * self.num_dof_per_module
        
        self._omniverse_robot = None
        self._robot_articulation = None
        # Some auxilary prim views
        self._knee_view = None
        self._tip_view = None

        self._omni_dof_names = []
        self._omni_dof_indices = []

        # Limits
        self._dof_limits = []
        self._dof_vel_limits = []
        self._dof_torque_limits = []
        self._positions_lower = []
        self._positions_upper = []
        self._torque_lower = []
        self._torque_upper = []
        self._velocity_lower = []
        self._velocity_upper = []

        self._local_origin_positions = None
        self._tip_local_origin_positions = None

        # State buffers
        ## Joint states
        self.joint_positions = None
        self.joint_velocities = None
        self.last_joint_velcoties = None
        self.joint_accelerations = None
        ## Body states
        self.base_positions = None
        self.base_quaternions = None
        self.base_linear_velocities = None
        self.base_angular_velocities = None
        ## Tip states
        self.tip_positions = None
        self.last_tip_positions = None
        self.tip_velocities = None
        ## Knee states
        self.knee_positions = None


    def init_omniverse_robot(self, default_zero_env_path="/World/envs/env_0"):
        """
            Initialize the Omniverse Robot instance; Add the first Omniverse Robot to the stage
            for cloning;
        """
        prim_path = default_zero_env_path + "/" + self.robot_description.robot_name
        
        self._omniverse_robot = OmniverseRobot(prim_path,
                                               self.robot_description.robot_name,
                                               self.robot_description.usd_file,
                                               self.robot_description.default_position,
                                               self.robot_description.default_quaternion)
        
    def init_robot_views(self, scene):
        self.init_robot_articulation(scene)
        self.init_knee_view(scene)
        self.init_tip_view(scene)

    def init_robot_articulation(self, scene):
        """
            Create the robot articulation; Add the articulation to the scene;
        """
        arti_root_name = "/World/envs/.*/" + self.robot_description.robot_name + self.robot_description.arti_root_prim
        self._robot_articulation = ArticulationView(prim_paths_expr=arti_root_name, name=self.robot_name)
        scene.add(self._robot_articulation)

    def init_knee_view(self, scene):
        prefix_names = ""
        for i, module_prefix in enumerate(self.robot_description.module_prefix_list):
            if i == 0:
                prefix_names += module_prefix
            else:
                prefix_names += "|" + module_prefix
        # knee_prim_path_expr = "/World/envs/.*/" + self.robot_description.robot_name + "/.*/({})_link[2,3]($|_right)".format(prefix_names)
        knee_prim_path_expr = "/World/envs/.*/" + self.robot_description.robot_name + "/({})_link[2,3]($|_right)".format(prefix_names)
        
        self._knee_view = RigidPrimView(knee_prim_path_expr, self.robot_name + "_knee_view", reset_xform_properties=False)

        scene.add(self._knee_view)

    def init_tip_view(self, scene):
        prefix_names = ""
        for i, module_prefix in enumerate(self.robot_description.module_prefix_list):
            if i == 0:
                prefix_names += module_prefix
            else:
                prefix_names += "|" + module_prefix
        if self.robot_description.num_modules == 1:
            # tip_prim_path_expr = "/World/envs/.*/" + self.robot_description.robot_name + "/.*/{}_fingertip_frame".format(prefix_names)
            tip_prim_path_expr = "/World/envs/.*/" + self.robot_description.robot_name + "/{}_fingertip_frame".format(prefix_names)
        else:
            # tip_prim_path_expr = "/World/envs/.*/" + self.robot_description.robot_name + "/.*/({})_fingertip_frame".format(prefix_names)
            tip_prim_path_expr = "/World/envs/.*/" + self.robot_description.robot_name + "/({})_fingertip_frame".format(prefix_names)
        
        self._tip_view = RigidPrimView(tip_prim_path_expr, self.robot_name + "_tip_view", reset_xform_properties=False)
        
        scene.add(self._tip_view)

    def post_reset_robot(self, env_pos, num_envs, time_interval, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        self.time_interval = time_interval

        for prefix in self.robot_description.module_prefix_list:
            for i in range(1, self.num_dof_per_module+1):
                self._omni_dof_names.append("{}_dof{}".format(prefix, i))

        # Read the dof indices
        if self._omni_dof_names:
            # If dof_names has been defined
            for dof_name in self._omni_dof_names:
                self._omni_dof_indices.append(self._robot_articulation.get_dof_index(dof_name))

            # Re-arrange dof_name by dof_indices
            self._omni_dof_names = sorted(self._omni_dof_names, key=lambda dof_name: self._omni_dof_indices[self._omni_dof_names.index(dof_name)])
            # Sort dof_indices
            self._omni_dof_indices = sorted(self._omni_dof_indices)

            # Convert dof indices to torch tensor
            self._omni_dof_indices = torch.tensor(self._omni_dof_indices, device=self.device, dtype=torch.long)

        print("Dof names: ", self._omni_dof_names)
        print("Dof indices: ", self._omni_dof_indices)

        # Get dof limit
        self._dof_limits = self._robot_articulation.get_dof_limits().to(device=self.device)[0, self._omni_dof_indices]

        # Get dof vel/torque limit
        raw_vel_limit = self.robot_description.velocity_limits # is in order of [a_dof1, a_dof2, a_dof3, b_dof1, ...]
        raw_torque_limit = self.robot_description.torque_limits
        raw_vel_limit_dict = {}
        raw_torque_limit_dict = {}
        for prefix_i, prefix in enumerate(self.robot_description.module_prefix_list):
            for dof_i in range(self.num_dof_per_module):
                dof_name = "{}_dof{}".format(prefix, dof_i+1)
                raw_index = prefix_i * self.num_dof_per_module + dof_i

                raw_vel_limit_dict[dof_name] = [-raw_vel_limit[raw_index], raw_vel_limit[raw_index]]
                raw_torque_limit_dict[dof_name] = [-raw_torque_limit[raw_index], raw_torque_limit[raw_index]]

        # Mapping to the order of sorted self._dof_names
        for i, dof_i in enumerate(self._omni_dof_indices):
            this_vel_limit = raw_vel_limit_dict[self._omni_dof_names[i]]
            this_torque_limit = raw_torque_limit_dict[self._omni_dof_names[i]]

            self._dof_vel_limits.append(this_vel_limit)
            self._dof_torque_limits.append(this_torque_limit)

        # Convert torque limits into the form of two seperate limit list
        self._torque_lower = [i[0] for i in self._dof_torque_limits]
        self._torque_upper = [i[1] for i in self._dof_torque_limits]
        self._torque_lower = torch.tensor(self._torque_lower, dtype=torch.float32, device=self.device)
        self._torque_upper = torch.tensor(self._torque_upper, dtype=torch.float32, device=self.device)
        # Convert velocity limits into the form of two seperate limit list
        self._velocity_lower = [i[0] for i in self._dof_vel_limits]
        self._velocity_upper = [i[1] for i in self._dof_vel_limits]
        self._velocity_lower = torch.tensor(self._velocity_lower, dtype=torch.float32, device=self.device)
        self._velocity_upper = torch.tensor(self._velocity_upper, dtype=torch.float32, device=self.device)
        # No limit is set for position
        self._positions_upper = torch.tensor([np.pi], dtype=torch.float32, device=self.device).repeat(1, self.num_dofs)
        self._positions_lower = -1*self._positions_upper

        print("Velocity limits: ", self._dof_vel_limits)
        print("Torque limits: ", self._dof_torque_limits)
        print("Position lower: ", self._positions_lower)
        print("Position upper: ", self._positions_upper)
        print("Velocity lower: ", self._velocity_lower)
        print("Velocity upper: ", self._velocity_upper)
        print("Torque lower: ", self._torque_lower)
        print("Torque upper: ", self._torque_upper)

        self._local_origin_positions = env_pos

        for i in range(self.num_modules):
            if i == 0:
                self._tip_local_origin_positions = self._local_origin_positions
            else:
                self._tip_local_origin_positions = torch.cat((self._tip_local_origin_positions,
                                                              self._local_origin_positions), dim=1)
        self._tip_local_origin_positions = self._tip_local_origin_positions.view(self.num_envs, self.num_modules, 3)

        self.set_gains(self.robot_description.joint_kps, self.robot_description.joint_kds)
        self.set_max_torque(self.robot_description.torque_limits)
        self.set_control_mode(self.robot_description.control_mode)

        self.init_state_buffers(device, num_envs)

    def init_state_buffers(self, device, num_envs):
        # State buffers
        ## Joint states
        self.joint_positions = torch.tensor(self.robot_description.init_joint_pos, dtype=torch.float32, device=device).repeat(num_envs, 1)
        self.joint_velocities = torch.zeros((num_envs, self.num_dofs), dtype=torch.float32, device=device)
        self.last_joint_velcoties = torch.zeros((num_envs, self.num_dofs), dtype=torch.float32, device=device)
        self.joint_accelerations = torch.zeros((num_envs, self.num_dofs), dtype=torch.float32, device=device)
        ## Body states
        self.base_positions = torch.tensor(self.robot_description.default_position, dtype=torch.float32, device=device).repeat((num_envs,1))
        self.base_quaternions = torch.tensor(self.robot_description.default_quaternion, dtype=torch.float32, device=device).repeat((num_envs,1))
        self.base_linear_velocities = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.base_angular_velocities = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        ## Tip states
        self.tip_positions = torch.zeros((num_envs, self.num_modules, 3), dtype=torch.float32, device=device)
        self.tip_velocities = torch.zeros((num_envs, self.num_modules, 3), dtype=torch.float32, device=device)
        ## Knee states
        self.knee_positions = torch.zeros((num_envs, self.num_modules*2, 3), dtype=torch.float32, device=device)

    def update_all_states(self):
        self.update_joint_states()
        self.update_base_poses()
        self.update_base_velocities()
        self.update_tip_states()
        self.update_knee_positions()

    def update_joint_states(self):
        '''
            Update the joint positions, velocities, and accelerations buffer
            i.e. self.joint_positions, self.joint_velocities, self.joint_accelerations
        '''
        self.joint_positions = self._robot_articulation.get_joint_positions(joint_indices=self._omni_dof_indices, clone=False)
        self.joint_velocities = self._robot_articulation.get_joint_velocities(joint_indices=self._omni_dof_indices, clone=False)
        self.joint_accelerations = (self.joint_velocities - self.last_joint_velcoties)/self.time_interval
        self.last_joint_velcoties = self.joint_velocities.clone()

    def update_base_poses(self):
        base_position_world, self.base_quaternions = self._robot_articulation.get_world_poses(clone=False)
        self.base_positions = base_position_world - self._local_origin_positions

    def update_base_velocities(self):
        '''
            Update the base linear velocity and angular velocity buffers
            i.e. self.base_angular_vels and self.base_linear_vels
        '''
        self.base_angular_velocities = self._robot_articulation.get_angular_velocities(clone=False)
        self.base_linear_velocities = self._robot_articulation.get_linear_velocities(clone=False)

    def update_tip_states(self):
        # update tip positions
        tip_positions, _ = self._tip_view.get_world_poses()
        # Transform to shape (num_envs, num_modules, 3)
        tip_positions_world = tip_positions.view(self.num_envs, self.num_modules, 3)
        self.tip_positions = tip_positions_world - self._tip_local_origin_positions

        self.tip_velocities = self._tip_view.get_linear_velocities()

    def update_knee_positions(self):
        '''
            Return the world position of the knees in shape of
            (num_envs, 8, 3)
        '''
        raw_positions = self._knee_view.get_world_poses(clone=False)[0].view(self.num_envs, self.num_modules*2, 3)
        local_origins = self._local_origin_positions.repeat(1, self.num_modules*2).view(self.num_envs, self.num_modules*2, 3)
        self.knee_positions = raw_positions - local_origins

    def set_control_mode(self, control_mode):
        """
            Supported Mode Names:
                (1) position
                (2) velocity
                (3) effort
        """
        assert control_mode in ["position", "velocity", "effort"], \
            "Invalid control mode name"
        self.robot_description.control_mode = control_mode
        return self._robot_articulation.switch_control_mode(control_mode, joint_indices=self._omni_dof_indices)

    def set_gains(self, joint_kps, joint_kds):
        """
            Args:
                joint_kps: an array of shape (num_dof) specifying the stiffness of each dof
                joint_kds: an array of shape (num_dof) specifying the damping of each dof

            Note: Joint kps and kds should match the name order of the dofs
        """
        joint_kps = torch.tensor(joint_kps, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        joint_kds = torch.tensor(joint_kds, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        return self._robot_articulation.set_gains(joint_kps, joint_kds, joint_indices=self._omni_dof_indices)

    def set_max_torque(self, max_torque):
        """
            Args:
                max_torque: an array of shape (num_dof) specifying the maximum torque of each dof

            Note: Maximum torque for each dof should match the name order of the dofs
        """
        max_efforts = torch.tensor(max_torque, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        return self._robot_articulation.set_max_efforts(max_efforts, joint_indices=self._omni_dof_indices)

    def set_joint_positions(self, positions, indices=None, full_joint_indices=False):
        """
            Args:
                positions: the positions of shape (num_envs, num_targets)
                indices: the env indices to be set
                full_joint_indices: whether or not the target positions include passive joints
        """
        if full_joint_indices:
            joint_indices = None
        else:
            joint_indices = self._omni_dof_indices
        self._robot_articulation.set_joint_positions(positions, indices=indices, joint_indices=joint_indices)

    def set_joint_position_targets(self, positions, indices=None, full_joint_indices=False):
        """
            Args:
                positions: the target positions of shape (num_envs, num_targets)
                indices: the env indices to be set
                full_joint_indices: whether or not the target positions include passive joints
        """
        if full_joint_indices:
            joint_indices = None
        else:
            joint_indices = self._omni_dof_indices
        self._robot_articulation.set_joint_position_targets(positions, indices=indices, joint_indices=joint_indices)

    def set_joint_velocities(self, velocities, indices=None, full_joint_indices=False):
        """
            Args:
                velocities: the velocities of shape (num_envs, num_targets)
                indices: the env indices to be set
                full_joint_indices: whether or not the target velocities include passive joints
        """
        if full_joint_indices:
            joint_indices = None
        else:
            joint_indices = self._omni_dof_indices
        self._robot_articulation.set_joint_velocities(velocities, indices=indices, joint_indices=joint_indices)

    def set_joint_velocity_targets(self, velocities, indices=None, full_joint_indices=False):
        """
            Args:
                velocities: the target velocities of shape (num_envs, num_targets)
                indices: the env indices to be set
                full_joint_indices: whether or not the target velocities include passive joints
        """
        if full_joint_indices:
            joint_indices = None
        else:
            joint_indices = self._omni_dof_indices
        self._robot_articulation.set_joint_velocity_targets(velocities, indices=indices, joint_indices=joint_indices)

    def set_robot_pose(self, positions=None, quaternions=None, indices=None):
        '''
            Set robot positions and orientations in their local frames
        '''
        # Calculate global positions
        global_positions = None
        if positions is not None:
            if indices is None:
                global_positions = positions + self._local_origin_positions
            else:
                # indices is specified
                global_positions = positions + self._local_origin_positions[indices, :]

        self._robot_articulation.set_world_poses(global_positions, quaternions, indices)

    def set_robot_velocities(self, velocities, indices=None):
        return self._robot_articulation.set_velocities(velocities, indices=indices)

    def reset_robot_pose(self, indices):
        reset_positions = torch.tensor(self.robot_description.default_position, dtype=torch.float32, device=self.device).repeat((len(indices),1))
        reset_quaternions = torch.tensor(self.robot_description.default_quaternion, dtype=torch.float32, device=self.device).repeat((len(indices),1))
        return self.set_robot_pose(reset_positions, reset_quaternions, indices)

    def reset_robot_joint_positions(self, indices):
        reset_dof_positions = torch.tensor(self.robot_description.init_joint_pos, dtype=torch.float32, device=self.device).repeat(len(indices), 1)
        return self.set_joint_positions(reset_dof_positions, indices=indices, full_joint_indices=True)

    def zero_robot_joint_velocities(self, indices):
        zero_dof_velocities = torch.zeros(len(indices), len(self._omni_dof_names)+2*self.num_modules)
        self.set_joint_velocities(zero_dof_velocities, indices=indices, full_joint_indices=True)

    def zero_robot_velocities(self, indices):
        zero_velocities = torch.zeros(len(indices), 6, device=self.device, dtype=torch.float32)
        self.set_robot_velocities(zero_velocities, indices)

    def take_action(self, actions):
        # Set the maximum torque according to the current joint velocities
        # max_torques = self.get_max_torque(self.joint_velocities)
        # self._robot_articulation.set_max_efforts(max_torques, joint_indices=self._omni_dof_indices)

        if self.robot_description.control_mode == 'position':
            unscaled_positions = unscale_transform(actions, self._positions_lower, self._positions_upper)
            self._robot_articulation.set_joint_position_targets(unscaled_positions, joint_indices=self._omni_dof_indices)
        elif self.robot_description.control_mode == 'velocity':
            unscaled_velocities = unscale_transform(actions, self._velocity_lower, self._velocity_upper)
            self._robot_articulation.set_joint_velocity_targets(unscaled_velocities, joint_indices=self._omni_dof_indices)
        elif self.robot_description.control_mode == 'effort':
            efforts = torch.zeros((self.num_envs, self._robot_articulation.num_dof), dtype=torch.float32, device=self.device)
            unscaled_efforts = unscale_transform(actions, self._torque_lower, self._torque_upper)    
            efforts[:, self._omni_dof_indices] = unscaled_efforts
            self._robot_articulation.set_joint_efforts(efforts)
        else:
            raise AttributeError()

    def get_max_torque(self, speed):
        '''
            Get the maximum torque according to current joint speed;
            The fitted speed-max_torque curve is quadratic;

            Shape of speed: (num_envs, num_dofs_per_env)
            Shape of return: (num_envs, num_dofs_per_env)
        '''
        # If the speed is lower than min_speed,
        # the max torque will be set
        min_speed = 1.0
        max_speed = 7.0
        max_torque = 2.0
        max_torque_filter = max_torque*torch.ones_like(speed, dtype=torch.float32, device=self.device)
        max_speed_filter = torch.zeros_like(speed, dtype=torch.float32, device=self.device)

        # Get the absolute value of speeds (no negative values)
        abs_speed = torch.abs(speed)

        fitted_max_torque = -0.01229*torch.square(abs_speed + 7.756) + 2.922

        # To check if there's any speed lower than min_speed
        filtered_max_torque = torch.where(abs_speed < min_speed, max_torque_filter, fitted_max_torque)

        # To check if there's large speed
        filtered_max_speed = torch.where(abs_speed >= max_speed, max_speed_filter, filtered_max_torque)

        return filtered_max_speed

    @property
    def robots(self):
        return self._robot_articulation

    @property
    def knees(self):
        return self._knee_view
    
    @property
    def tips(self):
        return self._tip_view