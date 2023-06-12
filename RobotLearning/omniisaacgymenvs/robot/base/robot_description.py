class RobotDescriptionOmni:
    def __init__(self, robot_configs=None) -> None:
        self.__num_dof_per_module = 3

        self._config_names = [
            "robot_name",
            "arti_root_prim", # Prim path of the articulation root
            "usd_file", # usd file of the robot
            "control_frequency", # Determines after how many timesteps the control signal will be sent to the robot
            "control_mode",
            "num_modules", # Number of modules in your robot
            "module_prefix_list", # A list of prefix of each module
            "init_joint_pos", # Initial joint positions for the robot
            "default_position", # Default position of the base
            "default_quaternion", # Default orientation of the base
            "joint_kps", # Stiffness/Kp gain of joints
            "joint_kds", # Damping/kd gain of joints
            "velocity_limits", # Velocity limits (max velocity) for each dof joint
            "torque_limits", # Torque limit (max torque) for each dof joint
        ]

        '''
            Extendable configs are those can extended to all the modules in
            your robot;
        '''
        self._extendable_configs = [
            "joint_kps",
            "joint_kds",
            "velocity_limits",
            "torque_limits"
        ]

        self._default_values = {
            "robot_name": "module",
            "prim_path": "",
            "control_frequency": 30,
            "control_mode": "position",
            "default_position": [0, 0, 0],
            "default_quaternion": [1, 0, 0, 0],
            "joint_kps": [5,5,5],
            "joint_kds": [1,1,1]
        }
        
        self._robot_configs = robot_configs

        # Initialize config values
        for config_name in self._config_names:
            self._set_config_values(config_name)
        
        # Extend configs
        self._extend_configs()

        # Check consistency between prefix_list and 
        # num of modules
        assert self.num_modules == len(self.module_prefix_list), \
            "Inconsistency between prefix_list <{}> and num_modules <{}>".format(self.module_prefix_list, self.num_modules)

    def _set_config_values(self, config_name):
        '''
            (1) If the config is not found in input robot_configs:
            Set to default values if default values are available; Otherwise
            raise a error

            (2) If the config is found in robot configs, set the value to 
            private variables
        '''
        if config_name in self._robot_configs.keys():
            # Config exits
            self.__setattr__(config_name, self._robot_configs[config_name])
        else:
            # Config value not available
            assert config_name in self._default_values.keys(), \
                "Config <{}> is not set. ".format(config_name)
            self.__setattr__(config_name, self._default_values[config_name])

    def _extend_configs(self):
        '''
            Extend values that are extendable

            i.e. if length of the config value is 3, extend it to
            3 * num_modules
        '''
        for extendable_config in self._extendable_configs:
            if len(self.__getattribute__(extendable_config)) == self.__num_dof_per_module:
                extended = self.__getattribute__(extendable_config) * self.num_modules
                self.__setattr__(extendable_config, extended)
            else:
                assert len(self.extendable_config) == self.__num_dof_per_module * self.num_modules

    def __str__(self):
        info_str = ""

        for config_name in self._config_names:
            info_str += "Config: <{}>; Value: <{}> \n".format(config_name, self.__getattribute__(config_name))

        return info_str


