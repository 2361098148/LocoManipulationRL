from utils.path import *

from robot.base.robot import RobotOmni
from robot.base.robot_description import RobotDescriptionOmni

class QuadrupedRobotOmni(RobotOmni):
    def __init__(self) -> None:
        _robot_description_dict = {
            "robot_name": "Quadruped",
            "arti_root_prim": "/quadruped_robot/quadruped_frame", # Prim path of the articulation root
            "usd_file": os.path.join(root_ws_dir, "Design", "RobotUSD", "quadruped_robot_v2", "quadruped_robot_v2_instanceable_with_foot_sensor.usd"), # usd file of the robot
            "control_mode": "velocity",
            "num_modules": 4, # Number of modules in your robot
            "module_prefix_list": ['a1', 'a2', 'a3', 'a4'] , # A list of prefix of each module
            "init_joint_pos":
                [-1.2, 1.2, 1.2, -1.2, 
                 -1.22, -1.92, 
                  1.92, 1.22, 
                  1.92, 1.22, 
                 -1.22, -1.92, 
                  0.953, -0.953, 0.953, -0.953, 
                  0.953, -0.953, 0.953, -0.953],
            "default_position": [0, 0, 0.18], # Default position of the base
            "joint_kps": [0,0,0],
            "joint_kds": [100, 100, 100],
            "velocity_limits": [3.0, 3.0, 3.0],
            "torque_limits": [2, 2, 2], # Torque limit for each dof joint
        }

        robot_description = RobotDescriptionOmni(_robot_description_dict)

        super().__init__(robot_description)

class QuadrupedRobotOVOmni(RobotOmni):
    def __init__(self) -> None:
        _robot_description_dict = {
            "robot_name": "Quadruped",
            "arti_root_prim": "/base", # Prim path of the articulation root
            "usd_file": os.path.join(root_ws_dir, "Design", "RobotUSD", "overconstrained_v1", "overconstrained_v1.usd"), # usd file of the robot
            # os.path.join(root_ws_dir, "Design", "RobotUSD", "quadruped_robot_ov", "quadruped_robot_ov_test.usd"), 
            # os.path.join(root_ws_dir, "Design", "RobotUSD", "overconstrained_v1", "overconstrained_v1.usd"), # usd file of the robot
            "control_mode": "velocity",
            "num_modules": 4, # Number of modules in your robot
            "module_prefix_list": ['a1', 'a2', 'a3', 'a4'] , # A list of prefix of each module
            "init_joint_pos":
                [-1.2, 1.2, 1.2, -1.2, 
                 -1.22, -1.92, 
                  1.92, 1.22, 
                  1.92, 1.22, 
                 -1.22, -1.92, 
                  0.953, -0.953, 0.953, -0.953, 
                  0.953, -0.953, 0.953, -0.953],
                # [-1.2, 1.7, 1.2, -1.7, 
                #  -1.22, -1.92, 
                #   1.92, 1.22, 
                #   1.92, 1.22, 
                #  -1.22, -1.92, 
                #   0.953, -0.953, 0.953, -0.953, 
                #   0.953, -0.953, 0.953, -0.953],
            "default_position": [0, 0, 0.18], # Default position of the base
            "joint_kps": [0,0,0],
            "joint_kds": [100, 100, 100],
            "velocity_limits": [3.0, 3.0, 3.0],
            "torque_limits": [1.5, 1.5, 1.5], # Torque limit for each dof joint
        }

        robot_description = RobotDescriptionOmni(_robot_description_dict)

        super().__init__(robot_description)

class QuadrupedRobotVerticalOVOmni(RobotOmni):
    def __init__(self) -> None:
        _robot_description_dict = {
            "robot_name": "Quadruped",
            "arti_root_prim": "/base", # Prim path of the articulation root
            "usd_file": os.path.join(root_ws_dir, "Design", "RobotUSD", "quadruped_ov_vertical", "quadruped_ov_vertical_test.usd"), # usd file of the robot
            "control_mode": "velocity",
            "num_modules": 4, # Number of modules in your robot
            "module_prefix_list": ['a1', 'a2', 'a3', 'a4'] , # A list of prefix of each module
            "init_joint_pos":
                [0.0, 0.0, 0.0, 0.0, 
                 0.35, -0.35, 
                 0.35, -0.35,
                 0.35, -0.35,
                 0.35, -0.35, 
                 0.95, -0.95, 0.95, -0.95, 
                 0.95, -0.95, 0.95, -0.95],
                # [-1.2, 1.7, 1.2, -1.7, 
                #  -1.22, -1.92, 
                #   1.92, 1.22, 
                #   1.92, 1.22, 
                #  -1.22, -1.92, 
                #   0.953, -0.953, 0.953, -0.953, 
                #   0.953, -0.953, 0.953, -0.953],
            "default_position": [0, 0, 0.35], # Default position of the base
            "joint_kps": [0,0,0],
            "joint_kds": [100, 100, 100],
            "velocity_limits": [3.0, 3.0, 3.0],
            "torque_limits": [1.5, 1.5, 1.5], # Torque limit for each dof joint
        }

        robot_description = RobotDescriptionOmni(_robot_description_dict)

        super().__init__(robot_description)

class QuadrupedRobotFixedBaseOmni(RobotOmni):
    def __init__(self) -> None:
        _robot_description_dict = {
                "robot_name": "Quadruped",
                "arti_root_prim": "/quadruped_robot", # Prim path of the articulation root
                "usd_file": os.path.join(root_ws_dir, "Design", "RobotUSD", "quadruped_robot_v2", "quadruped_robot_v2_instanceable-fixed_base.usd"), # usd file of the robot
                "control_mode": "velocity",
                "num_modules": 4, # Number of modules in your robot
                "module_prefix_list": ['a1', 'a2', 'a3', 'a4'] , # A list of prefix of each module
                # "init_joint_pos": 
                #     [-1.2, 1.2, 1.2, -1.2, -1.0472033,
                #      -2.0943933, 2.094372, 1.0471789, 2.0944254, 1.0471523, 
                #      -1.0521997, -2.0950465, 1.3694302, -1.3694298, 1.3694334, 
                #      -1.3694332, 1.3695242, -1.3695248, 1.3645132, -1.3645164],
                "init_joint_pos":
                    [-1.2, 1.2, 1.2, -1.2, 
                    -1.22, -1.92, 
                    1.92, 1.22, 
                    1.92, 1.22, 
                    -1.22, -1.92, 
                    0.953, -0.953, 0.953, -0.953, 
                    0.953, -0.953, 0.953, -0.953],
                "default_position": [0, 0, 0.3], # Default position of the base
                # "default_quaternion": [0, 0, 1, 0],
                "default_quaternion": [1, 0, 0, 0],
                "joint_kps": [0,0,0],
                "joint_kds": [100, 100, 100],
                "velocity_limits": [3.0, 3.0, 3.0],
                "torque_limits": [2, 2, 2], # Torque limit for each dof joint
            }
        robot_description = RobotDescriptionOmni(_robot_description_dict)

        super().__init__(robot_description)

class QuadrupedRobotOVFixedBaseOmni(RobotOmni):
    def __init__(self) -> None:
        _robot_description_dict = {
                "robot_name": "QuadrupedFixed",
                "arti_root_prim": "/base", # Prim path of the articulation root
                "usd_file": os.path.join(root_ws_dir, "Design", "RobotUSD", "overconstrained_v1", "overconstrained_v1_fixed.usd"),
                # os.path.join(root_ws_dir, "Design", "RobotUSD", "overconstrained_v1", "overconstrained_v1_fixed.usd"),
                # os.path.join(root_ws_dir, "Design", "RobotUSD", "quadruped_robot_ov", "quadruped_robot_ov_fixed.usd"), # usd file of the robot
                "control_mode": "velocity",
                "num_modules": 4, # Number of modules in your robot
                "module_prefix_list": ['a1', 'a2', 'a3', 'a4'] , # A list of prefix of each module
                # "init_joint_pos": 
                #     [-1.2, 1.2, 1.2, -1.2, -1.0472033,
                #      -2.0943933, 2.094372, 1.0471789, 2.0944254, 1.0471523, 
                #      -1.0521997, -2.0950465, 1.3694302, -1.3694298, 1.3694334, 
                #      -1.3694332, 1.3695242, -1.3695248, 1.3645132, -1.3645164],
                "init_joint_pos":
                    [-1.2, 1.2, 1.2, -1.2, 
                    -1.22, -1.92, 
                    1.92, 1.22, 
                    1.92, 1.22, 
                    -1.22, -1.92, 
                    0.953, -0.953, 0.953, -0.953, 
                    0.953, -0.953, 0.953, -0.953],
                #     [-1.2, 1.7, 1.2, -1.7, 
                #  -1.22, -1.92, 
                #   1.92, 1.22, 
                #   1.92, 1.22, 
                #  -1.22, -1.92, 
                #   0.953, -0.953, 0.953, -0.953, 
                #   0.953, -0.953, 0.953, -0.953],
                "default_position": [0, 0, 0.3], # Default position of the base
                # "default_quaternion": [0, 0, 1, 0],
                "default_quaternion": [1, 0, 0, 0],
                "joint_kps": [0,0,0],
                "joint_kds": [100, 100, 100],
                "velocity_limits": [3.0, 3.0, 3.0],
                "torque_limits": [1.5, 1.5, 1.5], # Torque limit for each dof joint
            }
        robot_description = RobotDescriptionOmni(_robot_description_dict)

        super().__init__(robot_description)

class QuadrupedRobotVerticalOVFixedOmni(RobotOmni):
    def __init__(self) -> None:
        _robot_description_dict = {
            "robot_name": "QuadrupedFixed",
            "arti_root_prim": "/base", # Prim path of the articulation root
            "usd_file": os.path.join(root_ws_dir, "Design", "RobotUSD", "quadruped_ov_vertical", "quadruped_ov_vertical_fixed.usd"), # usd file of the robot
            "control_mode": "velocity",
            "num_modules": 4, # Number of modules in your robot
            "module_prefix_list": ['a1', 'a2', 'a3', 'a4'] , # A list of prefix of each module
            "init_joint_pos":
                [0.0, 0.0, 0.0, 0.0, 
                 0.35, -0.35, 
                 0.35, -0.35,
                 0.35, -0.35,
                 0.35, -0.35, 
                 0.95, -0.95, 0.95, -0.95, 
                 0.95, -0.95, 0.95, -0.95],
                # [-1.2, 1.7, 1.2, -1.7, 
                #  -1.22, -1.92, 
                #   1.92, 1.22, 
                #   1.92, 1.22, 
                #  -1.22, -1.92, 
                #   0.953, -0.953, 0.953, -0.953, 
                #   0.953, -0.953, 0.953, -0.953],
            "default_position": [0, 0, 0.0], # Default position of the base
            "joint_kps": [0,0,0],
            "joint_kds": [100, 100, 100],
            "velocity_limits": [3.0, 3.0, 3.0],
            "torque_limits": [1.5, 1.5, 1.5], # Torque limit for each dof joint
        }

        robot_description = RobotDescriptionOmni(_robot_description_dict)

        super().__init__(robot_description)
