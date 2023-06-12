# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def initialize_task(config, env, init_sim=True):
    from .config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)

    from tasks.quadruped_pose_control_tasks.united_quadruped_pose_control_omni import UnitedQuadrupedPoseControlOmni
    from tasks.quadruped_manipulate_plate.united_quadruped_manipulate_plate_omni import UnitedQuadrupedManipulatePlateOmni
    from tasks.quadruped_manipulate_plate.united_quadruped_manipulate_plate_inverted_omni import UnitedQuadrupedManipulatePlateInvertedOmni
    
    from tasks.joint_train_locomanipulation.joint_locomanipulation import JointLocomanipulation
    from tasks.joint_train_locomanipulation.joint_locomanipulation_vertical import JointLocomanipulationVertical
    from tasks.quadruped_pose_control_tasks.quadruped_pose_control import QuadrupedPoseControl
    from tasks.quadruped_manipulate_plate.quadruped_manipulate_plate import QuadrupedManipulatePlate
    from tasks.quadruped_pose_control_tasks.quadruped_pose_control_custom_controller import QuadrupedPoseControlCustomController
    from tasks.quadruped_manipulate_plate.quadruped_manipulate_plate_custom_controller import QuadrupedManipulatePlateCustomController

    from tasks.quadruped_pose_control_tasks.quadruped_pose_control_vertical import QuadrupedPoseControlVertical
    from tasks.quadruped_manipulate_plate.quadruped_manipulate_plate_vertical import QuadrupedManipulatePlateVertical

    from tasks.load_experiment.free_fall_load_experiment_omni import FreeFallLoadExperimentOmni
    from tasks.load_experiment.vel_sin_load_experiment_omni import VelSinLoadExperimentOmni
    from tasks.load_experiment.pos_sin_load_experiment_omni import PosSinLoadExperimentOmni

    from tasks.quadruped_pose_control_tasks.quadruped_pose_control_custom_controller_dr import QuadrupedPoseControlCustomControllerDR
    from tasks.quadruped_pose_control_tasks.replay_real_quadruped_motions import RealQuadrupedMotionReplay

    from tasks.quadruped_manipulate_plate.quadruped_manipulate_plate_position_control import QuadrupedManipulatePlatePositionControl
    from tasks.quadruped_pose_control_tasks.quadruped_pose_control_position_control import QuadrupedPoseControlPositionControl
    from tasks.joint_train_locomanipulation.joint_locomanipulation_position_control import JointLocomanipulationPositionControl

    # Mappings from strings to environments
    task_map = {
        "UnitedQuadrupedPoseControlOmni": UnitedQuadrupedPoseControlOmni,
        "UnitedQuadrupedManipulatePlateOmni": UnitedQuadrupedManipulatePlateOmni,
        "UnitedQuadrupedManipulatePlateInvertedOmni": UnitedQuadrupedManipulatePlateInvertedOmni,

        "JointLocomanipulation": JointLocomanipulation,
        "QuadrupedPoseControl": QuadrupedPoseControl,
        "QuadrupedManipulatePlate": QuadrupedManipulatePlate,
        "QuadrupedPoseControlCustomController": QuadrupedPoseControlCustomController,
        "QuadrupedManipulatePlateCustomController": QuadrupedManipulatePlateCustomController,

        "QuadrupedPoseControlVertical": QuadrupedPoseControlVertical,
        "QuadrupedManipulatePlateVertical": QuadrupedManipulatePlateVertical,
        "JointLocomanipulationVertical": JointLocomanipulationVertical,

        "FreeFallLoadExperimentOmni": FreeFallLoadExperimentOmni,
        "VelSinLoadExperimentOmni": VelSinLoadExperimentOmni,
        "PosSinLoadExperimentOmni": PosSinLoadExperimentOmni,

        "QuadrupedPoseControlCustomControllerDR": QuadrupedPoseControlCustomControllerDR,
        "RealQuadrupedMotionReplay": RealQuadrupedMotionReplay,

        "QuadrupedManipulatePlatePositionControl": QuadrupedManipulatePlatePositionControl,
        "QuadrupedPoseControlPositionControl": QuadrupedPoseControlPositionControl,
        "JointLocomanipulationPositionControl": JointLocomanipulationPositionControl,
    }

    cfg = sim_config.config
    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    return task