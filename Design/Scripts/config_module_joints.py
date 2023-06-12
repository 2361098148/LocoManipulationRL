#Copy the following code and paste into the script editor of Isaac Sim after openning your mesh usd file

parent_prim_name = "/World/quadruped_robot"
close_chain_joint_name = "prefix_closed_chain_revolute"
non_driven_joints_keywords = [
    '_link4_to_link3',
    '_link1_to_link2',
    'closed_chain_revolute'
]
drive_params = [2, 0, 1000] # [max_force, damping, stiffness]
max_joint_velocity = 450

prefix_names = []
seed_link_name = "_xm430_0" # Link name used to infer prefix names

from sys import prefix
import omni.usd

def find_prim(stage, prim_name):
    result_prim = None
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_name in prim_path:
            result_prim = prim
    return result_prim

def find_prims_by_keyword(stage, keyword):
    result_prims = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if keyword in prim_path:
            result_prims.append(prim)
    return result_prims

def exclude_joint_from_articulations(stage, joint_name, exclude=True):
    # Find the joint
    joint_prim = find_prim(stage, joint_name)
    if not joint_prim:
        print("Joint {} not found".format(joint_name))
        exit(0)
    # If the joint prim is found, exclude it
    exclude_prim_ref = joint_prim.GetAttribute("physics:excludeFromArticulation")
    exclude_prim_ref.Set(exclude)
    print("Exclude {} from articulations. ".format(joint_name))

def set_drive_params(joint_prim, max_force, damping, stiffness):
    max_force_prim_ref = joint_prim.GetAttribute("drive:angular:physics:maxForce")
    damping_prim_ref = joint_prim.GetAttribute("drive:angular:physics:damping")
    stiff_prim_ref = joint_prim.GetAttribute("drive:angular:physics:stiffness")
    max_force_prim_ref.Set(max_force)
    damping_prim_ref.Set(damping)
    stiff_prim_ref.Set(stiffness)

def set_non_driven(stage):
    for keyword in non_driven_joints_keywords:
        prim_list = find_prims_by_keyword(stage, keyword)
        for non_driven_prim in prim_list:
            print("Setting {} to non-driven joint. ".format(str(non_driven_prim.GetPath()).split('/')[-1]))
            set_drive_params(non_driven_prim, 0, 0, 0)

def set_dof_drive_params(stage):
    dof_prims = find_prims_by_keyword(stage, "_dof")
    for dof_prim in dof_prims:
        print("Setting drive params of {} to: {}; max_vel: {}. ".format(str(dof_prim.GetPath()).split('/')[-1], drive_params, max_joint_velocity))
        # Set drive parameters
        set_drive_params(dof_prim, drive_params[0], drive_params[1], drive_params[2])
        # Set maximum joint_velocities
        max_vel_ref = dof_prim.GetAttribute("physxJoint:maxJointVelocity")
        max_vel_ref.Set(max_joint_velocity)

stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    first_char_index = prim_path.find(seed_link_name)
    if first_char_index != -1:
        # There exist the substring in the path
        str_ahead = prim_path[0:first_char_index].split('/')
        prefix_name = str_ahead[-1]
        if prefix_name not in prefix_names and "_" not in prefix_name:
            prefix_names.append(prefix_name)

# Exclude close joints from articulations
close_chain_joint_full_names = []
for prefix_name in prefix_names:
    close_chain_joint_full_names.append(close_chain_joint_name.replace('prefix', prefix_name))
for joint_full_name in close_chain_joint_full_names:
    exclude_joint_from_articulations(stage, joint_full_name)

# Set non-driven joints (max effort, stiffness, damping to 0)
set_non_driven(stage)

# Set dof driving parameters
set_dof_drive_params(stage)

