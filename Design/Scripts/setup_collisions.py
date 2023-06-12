#Copy the following code and paste into the script editor of Isaac Sim after openning your mesh usd file
parent_prim_name = "/quadruped_robot"
disable_collision_link_names = [
    'prefix_base_motor_frame', 
    'prefix_dual_servo_shell', 
    'prefix_xm430_0', 
    'prefix_xm430_1', 
    'prefix_xm430_2',
    "prefix_tip_bearings"
] # If the name is started with prefix, it will be replaced with the actual name prefix to which it belongs

prefix_names = []
seed_link_name = "_xm430_0" # Link name used to infer prefix names

from sys import prefix
import omni.usd

def disable_collision(stage, parent_prim, link_name, set_disabled=True):
    full_prim_path = parent_prim + "/" + link_name + "/collisions/mesh_0"
    link_prim = stage.GetPrimAtPath(full_prim_path)
    collision_enable_ref = link_prim.GetAttribute("physics:collisionEnabled")
    collision_enable_ref.Set(not set_disabled)

stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    first_char_index = prim_path.find(seed_link_name)
    if first_char_index != -1:
        # There exist the substring in the path
        str_ahead = prim_path[0:first_char_index].split('/')
        prefix_name = str_ahead[-1]
        if prefix_name not in prefix_names:
            prefix_names.append(prefix_name)

# Convert the names to full real names
full_link_names = []
for dc_link_name in disable_collision_link_names:
    if "prefix" in dc_link_name:
        for prefix_name in prefix_names:
            full_link_names.append(dc_link_name.replace('prefix', prefix_name))
    else:
        full_link_names.append(dc_link_name)

for full_link_name in full_link_names:
    disable_collision(stage, parent_prim_name, full_link_name)
