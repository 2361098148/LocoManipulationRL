import sys
import os
import yaml
from pxr import Usd

import omni.usd


'''
    The module accepts a yaml file used to modify the attributes
    of the usd file according to the parameters specified in that file

    The argument should be the relative path to the yaml file (relative to
    current script)
'''

class UsdModifier(object):
    def __init__(self, usd_file_path) -> None:
        self._usd_file_path = usd_file_path
        self._usd_stage = Usd.Stage.Open(self._usd_file_path)

    def exclude_joint_from_articulation(self, parent_prim, joint_name, exclude=True):
        joint_prim = self._usd_stage.GetPrimAtPath(parent_prim)

    def set_enable_link_collision(self, parent_prim, link_prim_path, enable=False):
        if parent_prim[-1] != '/':
            parent_prim += '/'
        link_prim = self._usd_stage.GetPrimAtPath(parent_prim + link_prim_path + '/collisions' + "/mesh_0")
        collision_enable_ref = link_prim.GetAttribute("physics:collisionEnabled")

if __name__ == "__main__":

    arg_len = len(sys.argv)
    if arg_len != 2:
        print("Usage: python modify_usd_attributes.py /path/to/file.yaml")
        sys.exit(0)
    
    attr_file_path = sys.argv[1]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    attr_file_path = os.path.join(current_dir, attr_file_path)
    attr_file_root = os.path.dirname(attr_file_path)

    # Check if the file exist
    if not os.path.isfile(attr_file_path):
        print("File {} does not exist! ".format(attr_file_path))
        exit(1)

    with open(attr_file_path, 'r') as attr_file:
        attrs = yaml.load(attr_file, Loader=yaml.SafeLoader)

        # Open the usd file needed to be modified
        usd_file = attrs['usd_file']
        usd_file_path = os.path.join(attr_file_root, usd_file)
        robot_usd = UsdModifier(usd_file_path)

        # Open the mesh reference usd file
        mesh_usd_file = attrs['mesh_usd_file']
        mesh_usd_file_path = os.path.join(attr_file_root, mesh_usd_file)
        mesh_usd = UsdModifier(mesh_usd_file_path)
        mesh_prim_path_prefix = attrs['mesh_prim_path_prefix']

        prefix_list = attrs['prefix_list']

        '''Enable collisions'''
        # Disable all collisions
        disable_collision_link_names = attrs['disable_collision_link_names']
        disable_names_full = []
        for disable_collision_link_name in disable_collision_link_names:
            if 'prefix' in disable_collision_link_name:
                for prefix_name in prefix_list:
                    disable_names_full.append(disable_collision_link_name.replace('prefix', prefix_name))
            else:
                disable_names_full.append(disable_collision_link_name)
        for disable_name in disable_names_full:
            mesh_usd.set_enable_link_collision(mesh_prim_path_prefix, disable_name)

        '''Exclude the close chain joint from articulation'''
        
