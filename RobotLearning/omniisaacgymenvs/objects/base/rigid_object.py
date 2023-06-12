from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView, XFormPrim, XFormPrimView
from omni.isaac.core.utils.stage import add_reference_to_stage

class RigidObjectOmni(object):
    def __init__(self, 
                 object_name, 
                 object_usd_path,
                 object_prim_name) -> None:
        self.object_name = object_name
        self.object_usd_path = object_usd_path
        self.object_prim_name = object_prim_name
        self.env_pos = None

        self.object_xform_prim = None
        self.object_view = None

    def init_stage_object(self, default_zero_env_path="/World/envs/env_0", sim_config=None):
        add_reference_to_stage(self.object_usd_path, 
                               default_zero_env_path + "/" + self.object_name)
        self.object_xform_prim = XFormPrim(prim_path=default_zero_env_path + self.object_prim_name,
                                           name=self.object_name)
        
        if sim_config is not None:
            sim_config.apply_articulation_settings(self.object_name, 
                                                   get_prim_at_path(self.object_xform_prim.prim_path), 
                                                   sim_config.parse_actor_config(self.object_name))

    def init_object_view(self, scene):
        self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/{}{}".format(self.object_name, self.object_prim_name), name=self.object_name)
        scene.add(self.object_view)
    
    def set_env_pos(self, env_pos):
        self.env_pos = env_pos

    def set_object_poses(self, positions=None, quaternions=None, indices=None):
        if positions is not None:
            # If positions are going to be reset
            if indices is not None:
                local_positions = self.env_pos[indices]
            else:
                local_positions = self.env_pos

            new_world_positions = local_positions + positions
            self.object_view.set_world_poses(new_world_positions, quaternions, indices=indices)
        else:
            self.object_view.set_world_poses(positions, quaternions, indices=indices)

    def set_object_velocities(self, velocities, indices=None):
        self.object_view.set_velocities(velocities, indices)

    def get_object_poses(self):
        world_position, quaternion = self.object_view.get_world_poses(clone=False)
        local_positions = world_position - self.env_pos
        return local_positions, quaternion

    def get_object_velocities(self):
        return self.object_view.get_linear_velocities(clone=False), self.object_view.get_angular_velocities(clone=False)