import os

from objects.base.rigid_object import RigidObjectOmni
from utils.path import *

class PoseIndicator(RigidObjectOmni):
    def __init__(self, object_name="pose_indicator", object_prim_name="/pose_indicator") -> None:
        object_usd_path = os.path.join(root_ws_dir, "Design", "ObjectUSD", "pose_indicator.usd")
        super().__init__(object_name, object_usd_path, object_prim_name)