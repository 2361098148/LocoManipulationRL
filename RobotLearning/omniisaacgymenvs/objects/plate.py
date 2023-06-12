import os

from objects.base.rigid_object import RigidObjectOmni
from utils.path import *

class Plate50cmOmni(RigidObjectOmni):
    def __init__(self) -> None:
        object_name = "plate"
        # object_usd_path = "/home/bionicdl/Downloads/plate_40cm/plate/plate.usd" # os.path.join(root_ws_dir, "Design", "ObjectUSD", "plate_50cm.usd")
        object_usd_path =  os.path.join(root_ws_dir, "Design", "ObjectUSD", "plate", "plate.usd")
        object_prim_name = "/plate"
        super().__init__(object_name, object_usd_path, object_prim_name)