class ModuleOmni(object):
    def __init__(self, 
                 module_prefix="a1",
                 joint_kp=5,
                 joint_kd=1,
                 max_torque=2,
                 ) -> None:
        self.module_prefix = module_prefix
        self.joint_kp = joint_kp
        self.joint_kd = joint_kd
        self.max_torque = max_torque