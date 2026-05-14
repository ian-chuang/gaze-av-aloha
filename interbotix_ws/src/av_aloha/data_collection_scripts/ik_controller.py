import numpy as np
import pyroki as pk
from yourdfpy import URDF
import sys
sys.path.append("/home/devi/giava/pyroki/examples")
from pyroki_snippets._solve_ik_with_multiple_targets import solve_ik_with_multiple_targets

class BimanualIKController:
    def __init__(self, urdf_path, target_link_names):
        urdf = URDF.load(urdf_path)
        self.robot = pk.Robot.from_urdf(urdf)
        self.target_link_names = target_link_names
        self.q = np.zeros(self.robot.joints.num_actuated_joints)

    def solve(self, left_pose, right_pose):
        target_positions = np.array([
            left_pose.translation,
            right_pose.translation
        ])

        target_wxyzs = np.array([
            left_pose.rotation.wxyz,
            right_pose.rotation.wxyz
        ])

        self.q = _solve_ik_with_multiple_targets(
            self.robot,
            ["leftgripper_base", "rightgripper_base"],
            target_wxyzs,
            target_positions,
            q_prev=self.q,
            smoothness_weight=0.1,
        )

        return self.q