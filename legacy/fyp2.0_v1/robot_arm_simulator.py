import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# 4-DOF ROBOT ARM SIMULATOR (FYP VERSION)
# Forward Kinematics + Simple 3D Visualization
# ============================================================

@dataclass
class JointState:
    base_yaw: float      # θ0 (rotation around Z)
    shoulder: float      # θ1
    elbow: float         # θ2
    wrist: float         # θ3


class RobotArm4DOF:
    def __init__(self):
        # -----------------------------
        # ARM LINK LENGTHS (meters)
        # -----------------------------
        self.L1 = 0.10   # base height
        self.L2 = 0.18   # upper arm
        self.L3 = 0.16   # forearm
        self.L4 = 0.08   # gripper offset

        # Joint limits (rad)
        self.limits = {
            "shoulder": (-np.pi/2, np.pi/2),
            "elbow": (0, np.pi),
            "wrist": (-np.pi/2, np.pi/2)
        }

        self.state = JointState(0, 0, 0, 0)

    # ============================================================
    # FORWARD KINEMATICS
    # ============================================================
    def forward_kinematics(self, state: JointState):
        """
        Returns joint positions in 3D space:
        base → shoulder → elbow → wrist → end-effector
        """

        θ0 = state.base_yaw
        θ1 = state.shoulder
        θ2 = state.elbow
        θ3 = state.wrist

        # Base
        base = np.array([0, 0, 0])

        # Shoulder position
        shoulder = np.array([
            0,
            0,
            self.L1
        ])

        # Planar projection after base yaw
        c0 = math.cos(θ0)
        s0 = math.sin(θ0)

        # Shoulder → Elbow
        x1 = self.L2 * math.cos(θ1)
        z1 = self.L2 * math.sin(θ1)

        elbow = np.array([
            c0 * x1,
            s0 * x1,
            shoulder[2] + z1
        ])

        # Elbow → Wrist
        x2 = self.L3 * math.cos(θ1 + θ2)
        z2 = self.L3 * math.sin(θ1 + θ2)

        wrist = np.array([
            elbow[0] + c0 * x2,
            elbow[1] + s0 * x2,
            elbow[2] + z2
        ])

        # Wrist → End Effector
        x3 = self.L4 * math.cos(θ1 + θ2 + θ3)
        z3 = self.L4 * math.sin(θ1 + θ2 + θ3)

        end_effector = np.array([
            wrist[0] + c0 * x3,
            wrist[1] + s0 * x3,
            wrist[2] + z3
        ])

        return base, shoulder, elbow, wrist, end_effector

    # ============================================================
    # INVERSE TARGETING (simple heuristic IK)
    # ============================================================
    def solve_ik(self, target):
        """
        Simple geometric IK (not full analytical solver)
        target: (x, y, z)
        """

        x, y, z = target

        # Base yaw
        θ0 = math.atan2(y, x)

        # planar distance
        r = math.sqrt(x**2 + y**2)
        z_eff = z - self.L1

        # clamp reach
        r = max(0.05, r)
        z_eff = max(0.05, z_eff)

        # law of cos approximation
        D = (r**2 + z_eff**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        D = np.clip(D, -1.0, 1.0)

        θ2 = math.acos(D)  # elbow

        θ1 = math.atan2(z_eff, r) - math.atan2(
            self.L3 * math.sin(θ2),
            self.L2 + self.L3 * math.cos(θ2)
        )

        θ3 = - (θ1 + θ2) * 0.5  # stabilize wrist

        return JointState(θ0, θ1, θ2, θ3)

    # ============================================================
    # UPDATE STATE
    # ============================================================
    def move_to(self, target_xyz):
        self.state = self.solve_ik(target_xyz)
        return self.forward_kinematics(self.state)

    # ============================================================
    # VISUALIZATION
    # ============================================================
    def plot(self, joints):
        base, shoulder, elbow, wrist, ee = joints

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = [base[0], shoulder[0], elbow[0], wrist[0], ee[0]]
        ys = [base[1], shoulder[1], elbow[1], wrist[1], ee[1]]
        zs = [base[2], shoulder[2], elbow[2], wrist[2], ee[2]]

        ax.plot(xs, ys, zs, marker='o', linewidth=3)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title("4-DOF Robot Arm Simulation (FYP)")

        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-0.4, 0.4)
        ax.set_zlim(0, 0.5)

        plt.show()


# ============================================================
# TEST RUN (standalone)
# ============================================================
if __name__ == "__main__":
    arm = RobotArm4DOF()

    # Example: book grasp point from your vision system
    target = (0.20, 0.10, 0.15)

    joints = arm.move_to(target)

    print("Joint State:")
    print(joints)

    arm.plot(joints)