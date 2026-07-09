"""
Simple 4-DOF planar inverse kinematics.

Coordinate System

Y = Height
Z = Forward Reach
X = AGV Left/Right

The AGV is fixed during simulation.

Robot Base = (0,0)
"""

import math

class InverseKinematics:

    def __init__(self):

        # Must match robot_arm.py
        self.base_height = 220.0

        self.link1 = 220.0

        self.link2 = 180.0

        self.gripper = 80.0

    def solve(self, target):

        """
        target : GraspTarget

        Returns

        theta1
        theta2
        theta3

        in degrees.
        """

        # Robot target
        y = target.y_mm
        z = target.z_mm

        # Move origin to shoulder
        y -= self.base_height
        
        # Wrist position (Remove gripper length)
        wrist_y = y
        wrist_z = z - self.gripper
        
        # Distance
        r = math.sqrt(
            wrist_y**2 +
            wrist_z**2
        )
        
        # Reachability
        max_reach = self.link1 + self.link2

        min_reach = abs(self.link1 - self.link2)

        if r > max_reach:
            r = max_reach

        if r < min_reach:
            r = min_reach

        # Elbow
        cos_theta2 = (
            r**2 -
            self.link1**2 -
            self.link2**2
        ) / (
            2 *
            self.link1 *
            self.link2
        )

        cos_theta2 = max(
            -1.0,
            min(
                1.0,
                cos_theta2
            )
        )

        theta2 = math.acos(
            cos_theta2
        )

        # Shoulder
        k1 = (
            self.link1 +
            self.link2 *
            math.cos(theta2)
        )

        k2 = (
            self.link2 *
            math.sin(theta2)
        )

        theta1 = (
            math.atan2(
                wrist_y,
                wrist_z
            )
            -
            math.atan2(
                k2,
                k1
            )
        )

        
        # Wrist Pitch
        theta3 = -(
            theta1 +
            theta2
        )

        return (
            math.degrees(theta1),
            math.degrees(theta2),
            math.degrees(theta3)
        )