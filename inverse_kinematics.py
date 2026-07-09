"""
Simple 4-DOF planar inverse kinematics.

Coordinate System

Y = Height
Z = Forward Reach
X = AGV Left/Right
"""

import math

BASE_Y = 300.0

# Joint Limits
SHOULDER_MIN = -120.0
SHOULDER_MAX = 120.0

ELBOW_MIN = -160.0
ELBOW_MAX = 160.0

WRIST_MIN = -180.0
WRIST_MAX = 180.0

class InverseKinematics:

    def __init__(self):
        # Must match robot_arm.py
        self.base_height = 160.0 # 220
        self.link1 = 220.0 # 260
        self.link2 = 180.0 # 240 
        self.gripper = 80.0 # 100
    
    def _within_limits(self, angles):
        t1, t2, t3 = angles

        return (
            SHOULDER_MIN <= t1 <= SHOULDER_MAX
            and
            ELBOW_MIN <= t2 <= ELBOW_MAX
            and
            WRIST_MIN <= t3 <= WRIST_MAX
        )

    def _score_solution(self, angles):

        """
        Higher score = preferred pose.

        Prefer

        • shoulder close to horizontal
        • elbow-down configuration
        • wrist away from limits
        """

        shoulder, elbow, wrist = angles

        score = 0.0

        # Prefer smaller shoulder magnitude
        score -= abs(shoulder)

        # Prefer elbow-down
        if elbow < 0:
            score += 100

        # Prefer wrist near zero
        score -= abs(wrist) * 0.1

        return score

    def solve(self, target):

        """
        Returns

        sucess, (theta1, theta2, theta3)
        """

        # Robot target
        y = target.grasp_y_mm
        z = target.grasp_z_mm

        # Move origin to shoulder
        y -= (BASE_Y + self.base_height)
        
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
            return False, None

        if r < min_reach:
            return False, None

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

        theta2_candidates = [
            math.acos(cos_theta2),
            -math.acos(cos_theta2)
        ]

        valid = []

        for theta2 in theta2_candidates:

            k1 = (
                self.link1
                +
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

            theta3 = -(
                theta1 +
                theta2
            )

            angles = (
                math.degrees(theta1),
                math.degrees(theta2),
                math.degrees(theta3)
            )

            if self._within_limits(
                angles
            ):

                valid.append(
                    angles
                )

        if len(valid) == 0:
            return False, None

        best = max(
            valid,
            key=self._score_solution
        )

        return True, best
