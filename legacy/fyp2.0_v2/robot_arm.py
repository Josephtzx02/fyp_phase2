"""
Simple 4-DOF Robot Arm

Robot Coordinate System

Y = Height
Z = Forward Reach
X = AGV Left/Right

The AGV is fixed during simulation.

Only the arm moves.
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d

@dataclass
class Joint:
    
    x: float
    y: float
    z: float

class RobotArm:

    def __init__(self):

        # Link lengths (mm)
        self.base_height = 220

        self.link1 = 220

        self.link2 = 180

        self.gripper = 80
        
        # Joint angles
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.theta3 = 0.0

        self.vis = None

        self.joint_meshes = []

        self.link_meshes = []

    def attach_visualizer(self, vis):

        self.vis = vis

    def set_angles(
        self,
        theta1,
        theta2,
        theta3
    ):

        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def forward_kinematics(self):

        """
        Returns every joint position.

        Shoulder
            ↓
        Elbow
            ↓
        Wrist
            ↓
        Gripper
        """

        t1 = np.deg2rad(self.theta1)
        t2 = np.deg2rad(self.theta2)
        t3 = np.deg2rad(self.theta3)

        shoulder = Joint(
            0,
            self.base_height,
            0
        )

        elbow = Joint(
            0,
            shoulder.y + self.link1 * np.sin(t1),
            shoulder.z + self.link1 * np.cos(t1)
        )

        wrist = Joint(
            0, 
            elbow.y + self.link2 * np.sin(t1 + t2),
            elbow.z + self.link2 * np.cos(t1 + t2)
        )

        gripper = Joint(
            0, 
            wrist.y + self.gripper * np.sin(t1 + t2 + t3),
            wrist.z + self.gripper * np.cos(t1 + t2 + t3)
        )

        return {

            "base": Joint(
                0,
                0,
                0
            ),

            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "gripper": gripper
        }

    def create_meshes(self):

        if self.vis is None:
            return

        joints = self.forward_kinematics()

        self.clear_meshes()

        # Create joint spheres

        for key in joints:

            p = joints[key]

            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=12
            )

            sphere.paint_uniform_color([
                0.85,
                0.15,
                0.15
            ])

            sphere.translate([
                p.x,
                p.y,
                p.z
            ])

            self.vis.add_geometry(
                sphere,
                reset_bounding_box=False
            )

            self.joint_meshes.append(
                sphere
            )

        self.create_links(
            joints
        )

    def create_links(self, joints):

        pairs = [
            ("base", "shoulder"),
            ("shoulder", "elbow"),
            ("elbow", "wrist"),
            ("wrist", "gripper")
        ]

        for a, b in pairs:

            p1 = joints[a]
            p2 = joints[b]

            line = o3d.geometry.LineSet()

            line.points = o3d.utility.Vector3dVector([
                [p1.x, p1.y, p1.z],
                [p2.x, p2.y, p2.z]
            ])

            line.lines = o3d.utility.Vector2iVector([
                [0,1]
            ])

            line.colors = o3d.utility.Vector3dVector([
                [0.2,0.4,1.0]
            ])

            self.vis.add_geometry(
                line,
                reset_bounding_box=False
            )

            self.link_meshes.append(
                line
            )

    def clear_meshes(self):

        if self.vis is None:
            return

        for mesh in self.joint_meshes:

            self.vis.remove_geometry(
                mesh,
                reset_bounding_box=False
            )

        for mesh in self.link_meshes:

            self.vis.remove_geometry(
                mesh,
                reset_bounding_box=False
            )

        self.joint_meshes.clear()

        self.link_meshes.clear()

    def update_visualization(self):

        self.create_meshes()