"""
Simple 4-DOF Robot Arm

Robot Coordinate System

Y = Height
Z = Forward Reach
X = AGV Left/Right
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d

BASE_X = 0.0
BASE_Y = 300.0
BASE_Z = 0.0

@dataclass
class Joint:
    
    x: float
    y: float
    z: float

class RobotArm:

    def __init__(self):

        # Must match with inverse_kinematics.py
        self.base_height = 160.0 # 220
        self.link1 = 220.0 # 260
        self.link2 = 180.0 # 240 
        self.gripper = 80.0 # 100
        
        # Joint angles
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.theta3 = 0.0
        self.theta4 = 0.0

        self.gripper_left = None
        self.gripper_right = None
        self.gripper_open = True

        self.book_thickness = 30.0 # Default to avoid startup crash

        self.base_x = 0.0 # Simulated AGV lateral position (mm)

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

    def set_wrist_rotation(self, angle):
        self.theta4 = angle

    def set_base_x(self, x):
        self.base_x = x

    def forward_kinematics(self):
        """
        Returns every joint position.
        Shoulder -> Elbow -> Wrist -> Gripper
        """

        t1 = np.deg2rad(self.theta1)
        t2 = np.deg2rad(self.theta2)
        t3 = np.deg2rad(self.theta3)

        base = Joint(
            self.base_x,
            BASE_Y,
            BASE_Z
        )

        shoulder = Joint(
            base.x,
            base.y + self.base_height,
            base.z
        )

        elbow = Joint(
            base.x,
            shoulder.y + self.link1 * np.sin(t1),
            shoulder.z + self.link1 * np.cos(t1)
        )

        wrist = Joint(
            base.x, 
            elbow.y + self.link2 * np.sin(t1 + t2),
            elbow.z + self.link2 * np.cos(t1 + t2)
        )

        gripper = Joint(
            base.x, 
            wrist.y + self.gripper * np.sin(t1 + t2 + t3),
            wrist.z + self.gripper * np.cos(t1 + t2 + t3)
        )

        return {

            "base": base,
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

        self.create_gripper(
            joints["gripper"]
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

    def create_gripper(self, tip):

        if self.gripper_open:
            gap = self.book_thickness / 2 + 5
        else:
            gap = max(
                self.book_thickness / 2,
                2
            )

        wrist_width = 60
        wrist_height = 10
        wrist_depth = 10

        finger_width = 4
        finger_height = 8
        finger_length = 30

        # Wrist Body
        wrist = o3d.geometry.TriangleMesh.create_box(
            width=wrist_width,
            height=wrist_height,
            depth=wrist_depth
        )

        wrist.paint_uniform_color([0.45,0.45,0.45])  

        wrist.translate([
            -wrist_width / 2,
            -wrist_height / 2,
            -finger_length / 2
        ])      

        # Fingers
        left = o3d.geometry.TriangleMesh.create_box(
            width=finger_width,
            height=finger_height,
            depth=finger_length
        )

        right = o3d.geometry.TriangleMesh.create_box(
            width=finger_width,
            height=finger_height,
            depth=finger_length
        )

        left.paint_uniform_color([0.2,0.2,0.2])
        right.paint_uniform_color([0.2,0.2,0.2])

        left.translate([
            -gap - finger_width / 2,
            -finger_height / 2,
            -wrist_depth / 2
        ])

        right.translate([
            gap - finger_width / 2,
            -finger_height / 2,
            -wrist_depth / 2
        ])

        R = wrist.get_rotation_matrix_from_xyz((
            0,
            0,
            np.deg2rad(self.theta4)
        ))

        wrist.rotate(
            R,
            center=(0,0,0)
        )

        left.rotate(
            R,
            center=(0,0,0)
        )

        right.rotate(
            R,
            center=(0,0,0)
        )

        # Move to Robot Tip
        translation = np.array([
            tip.x,
            tip.y,
            tip.z
        ])

        wrist.translate(translation)
        left.translate(translation)
        right.translate(translation)

        self.vis.add_geometry(
            wrist,
            reset_bounding_box=False
        )

        self.vis.add_geometry(
            left,
            reset_bounding_box=False
        )

        self.vis.add_geometry(
            right,
            reset_bounding_box=False
        )

        self.link_meshes.append(wrist)
        self.link_meshes.append(left)
        self.link_meshes.append(right)

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
