"""
Robot Arm Simulator

Receives GraspTarget objects.

Computes IK.

Updates RobotArm.

Displays everything in Open3D.
"""

import open3d as o3d
import numpy as np

from robot_arm import RobotArm
from inverse_kinematics import InverseKinematics

class Simulator:

    def __init__(self):

        self.vis = o3d.visualization.Visualizer()

        self.vis.create_window(
            window_name="Book Retrieval Simulator",
            width=900, #1280
            height=650 #720
        )

        self.robot = RobotArm()

        self.robot.attach_visualizer(
            self.vis
        )

        self.ik = InverseKinematics()

        self.book = None

        self.target = None

        self.coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=100
        )

        self.vis.add_geometry(
            self.coordinate
        )

        self.create_bookshelf()

        self.robot.create_meshes()

        self.setup_camera()

    def create_bookshelf(self):

        shelf = o3d.geometry.TriangleMesh.create_box(
            width=700,
            height=350,
            depth=20
        )

        shelf.paint_uniform_color(
            [0.65,0.45,0.20]
        )

        shelf.translate([
            -350,
            0,
            550
        ])

        self.vis.add_geometry(
            shelf
        )

    def setup_camera(self):

        ctr = self.vis.get_view_control()

        ctr.set_front([
            -0.6,
            0.3,
            -0.7
        ])

        ctr.set_up([
            0,
            1,
            0
        ])

        ctr.set_lookat([
            0,
            180,
            350
        ])

        ctr.set_zoom(
            0.65
        )

    def update_target(self, grasp_target):

        if grasp_target is None:
            return

        angles = self.ik.solve(
            grasp_target
        )

        self.robot.set_angles(
            *angles
        )

        self.robot.update_visualization()

        self.update_book(
            grasp_target
        )

        #self.setup_camera()

    def update_book(self, target):

        if self.book is not None:

            self.vis.remove_geometry(
                self.book,
                reset_bounding_box=False
            )

        book = o3d.geometry.TriangleMesh.create_box(
            width=target.thickness_mm,
            height=target.height_mm,
            depth=target.width_mm
        )

        angle = np.deg2rad(target.angle_deg)

        #angle = target.angle_deg

        R = book.get_rotation_matrix_from_xyz(
            (0, 0, angle)
        )

        book.rotate(
            R,
            center=book.get_center()
        )

        book.paint_uniform_color([
            0.2,
            0.8,
            0.2
        ])

        book.translate([
            target.x_mm - target.thickness_mm/2,
            target.y_mm - target.height_mm/2,
            target.z_mm - target.width_mm/2
        ])

        self.book = book

        self.vis.add_geometry(
            book,
            reset_bounding_box=False
        )

    def update(self):
        
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):

        while True:
            self.update()

    def destroy(self):
        
        self.vis.destroy_window()