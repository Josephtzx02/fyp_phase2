"""
robot_world.py

Phase A

Digital Twin Scene Manager

Responsible ONLY for:

- Creating the 3D world
- Holding all scene objects
- Updating scene objects

No inverse kinematics.
No robot mathematics.
"""

import open3d as o3d
import numpy as np

class RobotWorld:

    def __init__(self):

        self.vis = o3d.visualization.Visualizer()

        self.window_created = False

        self.coordinate_frame = None

        self.floor = None

        self.robot_base = None

        self.arm_mount = None

        self.camera = None

        self.bookshelf = None

        self.book = None

        self.grasp = None

        self.approach_line = None

    # --- Create Open3D Window ---
    def create_world(self):

        self.vis.create_window(
            "Library Robot Simulator",
            width=1400,
            height=900
        )

        opt = self.vis.get_render_option()
        opt.background_color = np.array([1,1,1])

        self._create_coordinate_frame()

        self._create_robot_base()

        self._create_arm_mount()

        self._create_camera()

        self._create_bookshelf()
        
        self._setup_view()

        print("Robot World Created.")

    # --- Coordinate Frame ---
    def _create_coordinate_frame(self):

        self.coordinate_frame = \
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=80,
                origin=[0, 0, 0]
            )

        self.vis.add_geometry(self.coordinate_frame)

    # --- Floor ---
    def _create_floor(self):

        floor = o3d.geometry.TriangleMesh.create_box(
            width=1000,
            height=5,
            depth=800
        )

        floor.paint_uniform_color([0.88, 0.88, 0.88])

        floor.translate([-500, -5, 0])

        self.floor = floor

        self.vis.add_geometry(floor)

    # --- Robot Base ---
    def _create_robot_base(self):

        base = o3d.geometry.TriangleMesh.create_box(
            width=120,
            height=60, 
            depth=120
        )

        base.paint_uniform_color([0.20,0.40,0.90])

        base.translate([-60, 0, -60])

        self.robot_base = base

        self.vis.add_geometry(base)

    def _create_arm_mount(self):

        mount = o3d.geometry.TriangleMesh.create_cylinder(
            radius=18,
            height=220
        )

        mount.paint_uniform_color([0.70,0.70,0.70])

        self.arm_mount = mount

        self.vis.add_geometry(mount)

    # --- Camera ---
    def _create_camera(self):

        camera = o3d.geometry.TriangleMesh.create_box(
            width=35,
            height=25,
            depth=25
        )

        camera.paint_uniform_color([1, 0.20, 0.20])

        camera.translate([-17, 270, -0])

        self.camera = camera

        self.vis.add_geometry(camera)

    # --- Bookshelf ---
    def _create_bookshelf(self):

        shelf = o3d.geometry.TriangleMesh.create_box(
            width=800,
            height=450,
            depth=15
        )

        shelf.paint_uniform_color([0.65, 0.45, 0.25])

        shelf.translate([-400, 0, 600])

        self.bookshelf = shelf

        self.vis.add_geometry(shelf)

    def update_book(self, grasp_target):

        if grasp_target is None:
            return

        if self.book is not None:
            self.vis.remove_geometry(
                self.book,
                reset_bounding_box=False
            )

        width = grasp_target.width_mm
        height = grasp_target.height_mm
        thickness = grasp_target.thickness_mm

        # --- Open3D box ---
        # X = Width, Y = Height, Z = Thickness
        book = o3d.geometry.TriangleMesh.create_box(
            width=width,
            height=height,
            depth=thickness #might be swapped with width
        )

        book.paint_uniform_color([0.2,0.8,0.2])

        # --- Move box centre to origin ---
        book.translate([
            -width/2,
            -height/2,
            -thickness/2
        ])

        # --- Rotate around centre ---

        angle = np.deg2rad(
            grasp_target.angle_deg
        )

        R = book.get_rotation_matrix_from_xyz(
            (
                0,
                angle,
                0
            )
        )

        book.rotate(
            R,
            center=(0,0,0)
        )

        # Move to grasp point
        book.translate([

            grasp_target.x_mm,
            grasp_target.y_mm,
            grasp_target.z_mm
        ])

        self.book = book

        self.vis.add_geometry(book)

    def update_grasp(self, grasp_target):

        if self.grasp is not None:
            self.vis.remove_geometry(
                self.grasp,
                reset_bounding_box=False
            )

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=8
        )

        sphere.paint_uniform_color([1,0,0])

        sphere.translate([
            grasp_target.x_mm,
            grasp_target.y_mm,
            grasp_target.z_mm
        ])

        self.grasp = sphere

        self.vis.add_geometry(sphere)

    def update_approach(self, grasp_target):

        if self.approach_line is not None:

            self.vis.remove_geometry(
                self.approach_line,
                reset_bounding_box=False
            )

        line = o3d.geometry.LineSet()

        line.points = o3d.utility.Vector3dVector([

            [0,0,50],

            [

                grasp_target.x_mm,

                grasp_target.y_mm,

                grasp_target.z_mm

            ]

        ])

        line.lines = o3d.utility.Vector2iVector([

            [0,1]

        ])

        line.colors = o3d.utility.Vector3dVector([

            [0.2,0.6,1.0]

        ])

        self.approach_line = line

        self.vis.add_geometry(line)

    # --- Camera View ---
    def _setup_view(self):

        ctr = self.vis.get_view_control()

        # Slightly above and behind the robot
        ctr.set_front([0.0, 180, 350])

        # Look towards the bookshelf
        ctr.set_lookat([-0.55, 0.35, -0.75])

        # Y is vertical in our convention
        ctr.set_up([0, 1, 0])

        ctr.set_zoom(0.50)

    def update_scene(self, grasp_target):

        if grasp_target is None:
            return

        self.update_book(grasp_target)
        self.update_grasp(grasp_target)
        self.update_approach(grasp_target)

        self.update()

    # --- Update ---
    def update(self):

        self.vis.poll_events()

        self.vis.update_renderer()

    def destroy(self):

        self.vis.destroy_window()