"""
Robot Arm Simulator

Receives GraspTarget objects.

Computes IK.

Updates RobotArm.

Displays everything in Open3D.
"""

import open3d as o3d
import numpy as np
import copy

from robot_arm import RobotArm
from inverse_kinematics import InverseKinematics

ROBOT_BASE_X = 0.0
ROBOT_BASE_Y = 0.0
ROBOT_BASE_Z = 0.0

STATE_IDLE = "IDLE"
STATE_APPROACH = "APPROACH"
STATE_INSERT = "INSERT"
STATE_GRASP = "GRASP"
STATE_PULL_OUT = "PULL_OUT"
STATE_LIFT = "LIFT"
STATE_HOME = "HOME"

BOOK_ON_SHELF = "ON_SHELF"
BOOK_ATTACHED = "ATTACHED"
BOOK_RETRIEVED = "RETRIEVED"

CAMERA_HEIGHT = 320.0

BOOKSHELF_DISTANCE = 700.0

SHELF_BOOK_HEIGHT = 700.0

HOME_THETA1 = -35
HOME_THETA2 = 70
HOME_THETA3 = -35

class Simulator:

    def __init__(self):

        self.vis = o3d.visualization.Visualizer()

        self.vis.create_window(
            window_name="Book Retrieval Simulator",
            width=960, #1280
            height=540, #720
            left=0,
            top=610
        )

        self.robot = RobotArm()

        self.robot.attach_visualizer(
            self.vis
        )

        self.ik = InverseKinematics()

        self.robot_state = STATE_IDLE

        self.current_target = None

        self.insert_target = None
        self.insert_speed = 3.0 # mm/frame
        self.pullout_target = None
        self.pullout_speed = self.insert_speed
        self.lift_target = None
        self.lift_speed = 3.0      # mm/frame
        self.lift_height = 80.0    # mm

        self.state_timer = 0

        self.book = None

        self.book_center = None
        self.grasp_point = None
        self.book_outline = None

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

        self.state_name = {
            STATE_IDLE: "Idle",
            STATE_APPROACH: "Approach",
            STATE_INSERT: "Insert",
            STATE_GRASP: "Grasp",
            STATE_PULL_OUT: "Pull Out",
            STATE_LIFT: "Lift",
            STATE_HOME: "Return Home"
        }

        self.book_state = BOOK_ON_SHELF

        self.current_angles = [HOME_THETA1, HOME_THETA2, HOME_THETA3]

        self.target_angles = self.current_angles.copy()

        self.animation_speed = 2.0

    def create_bookshelf(self):

        shelf = o3d.geometry.TriangleMesh.create_box(
            width=900,
            height=1200,
            depth=35
        )

        shelf.paint_uniform_color(
            [0.65,0.45,0.20]
        )

        shelf.translate([
            -350,
            0,
            BOOKSHELF_DISTANCE
        ])

        self.vis.add_geometry(shelf)

    def setup_camera(self):

        ctr = self.vis.get_view_control()

        ctr.set_front([
            -0.5, 
            0.1,
            -0.6
        ])

        ctr.set_up([
            0,
            1,
            0
        ])

        ctr.set_lookat([
            0,
            500,
            450
        ])

        ctr.set_zoom(
            0.65
        )

    def update_target(self, grasp_target):

        if grasp_target is None:
            return

        self.insert_target = None
        self.pullout_target = None
        self.lift_target = None

        self.state_timer = 0
        self.current_target = grasp_target
        self.robot_state = STATE_APPROACH

        self.robot.book_thickness = grasp_target.thickness_mm

        self.robot.set_wrist_rotation (grasp_target.angle_deg)

        self.update_book(grasp_target)
        self.update_markers(grasp_target)

        #self.setup_camera()

    def update_book(self, target):

        if self.book is not None:
            self.vis.remove_geometry(
                self.book,
                reset_bounding_box=False
            )

        if self.book_outline is not None:
            self.vis.remove_geometry(
                self.book_outline,
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

        if self.book_state == BOOK_ATTACHED:
            joints = self.robot.forward_kinematics()
            grip = joints["gripper"]
            draw_x = grip.x
            draw_y = grip.y
            draw_z = grip.z

        else:
            draw_x = target.book_x_mm
            draw_y = SHELF_BOOK_HEIGHT + target.book_y_mm
            draw_z = target.book_z_mm - target.width_mm / 2

        book.translate([
            draw_x - target.thickness_mm/2,
            draw_y - target.height_mm/2,
            draw_z
        ])       

        self.vis.add_geometry(
            book,
            reset_bounding_box=False
        )

        self.book = book

        #Rotated Book Outline

        w = target.thickness_mm
        h = target.height_mm
        d = target.width_mm

        # Box centre
        cx = draw_x
        cy = draw_y
        cz = draw_z + d/2

        # 8 vertices (local coordinates)

        vertices = np.array([

            [-w/2, -h/2, -d/2],
            [ w/2, -h/2, -d/2],
            [ w/2,  h/2, -d/2],
            [-w/2,  h/2, -d/2],

            [-w/2, -h/2,  d/2],
            [ w/2, -h/2,  d/2],
            [ w/2,  h/2,  d/2],
            [-w/2,  h/2,  d/2]

        ])

        # Rotate vertices
        vertices = vertices @ R.T

        # Translate to world
        vertices += np.array([
            cx,
            cy,
            cz
        ])

        lines = [

            [0,1],
            [1,2],
            [2,3],
            [3,0],

            [4,5],
            [5,6],
            [6,7],
            [7,4],

            [0,4],
            [1,5],
            [2,6],
            [3,7]

        ]

        outline = o3d.geometry.LineSet()

        outline.points = o3d.utility.Vector3dVector(
            vertices
        )

        outline.lines = o3d.utility.Vector2iVector(
            lines
        )

        outline.colors = o3d.utility.Vector3dVector(

            [[0.0,0.45,0.0]] * len(lines)

        )

        self.vis.add_geometry(
            outline,
            reset_bounding_box=False
        )

        self.book_outline = outline
    
    def update_markers(self, target):

        # Remove previous markers
        if self.book_center is not None:
            self.vis.remove_geometry(
                self.book_center,
                reset_bounding_box=False
            )

        if self.grasp_point is not None:
            self.vis.remove_geometry(
                self.grasp_point,
                reset_bounding_box=False
            )

        if self.book_state == BOOK_ATTACHED:

            joints = self.robot.forward_kinematics()

            grip = joints["gripper"]

            book_x = grip.x
            book_y = grip.y
            book_z = grip.z + target.width_mm / 2

            grasp_x = grip.x
            grasp_y = grip.y
            grasp_z = grip.z

            approach_x = grip.x
            approach_y = grip.y
            approach_z = grip.z - 70

        else:

            book_x = target.book_x_mm
            book_y = SHELF_BOOK_HEIGHT + target.book_y_mm
            book_z = target.book_z_mm

            grasp_x = target.grasp_x_mm
            grasp_y = SHELF_BOOK_HEIGHT + target.grasp_y_mm
            grasp_z = target.grasp_z_mm

            approach_x = target.approach_x_mm
            approach_y = SHELF_BOOK_HEIGHT + target.approach_y_mm
            approach_z = target.approach_z_mm

        # Book Centre (Blue)
        centre = o3d.geometry.TriangleMesh.create_cylinder(
            radius=4,
            height=target.thickness_mm + 2
        )

        centre.paint_uniform_color([
            0,
            0,
            1
        ])

        R = centre.get_rotation_matrix_from_xyz((
            0,
            np.pi/2,
            0
        ))

        centre.rotate(R)

        centre.translate([
            book_x,
            book_y,
            book_z
        ])

        self.vis.add_geometry(
            centre,
            reset_bounding_box=False
        )

        self.book_center = centre

        # Grasp Point (Red)
        grasp = o3d.geometry.TriangleMesh.create_sphere(
            radius=8
        )

        grasp.paint_uniform_color([
            1,
            0,
            0
        ])

        grasp.translate([
            grasp_x,
            grasp_y,
            grasp_z
        ])

        self.vis.add_geometry(
            grasp,
            reset_bounding_box=False
        )

        self.grasp_point = grasp
    
    def clear_target(self):
        
        if self.book is not None:
            self.vis.remove_geometry(
                self.book,
                reset_bounding_box=False
            )
            self.book = None

        if self.book_outline is not None:
            self.vis.remove_geometry(
                self.book_outline,
                reset_bounding_box=False
            )
            self.book_outline = None

        if self.book_center is not None:
            self.vis.remove_geometry(
                self.book_center,
                reset_bounding_box=False
            )
            self.book_center = None

        if self.grasp_point is not None:
            self.vis.remove_geometry(
                self.grasp_point,
                reset_bounding_box=False
            )
            self.grasp_point = None

        self.current_target = None

        self.insert_target = None
        self.pullout_target = None
        self.lift_target = None

        self.robot_timer = 0

        self.robot_state = STATE_IDLE

        self.target_angles = [HOME_THETA1, HOME_THETA2, HOME_THETA3]

    def update(self):

        # State Machine

        if self.current_target is not None:

            if self.robot_state == STATE_APPROACH:

                self.robot.gripper_open = True

                target = self.current_target

                approach_target = copy.deepcopy(self.current_target)

                approach_target.grasp_x_mm = target.approach_x_mm
                approach_target.grasp_y_mm = target.approach_y_mm
                approach_target.grasp_z_mm = target.approach_z_mm

                self.target_angles = list(
                    self.ik.solve(approach_target)
                )

            elif self.robot_state == STATE_INSERT:
                self.robot.gripper_open = True
                goal = self.current_target.grasp_z_mm
                current = self.insert_target.grasp_z_mm
                step = self.insert_speed

                if current < goal:
                    current = min(current + step, goal)

                else:
                    current = max(current - step, goal)

                self.insert_target.grasp_z_mm = current
                
                self.target_angles = list(
                    self.ik.solve(
                        self.insert_target
                    )   
                )

                if abs(current - goal) < 0.5:
                    self.robot_state = STATE_GRASP
                    self.state_timer = 0
                    self.book_state = BOOK_ATTACHED

            elif self.robot_state == STATE_PULL_OUT:

                self.robot.gripper_open = False

                goal = self.current_target.approach_z_mm
                current = self.pullout_target.grasp_z_mm
                step = self.pullout_speed

                if current < goal:
                    current = min(current + step, goal)
                else:
                    current = max(current - step, goal)

                self.pullout_target.grasp_z_mm = current

                self.target_angles = list(
                    self.ik.solve(
                        self.pullout_target
                    )
                )

                if abs(current - goal) < 0.5:
                    self.lift_target = copy.deepcopy(
                        self.pullout_target
                    )
                    self.robot_state = STATE_LIFT   

            elif self.robot_state == STATE_LIFT:

                self.robot.gripper_open = False

                goal = (self.current_target.grasp_y_mm + self.lift_height)
                current = self.lift_target.grasp_y_mm
                step = self.lift_speed

                if current < goal:
                    current = min(current + step, goal)
                else:
                    current = max(current - step, goal)

                self.lift_target.grasp_y_mm = current

                self.target_angles = list(
                    self.ik.solve(
                        self.lift_target
                    )
                )

                if abs(current - goal) < 0.5:
                    self.robot_state = STATE_HOME       

            elif self.robot_state == STATE_HOME:
                self.robot.gripper_open = True
                self.target_angles = [HOME_THETA1, HOME_THETA2, HOME_THETA3]

        # Smooth Joint Animation
        for i in range(3):

            error = (
                self.target_angles[i] - self.current_angles[i]
            )

            if abs(error) > 0.2:

                step = np.clip(
                    error,
                    -self.animation_speed,
                    self.animation_speed
                )

                self.current_angles[i] += step

        finished = True

        for i in range (3):
            if abs(
                self.target_angles[i] -
                self.current_angles[i]
            ) > 1:
                finished = False

        if finished:
            if self.robot_state == STATE_APPROACH:
                self.insert_target = copy.deepcopy(
                    self.current_target
                )
                # Start at the approach pose
                self.insert_target.grasp_x_mm = self.current_target.approach_x_mm
                self.insert_target.grasp_y_mm = self.current_target.approach_y_mm
                self.insert_target.grasp_z_mm = self.current_target.approach_z_mm
                
                self.robot_state = STATE_INSERT

            elif self.robot_state == STATE_GRASP:
                self.state_timer += 1
                if self.state_timer > 20:
                    self.pullout_target = copy.deepcopy(
                        self.current_target
                    )
                    self.robot_state = STATE_PULL_OUT
                    self.state_timer = 0

            elif self.robot_state == STATE_HOME:
                self.robot_state = STATE_IDLE
                self.current_target = None
                self.insert_target = None
                self.pullout_target = None
                self.lift_target = None
                self.state_timer = 0
                self.book_state = BOOK_RETRIEVED

        self.robot.set_angles(
            *self.current_angles
        )

        if self.current_target is not None:
            self.update_book(self.current_target)
            self.update_markers(self.current_target)

        self.robot.update_visualization()
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):

        while True:
            self.update()

    def destroy(self):
        
        self.vis.destroy_window()