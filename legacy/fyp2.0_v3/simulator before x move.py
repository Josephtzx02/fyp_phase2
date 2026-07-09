"""
Robot Arm Simulator
- Receives GraspTarget objects.
- Computes IK.
- Updates RobotArm.
- Displays everything in Open3D.
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

BOOK_ON_SHELF = "ON_SHELF"
BOOK_ATTACHED = "ATTACHED"
BOOK_RETRIEVED = "RETRIEVED"

CAMERA_HEIGHT = 320.0

PARK_POSE = (80.0, -140.0, 30.0) #(-35.0, 70.0, -35.0) Elbow-Up

# Shelf Parameters
SHELF_FRONT_Z = 380.0
SHELF_WIDTH = 900.0
SHELF_DEPTH = 250.0

BOARD_THICKNESS = 25.0
NUM_SHELVES = 3
SHELF_CLEARANCE = 300.0
BOTTOM_CLEARANCE = 50.0

LEFT_X = -350.0
BOTTOM_Y = 0.0

BOOK_CLEARANCE = 5.0

SHELF_HEIGHT = (
    BOTTOM_CLEARANCE
    +
    NUM_SHELVES * SHELF_CLEARANCE
    +
    (NUM_SHELVES + 1) * BOARD_THICKNESS
)

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
        self.insert_speed = 4.0 # mm/frame
        self.pullout_target = None
        self.pullout_speed = self.insert_speed
        self.lift_target = None
        self.lift_speed = 4.0      # mm/frame
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

        self.shelf_levels = []
        current = BOTTOM_CLEARANCE
        for _ in range(NUM_SHELVES):
            self.shelf_levels.append(current)
            current += SHELF_CLEARANCE + BOARD_THICKNESS

        self.shelf_boards = []
        self.shelf_outlines = []

        self.create_bookshelf()
        self.create_robot_base()
        self.robot.create_meshes()

        self.vis.poll_events()
        self.vis.update_renderer()
        self.setup_camera()

        self.state_name = {
            STATE_IDLE: "Idle",
            STATE_APPROACH: "Approach",
            STATE_INSERT: "Insert",
            STATE_GRASP: "Grasp",
            STATE_PULL_OUT: "Pull Out",
            STATE_LIFT: "Lift"
        }

        self.book_state = BOOK_ON_SHELF

        self.current_angles = list(PARK_POSE)

        self.target_angles = self.current_angles.copy()

        self.animation_speed = 2.0

    def create_bookshelf(self):

        wood = [0.65, 0.45, 0.20]

        def add_outline(mesh):

            bbox = mesh.get_axis_aligned_bounding_box()

            outline = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
                bbox
            )

            outline.paint_uniform_color([
                0.25,
                0.12,
                0.02
            ])

            self.vis.add_geometry(
                outline,
                reset_bounding_box=False
            )

            self.shelf_outlines.append(
                outline
            )

        left = o3d.geometry.TriangleMesh.create_box(
            width=BOARD_THICKNESS,
            height=SHELF_HEIGHT,   
            depth=SHELF_DEPTH
        )
        left.paint_uniform_color(wood)

        left.translate([
            LEFT_X,
            BOTTOM_Y,
            SHELF_FRONT_Z
        ])
        self.vis.add_geometry(left, reset_bounding_box=True)
        self.shelf_boards.append(left)
        add_outline(left)

        right = o3d.geometry.TriangleMesh.create_box(
            width=BOARD_THICKNESS,
            height=SHELF_HEIGHT,
            depth=SHELF_DEPTH
        )
        right.paint_uniform_color(wood)

        right.translate([
            LEFT_X + SHELF_WIDTH - BOARD_THICKNESS,
            BOTTOM_Y,
            SHELF_FRONT_Z
        ])
        self.vis.add_geometry(right, reset_bounding_box=False)
        self.shelf_boards.append(right)
        add_outline(right)

        bottom = o3d.geometry.TriangleMesh.create_box(
            width=SHELF_WIDTH,
            height=BOARD_THICKNESS,
            depth=SHELF_DEPTH
        )
        bottom.paint_uniform_color(wood)

        bottom.translate([
            LEFT_X,
            BOTTOM_Y,
            SHELF_FRONT_Z
        ])
        self.vis.add_geometry(bottom, reset_bounding_box=False)
        self.shelf_boards.append(bottom)
        add_outline(bottom)

        for level in self.shelf_levels:
            board = o3d.geometry.TriangleMesh.create_box(
                width=SHELF_WIDTH,
                height=BOARD_THICKNESS,
                depth=SHELF_DEPTH
            )
            board.paint_uniform_color(wood)
            board.translate([
                LEFT_X,
                level,
                SHELF_FRONT_Z
            ])
            self.vis.add_geometry(board, reset_bounding_box=False)
            self.shelf_boards.append(board)
            add_outline(board)

        top = o3d.geometry.TriangleMesh.create_box(
            width=SHELF_WIDTH,
            height=BOARD_THICKNESS,
            depth=SHELF_DEPTH
        )
        top.paint_uniform_color(wood)

        top.translate([
            LEFT_X,
            SHELF_HEIGHT - BOARD_THICKNESS,
            SHELF_FRONT_Z
        ])
        self.vis.add_geometry(top, reset_bounding_box=False)
        self.shelf_boards.append(top)
        add_outline(top)

    def create_robot_base(self):

        base = o3d.geometry.TriangleMesh.create_box(
            width=400,
            height=30,
            depth=400
        )

        base.paint_uniform_color([
            0.28,
            0.28,
            0.28
        ])

        # Centre at robot origin
        base.translate([
            -225,
            -30,
            -120
        ])

        self.vis.add_geometry(
            base,
            reset_bounding_box=False
        )

        pedestal = o3d.geometry.TriangleMesh.create_box(
            width=120,
            height=240,
            depth=120
        )

        pedestal.paint_uniform_color([
            0.52,
            0.52,
            0.52
        ])

        pedestal.translate([
            -60,
            0,
            -60
        ])

        self.vis.add_geometry(
            pedestal,
            reset_bounding_box=False
        )

        turntable = o3d.geometry.TriangleMesh.create_cylinder(
            radius=60,
            height=20
        )

        turntable.paint_uniform_color([
            0.18,
            0.18,
            0.18
        ])

        turntable.translate([
            0,
            290,
            0
        ])

        self.vis.add_geometry(
            turntable,
            reset_bounding_box=False
        )

    def get_current_shelf_y(self, target):
        # Middle shelf for now
        shelf_board = self.shelf_levels[1]
        return shelf_board + BOARD_THICKNESS + BOOK_CLEARANCE + target.height_mm / 2

    def setup_camera(self):
        
        ctr = self.vis.get_view_control()
        ctr.set_front([-0.5, 0.1, -0.6])
        ctr.set_up([0, 1, 0])
        ctr.set_lookat([0, 450, 700])
        ctr.set_zoom(0.65)

    def update_target(self, grasp_target):

        if grasp_target is None:
            return

        self.insert_target = None
        self.pullout_target = None
        self.lift_target = None

        self.state_timer = 0

        calibrated = copy.deepcopy(grasp_target)

        book_center_y = self.get_current_shelf_y(calibrated)

        offset = book_center_y - calibrated.book_y_mm

        calibrated.book_y_mm += offset
        calibrated.grasp_y_mm += offset
        calibrated.approach_y_mm += offset

        # Simulator always works on a calibrated copy.
        # Original perception data remains unchanged.
        self.current_target = calibrated

        self.book_state = BOOK_ON_SHELF
        self.robot_state = STATE_APPROACH

        self.robot.book_thickness = calibrated.thickness_mm

        self.robot.set_wrist_rotation(calibrated.angle_deg)

        self.update_book(calibrated)
        self.update_markers(calibrated)

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
            draw_y = target.book_y_mm
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
            book_y = target.book_y_mm
            book_z = target.book_z_mm

            grasp_x = target.grasp_x_mm
            grasp_y = target.grasp_y_mm
            grasp_z = target.grasp_z_mm

            approach_x = target.approach_x_mm
            approach_y = target.approach_y_mm
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

        self.state_timer = 0
        self.robot_state = STATE_IDLE
        self.book_state = BOOK_ON_SHELF
        self.robot.gripper_open = True
        self.target_angles = list(PARK_POSE)

    def solve_target(self, target):
        # True  -> valid solution found
        # False -> target unreachable

        success, angles = self.ik.solve(target)

        if not success:
            print("[IK] Target unreachable.")
            return False

        self.target_angles = list(angles)

        return True

    def update(self):
        if self.current_target is not None:

            if self.robot_state == STATE_APPROACH:

                self.robot.gripper_open = True

                target = self.current_target

                approach_target = copy.deepcopy(self.current_target)

                approach_target.grasp_x_mm = target.approach_x_mm
                approach_target.grasp_y_mm = target.approach_y_mm
                approach_target.grasp_z_mm = target.approach_z_mm

                if not self.solve_target(approach_target):
                    return

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
                
                if not self.solve_target(self.insert_target):
                    return
                
                if abs(current - goal) < 0.5:
                    self.robot_state = STATE_GRASP
                    self.state_timer = 0
                    self.book_state = BOOK_ATTACHED

            elif self.robot_state == STATE_PULL_OUT:

                self.robot.gripper_open = False

                goal = self.current_target.grasp_z_mm - self.current_target.width_mm - BOARD_THICKNESS 
                current = self.pullout_target.grasp_z_mm
                step = self.pullout_speed

                if current < goal:
                    current = min(current + step, goal)
                else:
                    current = max(current - step, goal)

                self.pullout_target.grasp_z_mm = current

                if not self.solve_target(self.pullout_target):
                    return

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

                if not self.solve_target(self.lift_target):
                    return

                if abs(current - goal) < 0.5:
                    self.state_timer = 1

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

            elif self.robot_state == STATE_LIFT:
                if self.state_timer == 1:
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