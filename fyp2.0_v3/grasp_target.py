"""
This module converts perception output into a unified robotic grasp target.

Nothing downstream should read the raw YOLO detection dictionary anymore.

Everything should use GraspTarget.
"""

from dataclasses import dataclass
from world_transform import camera_to_world

@dataclass
class GraspTarget:
    """
    Unified grasp target used by the
    simulator and future robot arm.
    """
    
    # Physical Book Dimensions
    book_x_mm: float
    book_y_mm: float
    book_z_mm: float

    width_mm: float
    height_mm: float
    thickness_mm: float
    
    angle_deg: float # Gripper Rotation  

    # Grasp
    grasp_x_mm: float
    grasp_y_mm: float
    grasp_z_mm: float

    # Approach
    approach_x_mm: float
    approach_y_mm: float
    approach_z_mm: float

    # Original Camera Coordinate (Useful for debugging)
    camera_x_mm: float
    camera_y_mm: float
    camera_z_mm: float

    @property
    def book_position(self):
        return (
            self.book_x_mm,
            self.book_y_mm,
            self.book_z_mm
        )

    @property
    def grasp_position(self):
        return (
            self.grasp_x_mm,
            self.grasp_y_mm,
            self.grasp_z_mm
        )

    @property
    def approach_position(self):
        return (
            self.approach_x_mm,
            self.approach_y_mm,
            self.approach_z_mm
        )

    @property
    def camera_position(self):
        """ Camera XYZ """
        return (
            self.camera_x_mm,
            self.camera_y_mm,
            self.camera_z_mm
        )

def create_grasp_target(det):
    # Convert one perception detection into one grasp target.
    camera_coord = det.get("camera_coord")

    if camera_coord is None:
        return None

    world = camera_to_world(camera_coord)

    book_x = float(world.x_mm)
    book_y = float(world.y_mm)

    width = det.get("pred_width_mm")
    height = det.get("height_mm")
    thickness = det.get("thickness_mm")

    grasp_x = book_x
    grasp_y = book_y

    approach_x = grasp_x
    approach_y = grasp_y

    APPROACH_OFFSET = 80.0

    # Just for z
    grasp_z = float(world.z_mm)
    book_z = grasp_z + width / 2
    approach_z = grasp_z - APPROACH_OFFSET

    if world is None:
        return None

    # Fallback values
    if width is None:
        width = 150.0

    if height is None:
        height = 220.0

    if thickness is None:
        thickness = 30.0

    return GraspTarget(

        # Book
        book_x_mm=book_x,
        book_y_mm=book_y,
        book_z_mm=book_z,
        width_mm=float(width),
        height_mm=float(height),
        thickness_mm=float(thickness),
        angle_deg=float(det["angle"]),

        # Grasp
        grasp_x_mm=grasp_x,
        grasp_y_mm=grasp_y,
        grasp_z_mm=grasp_z,

        # Approach
        approach_x_mm=approach_x,
        approach_y_mm=approach_y,
        approach_z_mm=approach_z,

        # Camera Coordinate
        camera_x_mm=float(camera_coord.x_mm),
        camera_y_mm=float(camera_coord.y_mm),
        camera_z_mm=float(camera_coord.z_mm)
    )