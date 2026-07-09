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
    
    # Robot World Coordinate
    x_mm: float
    y_mm: float
    z_mm: float
    
    # Physical Book Dimensions
    width_mm: float
    height_mm: float
    thickness_mm: float
    
    # Gripper Rotation    
    angle_deg: float

    # Original Camera Coordinate (Useful for debugging)

    camera_x_mm: float
    camera_y_mm: float
    camera_z_mm: float

    @property
    def world_position(self):
        """ Robot World XYZ"""
        return (
            self.x_mm,
            self.y_mm,
            self.z_mm
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

    if world is None:
        return None

    width = det.get("pred_width_mm")
    height = det.get("height_mm")
    thickness = det.get("thickness_mm")

    # Fallback values
    if width is None:
        width = 150.0

    if height is None:
        height = 220.0

    if thickness is None:
        thickness = 30.0

    return GraspTarget(

        # Robot World
        x_mm=float(world.x_mm),
        y_mm=float(world.y_mm),
        z_mm=float(world.z_mm),

        # Book
        width_mm=float(width),
        height_mm=float(height),
        thickness_mm=float(thickness),

        # Rotation
        angle_deg=float(det["angle"]),

        # Camera Coordinate
        camera_x_mm=float(camera_coord.x_mm),
        camera_y_mm=float(camera_coord.y_mm),
        camera_z_mm=float(camera_coord.z_mm)
    )