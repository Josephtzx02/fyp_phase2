"""
world_transform.py

Stage 4A

Convert Camera Coordinates
into Robot World Coordinates.

Camera Coordinate System
------------------------

        +Z (forward)
         ^
         |
         |
 +Y      |
 /        |
O-------> +X

Robot World Coordinate System

          +Y (Height)
             ^
             |
             |
O---------> +X (Left / Right)
 \
  \
   +Z (Forward to bookshelf)

Robot Base = (0,0,0)

All distances are millimetres.
"""

from dataclasses import dataclass

@dataclass
class WorldCoordinate:
    x_mm: float
    y_mm: float
    z_mm: float


def camera_to_world(camera_coord):
    """
    Convert RealSense camera coordinates
    into robot world coordinates.

    Camera:
        X = left/right
        Y = up/down
        Z = distance from camera

    Robot World:
        X = left/right; AGV
        Y = Height of book centre; Shoulder/Elbow lifting
        Z = Forward reach (camera → book distance); Arm extension
    """

    if camera_coord is None:
        return None

    world_x = -camera_coord.x_mm      # AGV Left/Right
    world_y = -camera_coord.y_mm      # Height
    world_z = camera_coord.z_mm       # Forward reach

    return WorldCoordinate(
        x_mm=world_x,
        y_mm=world_y,
        z_mm=world_z
    )