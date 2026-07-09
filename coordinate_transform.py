"""
Stage 1

Convert image pixel coordinates into
3D camera coordinates using the
RealSense intrinsic parameters.

Camera Coordinate System (RealSense)

            +Y (Down)
             |
             |
             |
+X ----------O----------> Right
            /
           /
         +Z (Forward)

where

X = Left / Right from camera centre
Y = Up / Down from camera centre
Z = Distance from camera

This module performs ONLY the conversion:

    Pixel (u,v) + Depth
                ↓
        CameraCoordinate (X,Y,Z)

Robot coordinate conversion is handled separately
inside world_transform.py.

"""

from dataclasses import dataclass

@dataclass
class CameraCoordinate:

    u: float
    v: float

    x_mm: float
    y_mm: float
    z_mm: float

def pixel_to_camera(u,
                    v,
                    depth_mm,
                    intrinsics):

    if depth_mm is None:
        return None

    fx = intrinsics.fx
    fy = intrinsics.fy

    cx = intrinsics.ppx
    cy = intrinsics.ppy

    x_mm = (u - cx) * depth_mm / fx
    y_mm = (v - cy) * depth_mm / fy
    z_mm = depth_mm

    return CameraCoordinate(
        u=u,
        v=v,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm
    )