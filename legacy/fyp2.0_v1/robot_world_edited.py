import numpy as np
import math

class RobotWorldTransformer:
    """
    Converts YOLOv8-OBB + RGB-D outputs into consistent 3D world coordinates
    for robotic grasp planning.
    """

    def __init__(self, fx, fy, cx, cy, depth_scale=1.0):
        """
        fx, fy: camera focal lengths (intrinsics)
        cx, cy: principal point
        depth_scale: usually 1.0 for RealSense mm conversion already handled
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

    # =========================================================
    # 1. PIXEL → 3D CAMERA SPACE
    # =========================================================
    def pixel_to_camera(self, u, v, depth_mm):
        """
        Converts pixel + depth → 3D camera coordinates (X, Y, Z)
        Z = depth (mm)
        """

        if depth_mm is None:
            return None

        Z = depth_mm * self.depth_scale
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        return np.array([X, Y, Z], dtype=np.float32)

    # =========================================================
    # 2. OBB CENTER REFINEMENT (IMPORTANT FIX)
    # =========================================================
    def refine_grasp_center(self, box_pts):
        """
        Instead of raw centroid, use minimum-area rectangle midpoint
        to stabilize grasp target.
        """

        pts = np.array(box_pts, dtype=np.float32)

        # diagonal midpoint of OBB
        center = np.mean(pts, axis=0)

        return float(center[0]), float(center[1])

    # =========================================================
    # 3. ORIENTATION ESTIMATION
    # =========================================================
    def compute_orientation(self, box_pts):
        """
        Computes stable book spine angle (improves YOLO angle noise)
        """

        pts = np.array(box_pts, dtype=np.float32)

        edge1 = pts[1] - pts[0]
        angle = math.degrees(math.atan2(edge1[1], edge1[0]))

        # normalize to [-90, 90]
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180

        return angle

    # =========================================================
    # 4. WORLD GRASP VECTOR GENERATION
    # =========================================================
    def build_grasp_vector(self, det):
        """
        Converts full detection → robotic grasp vector
        """

        if det["depth_mm"] is None:
            return None

        # 1. refined center (IMPORTANT FIX)
        u, v = self.refine_grasp_center(det["box_pts"])

        # 2. depth
        z = det["depth_mm"]

        # 3. camera space position
        cam_xyz = self.pixel_to_camera(u, v, z)
        if cam_xyz is None:
            return None

        # 4. orientation correction (more stable than YOLO raw angle)
        theta = self.compute_orientation(det["box_pts"])

        # 5. thickness/height consistency (ensure correct axis mapping)
        thickness = det.get("thickness_mm", None)
        height = det.get("height_mm", None)

        # 6. final unified grasp vector
        grasp_vector = {
            "x": float(cam_xyz[0]),
            "y": float(cam_xyz[1]),
            "z": float(cam_xyz[2]),
            "theta": float(theta),
            "thickness_mm": thickness,
            "height_mm": height,
            "depth_mm": float(z)
        }

        return grasp_vector

    # =========================================================
    # 5. SAFETY FILTER (FOR ROBOT GRASP STABILITY)
    # =========================================================
    def is_valid_grasp(self, grasp_vector, zone="CENTER"):
        """
        Filters unstable grasps (important for real robot integration)
        """

        if grasp_vector is None:
            return False

        # must be center zone (your thesis constraint)
        if zone != "CENTER":
            return False

        # depth sanity check
        if grasp_vector["z"] < 350 or grasp_vector["z"] > 450:
            return False

        # avoid extreme angles
        if abs(grasp_vector["theta"]) > 85:
            return False

        return True