import cv2
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import os
import pyrealsense2 as rs
import joblib
from coordinate_transform import pixel_to_camera
from grasp_target import create_grasp_target
from robot_world import RobotWorld

MODEL_PATH = "runs/obb/train3/weights/best.pt"
USE_REALSENSE = True
SOURCE = 2
START_CONF = 0.8
CSV_OUTPUT = "selected_books2.csv"
DISPLAY_SCALE = 0.75
INFO_PANEL_WIDTH = 420
CENTER_ZONE_RATIO = 0.15
MIN_VALID_DEPTH_MM = 370
MAX_VALID_DEPTH_MM = 405
CLASS_NAMES = {
    0: "book",
    1: "shelf"
}

conf_threshold = START_CONF
detections = []
selected_det = None

intrinsics = None
display_resize_scale = 1.0  

model = YOLO(MODEL_PATH)

width_model = joblib.load("width_model.pkl")
width_features = joblib.load("width_features.pkl")

weight_model = joblib.load("weight_model_D_huber.pkl")
weight_features = joblib.load("weight_features_D.pkl")

BOOK_SIZE_PRIORS = [
    ("US_Pocket", 175, 108),
    ("Mass_Market", 178, 110),
    ("B6", 176, 125),
    ("B6_Slim", 176, 110),
    ("US_Trade_Small", 203, 127),
    ("A5", 210, 148),
    ("Digest", 216, 140),
    ("Demy_UK", 216, 138),
    ("US_Standard", 229, 152),
    ("Royal_UK", 234, 156),
    ("B5", 250, 176),
    ("Executive", 267, 184),
    ("Large_Textbook", 280, 215),
    ("A4", 297, 210),
]

def infer_book_size_from_height(h_mm):
    best_family, best_h, best_w = min(
        BOOK_SIZE_PRIORS,
        key=lambda x: abs(h_mm - x[1])
    )
    return best_family, best_w, abs(h_mm - best_h)

def make_feature_row(det):
    height_mm = det["height_mm"]
    thickness_mm = det["thickness_mm"]
    depth_mm = det["depth_mm"]
    height_px = det["height_px"]
    thickness_px = det["thickness_px"]

    family, standard_width_prior, height_to_standard_error = infer_book_size_from_height(height_mm)

    row = {
        "height_px": height_px,
        "thickness_px": thickness_px,
        "depth_rs_mm": depth_mm,
        "height_mm_est": height_mm,
        "thickness_mm_est": thickness_mm,
        "aspect_ratio_px": height_px / thickness_px,
        "scale_h": height_mm / height_px,
        "thickness_ratio": thickness_mm / height_mm,
        "pixel_area_proxy": height_px * thickness_px,
        "standard_width_prior": standard_width_prior,
        "height_to_standard_error": height_to_standard_error,
        "prior_width_ratio": standard_width_prior / height_mm,
        "auto_paper_family": family,
        "height_x_thickness_est": height_mm * thickness_mm,
        "height_x_prior_width": height_mm * standard_width_prior,
        "thickness_x_prior_width": thickness_mm * standard_width_prior,
        "depth_x_height": depth_mm * height_mm,
        "thinness_score": height_mm / thickness_mm,
        "px_density_proxy": (height_px * thickness_px) / depth_mm,
        "thickness_to_height_est": thickness_mm / height_mm,
        "thickness_x_depth": thickness_mm * depth_mm,
        "height_x_depth": height_mm * depth_mm,
    }

    for fam, h_std, w_std in BOOK_SIZE_PRIORS:
        row[f"dist_h_{fam}"] = abs(height_mm - h_std)

    return row

def predict_width_weight(det):
    if det["mode_used"] != "DEPTH-AWARE":
        return None, None

    row = make_feature_row(det)

    X_width = pd.DataFrame([{col: row[col] for col in width_features}])
    width_pred = float(width_model.predict(X_width)[0])

    row["width_pred_mm"] = width_pred
    row["est_volume_proxy"] = row["height_mm_est"] * row["thickness_mm_est"] * width_pred

    X_weight = pd.DataFrame([{col: row[col] for col in weight_features}])
    weight_pred = float(weight_model.predict(X_weight)[0])

    return width_pred, weight_pred

def save_to_csv(det):
    if det["mode_used"] != "DEPTH-AWARE":
        print("⚠️ Not saved: invalid depth / off-axis / out of range")
        return

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "class": det["class_name"],
        "confidence": det["conf"],
        "thickness_px": det["thickness_px"],
        "height_px": det["height_px"],
        "angle_deg": det["angle"],
        "depth_mm": det["depth_mm"],
        "thickness_mm": det["thickness_mm"],
        "height_mm": det["height_mm"],
        "pred_width_mm": det["pred_width_mm"],
        "pred_weight_g": det["pred_weight_g"],
        "zone": det["zone"],         
        "mode": det["mode_used"]
    }
    df = pd.DataFrame([row])
    df.to_csv(CSV_OUTPUT, mode="a",
              header=not os.path.exists(CSV_OUTPUT),
              index=False)
    print("💾 Saved to CSV")

def check_center(cx, img_w):
    center_x = img_w / 2
    half_zone = (img_w * CENTER_ZONE_RATIO) / 2
    return abs(cx - center_x) <= half_zone

def point_in_obb(pt, obb_pts):
    return cv2.pointPolygonTest(obb_pts, pt, False) >= 0

def mouse_callback(event, x, y, flags, param):
    global selected_det
    if event == cv2.EVENT_LBUTTONDOWN:
        rx = int(x / DISPLAY_SCALE / display_resize_scale)   # 🔧 SCALE FIX
        ry = int(y / DISPLAY_SCALE / display_resize_scale)
        for det in detections:
            if point_in_obb((rx, ry), det["box_pts"]):
                selected_det = det
                if selected_det["grasp_target"]:
                    world.update_target(selected_det["grasp_target"])
                print(f"📌 Selected: {det['class_name']} | Conf={det['conf']:.3f}")
                break

cap = None
img = None
pipeline = None
align = None
depth_frame = None
IS_IMAGE = False

if USE_REALSENSE:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    print("✅ RealSense initialized")
else:
    # Webcam or Video File or Image
    if isinstance(SOURCE, str) and SOURCE.lower().endswith((".jpg", ".png", ".jpeg")):
        IS_IMAGE = True
        img = cv2.imread(SOURCE)
        if img is None:
            raise RuntimeError(f"❌ Cannot load image: {SOURCE}")
        print(f"🖼 Loaded image: {SOURCE}")
    else:
        # Webcam
        cap = cv2.VideoCapture(SOURCE)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open video source: {SOURCE}")
        print(f"📹 Webcam opened: {SOURCE}")

cv2.namedWindow("OBB Demo", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("OBB Demo", mouse_callback)

def get_median_depth_mm(depth_frame, cx, cy, patch=7):
    if depth_frame is None:
        return None

    half = patch // 2
    depths = []

    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            x = int(cx + dx)
            y = int(cy + dy)
            w = depth_frame.get_width()
            h = depth_frame.get_height()
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            try:
                d = depth_frame.get_distance(x, y)
                if d > 0:
                    depths.append(d * 1000)  # m → mm
            except:
                pass

    if len(depths) == 0:
        return None

    return float(np.median(depths))

world = RobotWorld()
world.create_world()

# === MAIN LOOP ===
while True:

    # ---------- FRAME ----------
    if USE_REALSENSE:
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            continue
        measure_canvas = np.asanyarray(color_frame.get_data())
    else:
        if IS_IMAGE:
            measure_canvas = img.copy()
        else:
            ret, measure_canvas = cap.read()
            if not ret:
                break

    detections.clear()

    # ---------- YOLO (MEASUREMENT IMAGE) ----------
    results = model(measure_canvas, conf=conf_threshold, verbose=False)[0]

    if results.obb:
        for box, conf, cls in zip(results.obb.xywhr,
                                  results.obb.conf,
                                  results.obb.cls):

            x, y, bt, bh, a = box.tolist()
            angle_deg = math.degrees(a)

            px_t = int(min(bt, bh))
            px_h = int(max(bt, bh))

            rect = ((x, y), (bt, bh), angle_deg)
            box_pts = cv2.boxPoints(rect).astype(int)

            # ---------- CENTER CHECK ----------
            is_center = check_center(x, measure_canvas.shape[1])
            zone = "CENTER" if is_center else "OFF-AXIS"

            # ---------- DEPTH ----------
            depth_mm = get_median_depth_mm(depth_frame, x, y) if USE_REALSENSE else None
            depth_out_of_range = (
                depth_mm is not None and
                (depth_mm < MIN_VALID_DEPTH_MM or depth_mm > MAX_VALID_DEPTH_MM)
            )

            DEFAULT_FX = 600
            DEFAULT_FY = 600

            depth_ok = (
                depth_mm is not None and
                not depth_out_of_range and
                is_center and
                intrinsics
            )

            if depth_ok:
                thickness_mm = px_t * depth_mm / intrinsics.fx
                height_mm = px_h * depth_mm / intrinsics.fy
                mode_used = "DEPTH-AWARE"
                camera_coord = pixel_to_camera(
                    u=x,
                    v=y,
                    depth_mm=depth_mm,
                    intrinsics=intrinsics
                )
            else:
                thickness_mm = None
                height_mm = None
                mode_used = "INVALID DEPTH"
                camera_coord = None
            
            temp_det = {
                "thickness_px": px_t,
                "height_px": px_h,
                "angle": angle_deg,
                "conf": float(conf),
                "center_u": float(x),
                "center_v": float(y),
                "class_name": CLASS_NAMES.get(int(cls), str(cls)),
                "box_pts": box_pts,
                "depth_mm": depth_mm,
                "depth_out_of_range": depth_out_of_range,
                "thickness_mm": thickness_mm,
                "height_mm": height_mm,
                "zone": zone,
                "mode_used": mode_used
            }

            pred_width, pred_weight = predict_width_weight(temp_det)
            temp_det["pred_width_mm"] = pred_width
            temp_det["pred_weight_g"] = pred_weight
            temp_det["camera_coord"] = camera_coord
            temp_det["grasp_target"] = create_grasp_target(temp_det)

            detections.append(temp_det)

    # ---------- DRAW ----------
    display_canvas = measure_canvas.copy()

    for det in detections:
    # dim lines before selection
        color = (0, 180, 0) if det["zone"] == "CENTER" else (0, 180, 180)
        cv2.drawContours(display_canvas, [det["box_pts"]], 0, color, 2)

    # draw selected on top
    if selected_det:
        sel_color = (0, 255, 0) if selected_det["zone"] == "CENTER" else (0, 255, 255)
        cv2.drawContours(display_canvas, [selected_det["box_pts"]], 0, sel_color, 3)

    # ---------- DISPLAY SCALE (UI ONLY) ----------
    h, w = display_canvas.shape[:2]
    display_resize_scale = min(1280 / w, 720 / h)

    display_canvas = cv2.resize(
        display_canvas,
        None,
        fx=display_resize_scale,
        fy=display_resize_scale,
        interpolation=cv2.INTER_AREA
    )

    display_img = cv2.resize(
        display_canvas,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    # ---------- UI PANEL (UNCHANGED) ----------
    dh, dw = display_img.shape[:2]
    panel = np.zeros((dh, INFO_PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (35, 35, 35)

    cv2.putText(panel, "Object Information", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    y = 60
    step = 26

    if selected_det:
        lines = [
            f"Class        : {selected_det['class_name']}",
            f"Confidence   : {selected_det['conf']:.3f}",
            f"Thickness    : {selected_det['thickness_px']} px",
            f"Height       : {selected_det['height_px']} px",
            f"Angle        : {selected_det['angle']:.2f} deg",
            (f"Depth (Z)   : {selected_det['depth_mm']:.1f} mm (OUT OF RANGE)"
             if selected_det.get("depth_out_of_range")
             else f"Depth (Z)    : {selected_det['depth_mm']:.1f} mm") if selected_det['depth_mm'] else "Depth        : N/A",
            f"Camera X : {selected_det['camera_coord'].x_mm:.1f} mm" if selected_det["camera_coord"] else "Camera X : N/A",
            f"Camera Y : {selected_det['camera_coord'].y_mm:.1f} mm" if selected_det["camera_coord"] else "Camera Y : N/A",
            f"Camera Z : {selected_det['camera_coord'].z_mm:.1f} mm" if selected_det["camera_coord"] else "Camera Z : N/A",
            f"Zone         : {selected_det['zone']}",
            f"Thickness    : {selected_det['thickness_mm']:.1f} mm" if selected_det['thickness_mm'] is not None else "Thickness    : N/A",
            f"Height       : {selected_det['height_mm']:.1f} mm" if selected_det['height_mm'] is not None else "Height       : N/A",
            f"Pred Width   : {selected_det['pred_width_mm']:.1f} mm" if selected_det['pred_width_mm'] is not None else "Pred Width   : N/A",
            f"Pred Weight  : {selected_det['pred_weight_g']:.0f} g" if selected_det['pred_weight_g'] is not None else "Pred Weight  : N/A",
            f"Mode         : {selected_det['mode_used']}"
        ]

        if selected_det["grasp_target"]:
            gt = selected_det["grasp_target"]
            lines.extend([
                "",
                "----- Grasp Target -----",
                f"Target X    : {gt.x_mm:.1f} mm",
                f"Target Y    : {gt.y_mm:.1f} mm",
                f"Target Z    : {gt.z_mm:.1f} mm",
                f"Grip Angle  : {gt.angle_deg:.1f} deg"
            ])

    else:
        lines = [
            "No object selected",
            "",
            "-> Click inside a book",
            "-> Press C to cancel",
            "Mode: DEPTH-AWARE only"
        ]

    for line in lines:
        cv2.putText(panel, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        y += step

    cv2.putText(panel, "W/X : Conf +/-  ",
                (20, dh - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(panel, "C : Cancel   S : Save   Q : Quit",
                (20, dh - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(panel, f"Current Conf: {conf_threshold:.2f}",
                (20, dh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("OBB Demo", np.hstack((display_img, panel)))

    # ---------- KEYS ----------
    world.update()
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord("w"):
        conf_threshold = min(conf_threshold + 0.05, 0.95)
    elif k == ord("x"):
        conf_threshold = max(conf_threshold - 0.05, 0.05)
    elif k == ord("c"):
        selected_det = None
    elif k == ord("s") and selected_det:
        save_to_csv(selected_det)

if USE_REALSENSE:
    pipeline.stop()
if cap:
    cap.release()
cv2.destroyAllWindows()