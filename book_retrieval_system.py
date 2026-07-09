import cv2
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import os
import pyrealsense2 as rs
import joblib
from simulator import Simulator
from coordinate_transform import pixel_to_camera
from grasp_target import create_grasp_target

MODEL_PATH = "models/best.pt"
USE_REALSENSE = True
SOURCE = 2
START_CONF = 0.8
CSV_OUTPUT = "selected_books.csv"
DISPLAY_SCALE = 0.75
INFO_PANEL_WIDTH = 430
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
SHOW_FPS = False

intrinsics = None
display_resize_scale = 1.0  

model = YOLO(MODEL_PATH)

width_model = joblib.load("models/width_model.pkl")
width_features = joblib.load("models/width_features.pkl")

weight_model = joblib.load("models/weight_model_D_huber.pkl")
weight_features = joblib.load("models/weight_features_D.pkl")

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
        "grasp_x": det["grasp_target"]["x"] if det.get("grasp_target") else None,
        "grasp_y": det["grasp_target"]["y"] if det.get("grasp_target") else None,
        "grasp_z": det["grasp_target"]["z"] if det.get("grasp_target") else None,
        "grasp_theta": det["grasp_target"]["theta"] if det.get("grasp_target") else None,
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

def emit_grasp_command(det):
    cmd = create_grasp_target(det)

    # Convert to robot-friendly format (4DOF-style)
    robot_cmd = {
        "cmd": "GRASP",
        "x": cmd["x"],
        "y": cmd["y"],
        "z": cmd["z"],
        "theta": cmd["theta"],
        "grip_width": cmd["width_mm"],
        "weight": cmd["weight_g"]
    }
    print("\n🤖 GRASP COMMAND GENERATED:")
    print(robot_cmd)

    return robot_cmd

def mouse_callback(event, x, y, flags, param):
    global selected_det
    if event == cv2.EVENT_LBUTTONDOWN:
        rx = int(x / DISPLAY_SCALE / display_resize_scale)   # 🔧 SCALE FIX
        ry = int(y / DISPLAY_SCALE / display_resize_scale)
        for det in detections:
            if point_in_obb((rx, ry), det["box_pts"]):
                selected_det = det

                if selected_det["grasp_target"]:
                    simulator.update_target(
                        selected_det["grasp_target"]
                    )
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

cv2.namedWindow("Book Perception", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Robot Controller", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Book Perception", 0, 0)
cv2.moveWindow("Robot Controller", 1390, 0)
cv2.setMouseCallback("Book Perception", mouse_callback)\

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

import time
prev_time = time.time()
fps = 0.0
simulator = Simulator()

# === MAIN LOOP ===
while True:
    current_time = time.time()
    instant_fps = 1.0 / max(current_time - prev_time, 1e-6)
    fps = 0.9 * fps + 0.1 * instant_fps
    prev_time = current_time

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

            if bt > bh:
                bt, bh = bh, bt
                angle_deg += 90

            # Normalize angle to [-90, 90)
            while angle_deg >= 90:
                angle_deg -= 180

            while angle_deg < -90:
                angle_deg += 180

            px_t = int(bt)
            px_h = int(bh)

            #px_t = int(min(bt, bh))
            #px_h = int(max(bt, bh))

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

    cv2.putText(panel, "Book Information", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    y = 60
    step = 26

    if selected_det:
        gt = selected_det.get("grasp_target")
        lines = [
            f"Class        : {selected_det['class_name']}",
            f"Confidence   : {selected_det['conf']:.3f}",
            f"Thickness    : {selected_det['thickness_px']} px",
            f"Height       : {selected_det['height_px']} px",
            f"Angle        : {selected_det['angle']:.2f} deg",
            (f"Depth (Z)   : {selected_det['depth_mm']:.1f} mm (OUT OF RANGE)"
             if selected_det.get("depth_out_of_range")
             else f"Depth (Z)   : {selected_det['depth_mm']:.1f} mm") if selected_det['depth_mm'] else "Depth        : N/A",
            f"Zone         : {selected_det['zone']}",
            f"Thickness    : {selected_det['thickness_mm']:.1f} mm" if selected_det['thickness_mm'] is not None else "Thickness    : N/A",
            f"Height       : {selected_det['height_mm']:.1f} mm" if selected_det['height_mm'] is not None else "Height       : N/A",
            f"Pred Width   : {selected_det['pred_width_mm']:.1f} mm" if selected_det['pred_width_mm'] is not None else "Pred Width   : N/A",
            f"Pred Weight  : {selected_det['pred_weight_g']:.0f} g" if selected_det['pred_weight_g'] is not None else "Pred Weight  : N/A"
        ]

    else:
        lines = [
            "No book selected",
            "",
            "-> Click on a book",
            "-> Press C to cancel",
            "Mode: DEPTH-AWARE only"
        ]

    for line in lines:
        cv2.putText(panel, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        y += step

    cv2.putText(panel, "W/X : Conf +/-   1: F  2: T  3: L  4: R  5: ISO",
                (20, dh - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(panel, "C : Cancel   S : Save   F : FPS   Q : Quit",
                (20, dh - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(panel, f"Current Conf: {conf_threshold:.2f}",
                (20, dh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if SHOW_FPS:
        cv2.putText(display_img, f"FPS : {fps:.1f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Book Perception", np.hstack((display_img, panel)))

    #Robot Controller Window
    robot_panel = np.zeros(
        (1000, 460, 3),
        dtype=np.uint8
    )

    robot_panel[:] = (30,30,30)

    cv2.putText(robot_panel, "Robot Controller",
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        
    y = 75
    step = 28

    if simulator.current_base_x > 1:
        agv_arrow = "[<-] "

    elif simulator.current_base_x < -1:
        agv_arrow = "[->] "

    else:
        agv_arrow = ""

    robot_lines = [
        "========== ROBOT ==========",
        f"Gripper     : {'OPEN' if simulator.robot.gripper_open else 'CLOSED'}",
        f"Book State : {simulator.book_state}",
        f"AGV X      : {agv_arrow}{simulator.current_base_x:.1f} mm",
        "",
        "======== JOINT ANGLES ========",
        f"Shoulder : {simulator.current_angles[0]:.1f} deg",
        f"Elbow    : {simulator.current_angles[1]:.1f} deg",
        f"Wrist     : {simulator.current_angles[2]:.1f} deg"
    ]

    if simulator.robot_state == "IDLE":
        state_text = "IDLE"
        state_colour = (0,255,0)
    elif simulator.robot_state == "APPROACH":
        state_text = "APPROACH"
        state_colour = (0,255,255)
    elif simulator.robot_state == "INSERT":
        state_text = "INSERT"
        state_colour = (0,170,255)
    elif simulator.robot_state == "GRASP":
        state_text = "GRASP"
        state_colour = (255,150,0)
    elif simulator.robot_state == "PULL_OUT":
        state_text = "PULL OUT"
        state_colour = (255,0,255)
    elif simulator.robot_state == "LIFT":
        state_text = "LIFT"
        state_colour = (255,220,50)
    else:
        state_text = simulator.robot_state
        state_colour = (230,230,230)

    if simulator.book_state == "ON_SHELF":
        book_colour = (230,230,230)
    elif simulator.book_state == "ATTACHED":
        book_colour = (255,220,50)
    elif simulator.book_state == "RETRIEVED":
        book_colour = (0,255,0)
    else:
        book_colour = (230,230,230)

    #Nothing Selected
    if selected_det is None:
        robot_lines.extend([
            "",
            "========== TARGET ==========",
            "No target selected."
        ])

    #Something Selected
    else: 
        gt = selected_det.get("grasp_target")
        # No valid grasp target
        if gt is None:
            robot_lines.extend([
                "",
                "========== TARGET ==========",
                "No grasp target."
            ])

        else: 
            robot_lines.extend([
                "",
                "========== BOOK ==========",
                f"Book X : {gt.book_x_mm:.1f} mm",
                f"Book Y : {gt.book_y_mm:.1f} mm",
                f"Book Z : {gt.book_z_mm:.1f} mm",
                "",
                "========== GRASP ==========",
                f"Grasp X : {gt.grasp_x_mm:.1f} mm",
                f"Grasp Y : {gt.grasp_y_mm:.1f} mm",
                f"Grasp Z : {gt.grasp_z_mm:.1f} mm",
                "",
                f"Grip Angle : {gt.angle_deg:.1f} deg",
                "",
                "========== CAMERA ==========",
                f"X : {gt.camera_x_mm:.1f} mm",
                f"Y : {gt.camera_y_mm:.1f} mm",
                f"Z : {gt.camera_z_mm:.1f} mm",
                "",
                "========== APPROACH ==========",
                f"Approach X : {gt.approach_x_mm:.1f} mm",
                f"Approach Y : {gt.approach_y_mm:.1f} mm",
                f"Approach Z : {gt.approach_z_mm:.1f} mm"
            ])

    for line in robot_lines:
        if line == "========== ROBOT ==========":
            cv2.putText(robot_panel, line,
                        (20,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1)
            y += step

            cv2.putText(robot_panel, f"State       : {state_text}",
                        (20,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_colour, 2)
            y += step
            continue

        if line.startswith("Book State"):
            cv2.putText(robot_panel, line,
                        (20,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, book_colour, 2)
            y += step
            continue

        cv2.putText(robot_panel, line,
                    (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1) 
        y += step

    cv2.imshow("Robot Controller", robot_panel)

    # --- KEYS ---
    simulator.update()
    k = cv2.waitKey(1) & 0xFF
    ctr = simulator.vis.get_view_control()
    if k == ord("q"):
        break
    elif k == ord("w"):
        conf_threshold = min(conf_threshold + 0.05, 0.95)
    elif k == ord("x"):
        conf_threshold = max(conf_threshold - 0.05, 0.30)
    elif k == ord("f"):
        SHOW_FPS = not SHOW_FPS
    elif k == ord("c"):
        selected_det = None
        simulator.clear_target()
    elif k == ord("s") and selected_det:
        save_to_csv(selected_det)
        
    elif k == ord("1"): # Front
        ctr.set_front([0,0,-1])
        ctr.set_up([0,1,0])
        ctr.set_lookat([0,500,450]) # H, V, D
        ctr.set_zoom(0.55) # 0.5 nearer, 0.6 further
    elif k == ord("2"): # Top
        ctr.set_front([0,1,0])
        ctr.set_up([0,0,1])
        ctr.set_lookat([0,300,350]) 
        ctr.set_zoom(0.55)
    elif k == ord("3"): # Left
        ctr.set_front([1,0,0])
        ctr.set_up([0,1,0])
        ctr.set_lookat([0,300,317])
        ctr.set_zoom(0.60)
    elif k == ord("4"): # Right
        ctr.set_front([-1,0,0])
        ctr.set_up([0,1,0])
        ctr.set_lookat([0,300,317])
        ctr.set_zoom(0.60)
    elif k == ord("5"): # Iso
        ctr.set_front([-0.6,0.45,-0.6])
        ctr.set_up([0,1,0])
        ctr.set_lookat([0,300,450])
        ctr.set_zoom(0.65)

if USE_REALSENSE:
    pipeline.stop()
if cap:
    cap.release()
cv2.destroyAllWindows()
