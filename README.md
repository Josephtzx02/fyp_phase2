# 3D Book Retrieval Robot Simulator
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Open3D](https://img.shields.io/badge/Open3D-Simulation-success)
![YOLOv8-OBB](https://img.shields.io/badge/YOLOv8-OBB-orange)
![Intel RealSense](https://img.shields.io/badge/Intel-RealSense-0071C5)
![Robot](https://img.shields.io/badge/Robot-4DOF-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> Final Year Project (FYP) – Phase 2
>
> Bachelor of Manufacturing Engineering with Management (Hons.)
>
> Universiti Sains Malaysia (USM)

---

# Overview

This repository contains the robotic manipulation and simulation framework developed during the second phase of my Final Year Project.

Unlike Phase 1, which focuses on computer vision and machine learning, this repository demonstrates how the detected books are transformed into robotic grasp targets and executed within a complete Open3D robotic simulation.

The simulator receives perception outputs from the Phase 1 detection system and performs:

- Coordinate transformation
- Robot world calibration
- Grasp target generation
- Inverse kinematics
- Forward kinematics
- Mobile robot positioning
- Open3D visualization
- Automated book retrieval simulation

The complete system demonstrates the entire perception-to-manipulation pipeline for an intelligent robotic book retrieval system.

---

# Relationship with Phase 1

The complete project consists of two repositories.

```
Phase 1
RGB-D Vision
YOLOv8n-OBB
Machine Learning
        │
        ▼
Phase 2
Robot Simulation
Motion Planning
Open3D Visualization
Book Retrieval
```

Phase 1 Repository:

> 3D Scene Analysis and Robotic Vision for Intelligent Book Retrieval Systems

Phase 2 Repository:

> 3D Book Retrieval Robot Simulator

---

# System Architecture

```
YOLO OBB Detection
        │
        ▼
coordinate_transform.py
(Pixel → Camera Coordinate)
        │
        ▼
world_transform.py
(Camera → Robot World)
        │
        ▼
grasp_target.py
(Build GraspTarget Object)
        │
        ▼
simulator.py
(Main Robot Controller)
        │
        ├── inverse_kinematics.py
        │
        └── robot_arm.py
        │
        ▼
Open3D Mobile Manipulator Simulation
```

---

# Processing Pipeline

## Stage 1

coordinate_transform.py

Pixel Coordinate

↓

Camera Coordinate

---

## Stage 2

world_transform.py

Camera Coordinate

↓

Robot World Coordinate

---

## Stage 3

grasp_target.py

Robot World Coordinate

↓

Unified GraspTarget

---

## Stage 4

inverse_kinematics.py

Cartesian Target

↓

Joint Angles

---

## Stage 5

robot_arm.py

Forward Kinematics

↓

Robot Visualization

---

## Stage 6

simulator.py

Robot Controller

Mobile Robot (AGV)

State Machine

Book Retrieval Logic

Motion Planning

---

## Stage 7

book_retrieval_system.py

Complete System Integration

- RealSense RGB-D Camera
- YOLOv8-OBB Detection
- Mouse Interaction
- OpenCV User Interface
- Open3D Robot and Book Retrieval Simulation

---

# Demonstration

The following sequence illustrates the complete perception-to-manipulation workflow implemented in this repository.

1. Detect books using YOLOv8-OBB.
2. Select a target book using the mouse.
3. Generate the robot grasp target and approach positions.
4. Move the mobile AGV laterally to align the robotic arm with the selected book.
5. Move the robotic arm to the approach position.
6. Insert the gripper into the bookshelf.
7. Grasp the selected book by closing the gripper.
8. Pull the book completely out from the shelf.
9. Lift the retrieved book safely above the bookshelf.
10. Return to the idle state, ready for the next retrieval task.

---

# Repository Structure

```
.
├── models/
│   ├── best.pt
│   ├── width_model.pkl
│   ├── width_features.pkl
│   ├── weight_features_D.pkl
│   └── weight_model_D_huber.pkl
│
├── legacy/
│   ├── fyp2.0_v1/
│   ├── fyp2.0_v2/
│   └── fyp2.0_v3/
│
├── coordinate_transform.py
├── world_transform.py
├── grasp_target.py
├── inverse_kinematics.py
├── robot_arm.py
├── simulator.py
├── book_retrieval_system.py
│
└── README.md
```

---

# File Description

## book_retrieval_system.py

Main application of the robotic simulation.

Functions include:

- Real-time RGB-D acquisition
- YOLOv8-OBB inference
- Book selection using mouse interaction
- Coordinate transformation
- Grasp target generation
- Robot simulation
- OpenCV operator interface
- Open3D visualization
- Real-time robot state monitoring

---

## coordinate_transform.py

Converts image pixel coordinates into camera coordinates using Intel RealSense intrinsic parameters.

```
Pixel Coordinate

↓

Camera Coordinate
```

---

## world_transform.py

Transforms camera coordinates into the robot world coordinate system using the calibrated camera pose.

```
Camera Coordinate

↓

Robot World Coordinate
```

---

## grasp_target.py

Creates a unified GraspTarget object containing all information required for robotic manipulation.

Including:

- Book Position
- Grasp Position
- Approach Position
- Book Dimensions
- Grasp Orientation

---

## inverse_kinematics.py

Computes the required robot joint angles from the desired grasp target.

Robot Joints

- Shoulder
- Elbow
- Wrist Pitch

The wrist roll is controlled independently for book orientation and therefore is not included in the inverse kinematics computation.

---

## robot_arm.py

Implements the robot forward kinematics and Open3D visualization.

Functions include:

- Robot joint modelling
- Link generation
- Gripper visualization
- Wrist rotation
- Mobile base movement

---

## simulator.py

Main robot controller.

Responsible for:

- State machine
- Book spawning
- Robot motion animation
- Mobile robot movement
- Grasp sequence
- Book retrieval
- Open3D scene management

Robot Sequence

```
Approach

↓

Insert

↓

Grasp

↓

Pull Out

↓

Lift

↓

Retrieved
```

---

# Robot Configuration

Robot Type

- Mobile Manipulator

Degrees of Freedom

- 4 Degrees of Freedom (4-DOF)

End Effector

- Parallel Gripper

Visualization

- Open3D

Motion Control

- Inverse Kinematics
- Forward Kinematics
- State Machine Controller

---

# Robot Coordinate System

The robot simulation adopts a right-handed coordinate system designed for a mobile library retrieval robot.

```
             +Y
             ↑
             │
             │
             │
             ●────────→ +Z
            /
           /
         +X
```

Where:

- **X-axis** : Mobile robot (AGV) left/right translation
- **Y-axis** : Vertical height
- **Z-axis** : Forward reach toward the bookshelf

The robotic arm is mounted on a mobile platform, allowing lateral movement along the X-axis while the arm performs manipulation in the Y-Z plane.

---

## Robot Kinematic Structure

```
          Wrist Roll (DOF 4)
                ○
                │
         Wrist Pitch (DOF 3)
                ○
                │
          Elbow Joint (DOF 2)
                ○
                │
        Shoulder Joint (DOF 1)
                ○
                │
         Mobile AGV (X-axis)
```

The mobile manipulator consists of a **4-DOF robotic arm** mounted on an Automated Guided Vehicle (AGV).

The manipulator includes:

1. Shoulder Joint
2. Elbow Joint
3. Wrist Pitch
4. Wrist Roll

The AGV provides lateral translation along the X-axis, while the robotic arm performs manipulation primarily within the Y-Z plane.

---

# Installation

Python Version

```
Python 3.10
```

Install the required libraries:

```bash
pip install open3d
pip install ultralytics
pip install pyrealsense2
pip install opencv-python
pip install numpy
```

---

# Running the Simulator

Execute the main application:

```bash
python book_retrieval_system.py
```

The program automatically performs:

- RGB-D acquisition
- YOLOv8-OBB inference
- Coordinate transformation
- Grasp target generation
- Robot simulation
- OpenCV visualization
- Open3D visualization

---

## Prerequisites

Before running the simulator, ensure that:

- Intel RealSense D435 is connected.
- The trained YOLOv8-OBB model (`best.pt`) is located inside the `models/` directory.
- The regression model files (`.pkl`) are available inside the `models/` directory.

---

# Current Features

- RGB-D perception
- YOLOv8 Oriented Bounding Box detection
- Coordinate transformation
- Robot world calibration
- Unified grasp target generation
- Mobile AGV movement (X-axis)
- 4-DOF robotic arm simulation
- Shoulder, elbow, wrist pitch and wrist roll control
- Inverse kinematics
- Forward kinematics
- Interactive Open3D visualization
- Book retrieval state machine
- Interactive robot control panel
- Real-time operator interface

---

# Legacy Versions

To preserve the complete development history of this project, previous prototype implementations have been retained in the following folders:

```
fyp2.0_v1/
fyp2.0_v2/
fyp2.0_v3/
```

These folders contain earlier development versions, experimental features, and intermediate implementations created during the design and testing process.

They are **not required** for the final robotic simulation and are retained solely for historical reference and reproducibility.

The Python files located in the project root directory represent the latest stable implementation and should be used for all future development and execution.

---

# Future Work

Possible future extensions include:

- Physical robot deployment
- ROS2 integration
- Motion planning using MoveIt
- Collision avoidance
- Dynamic obstacle avoidance
- Multi-book retrieval

---

# License

This repository is shared for academic and educational purposes.

Please provide appropriate attribution if any part of the source code or methodology is used in future research.
