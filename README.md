# Object-Centric Robot Imitation Learning

A robotics research platform for real-world manipulation using VR teleoperation, object-centric perception, transformer-based imitation learning, and optimization-based robot control.

---

## Overview

This repository contains the software infrastructure, training pipelines, and experimental research code used to study robotic manipulation from demonstration.

The project began as an exploration of whether explicit object-centric representations could improve the performance and robustness of transformer-based imitation learning policies. Over time, it evolved into a broader platform supporting data collection, perception, policy learning, robot control, and inverse kinematics research on real robotic hardware.

Current research directions include:

* Object-centric visual conditioning for imitation learning
* Segmentation-assisted robot perception
* Transformer-based visuomotor policies
* Demonstration learning from VR teleoperation
* Real-time robotic manipulation
* Optimization-based inverse kinematics
* Learned and model-based control integration

---

## System Architecture

The platform consists of four primary components:

### 1. VR Teleoperation

Human demonstrations are collected through immersive teleoperation using a VR headset and handheld controllers.

The teleoperation stack supports:

* Real-time robot control
* Episode recording
* Demonstration replay
* Data synchronization
* Multi-camera capture

Demonstrations are automatically logged and converted into datasets suitable for imitation learning.

### 2. Perception

The robot observes the environment through multiple RGB and RGB-D cameras, including both wrist-mounted and external viewpoints.

The perception pipeline supports:

* RGB observations
* RGB-D observations
* Object segmentation
* Object localization
* Geometric feature extraction
* Dataset annotation and validation

Object-centric representations are generated using segmentation masks and geometric scene descriptors derived from detected objects.

### 3. Imitation Learning

Policies are trained using Action Chunking with Transformers (ACT).

Rather than predicting a single control command at each timestep, ACT predicts short sequences of future actions, enabling more stable and temporally coherent behavior.

The training stack includes:

* Dataset preprocessing
* ACT training
* Hyperparameter sweeps
* Rollout evaluation
* Experiment tracking
* Policy benchmarking

Supported visual representations include:

* Raw RGB observations
* Segmentation masks
* Masked RGB inputs
* Object centroids
* Object-target vectors
* Hybrid object-centric encodings

### 4. Robot Control

The repository contains interfaces and utilities for controlling physical robot hardware, including:

* Joint-space control
* Trajectory execution
* State estimation
* Gripper control
* Real-time policy deployment

---

## Object-Centric Visual Conditioning

A major focus of this repository is the study of object-centric representations for robotic imitation learning.

Traditional behavioral cloning systems must simultaneously learn perception, object identification, and control directly from pixels. This work investigates whether explicitly representing task-relevant objects can simplify learning and improve robustness.

Representations explored include:

* Segmentation masks
* Object centroids
* Direction vectors
* Geometric scene descriptors
* Hybrid visual-geometric representations

Experiments on real-world manipulation tasks suggest that object-centric conditioning improves object localization, reduces unnecessary corrective behavior, and can outperform purely pixel-based representations despite requiring significantly less information.

---

## Segmentation Pipeline

To generate object-centric observations, we developed a semi-automated segmentation pipeline.

The annotation workflow combines:

* Foundation-model segmentation
* Classical computer vision techniques
* Scene-specific heuristics
* Automated quality control
* YOLO fine-tuning

The resulting segmentation models run in real time and provide masks that can be used both during training and deployment.

This allows task-relevant scene structure to be made explicit to the policy while maintaining practical deployment constraints.

---

## Experimental Inverse Kinematics Research

The repository also contains ongoing work on optimization-based inverse kinematics and trajectory generation.

Current investigations include:

* Trajectory-constrained IK
* Collision-aware optimization
* Manipulability objectives
* Bimanual IK formulations
* JAX-based optimization pipelines
* Integration of learned policies with IK-based controllers

This work is currently experimental and under active development.

---

## Hardware Platform

Current experiments are conducted on:

* Interbotix robotic manipulators
* Parallel-jaw grippers
* Intel RealSense RGB-D cameras
* Wrist-mounted cameras
* External overhead cameras
* VR headsets and handheld controllers

---

## Repository Structure

```text
interbotix_ws/
├── src/
│   ├── av_aloha/
│   │   ├── data_collection_scripts/
│   │   ├── teleoperation/
│   │   ├── perception/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── robot_control/
│   ├── lerobot/
│   └── interbotix_ros_*
│
├── datasets/
├── checkpoints/
├── experiments/
└── outputs/
```

---

## Research Contributions

This repository has been used to investigate:

* Object-centric visual conditioning for ACT policies
* Segmentation-assisted imitation learning
* Real-world robot manipulation from demonstration
* Geometric scene representations for control
* Teleoperation-driven data collection pipelines

Recent experiments suggest that low-dimensional object-centric representations based on centroids and geometric relationships can rival or outperform richer visual inputs on manipulation tasks while requiring substantially less computation.

---

## Future Work

Ongoing research directions include:

* Multi-object manipulation
* Task-conditioned policies
* Improved grasp planning
* Foundation-model-assisted perception
* Learned world models
* Integration of optimization-based IK with learned policies
* Generalization across objects, scenes, and tasks

---
## Associated Research

This repository accompanies ongoing research in robot imitation learning, object-centric perception, and real-world manipulation.

If you use this repository in academic work, please cite the associated publications when available.

---
config.py
    Defines system constants.

robot_factory.py
    Creates robot objects.

arm_controller.py
    Commands arm motion.

gripper.py
    Commands gripper motion.

camera_manager.py
    Manages camera streams.

teleop_utils.py
    Math and transformations.
