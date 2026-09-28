# Hexacopter Ground-Target Geolocation — Static Geometric Validation

This repository contains the **static geometric ground-target geolocation implementation and multi-configuration simulation validation data** for the hexacopter ground-target geolocation project.

This repository continues the development documented in the previous repository:

**Previous repository:**  
https://github.com/radifannn/hexacopter-ground-target-geolocation_2.0

The current stage focuses on validating the geometric geolocation method under different UAV positions, altitudes, viewing geometries, UAV headings, and gimbal configurations before moving to EKF-based target-state estimation.

---

# 1. Project Objective

The overall project aims to estimate the geographic position of a ground target observed by a UAV-mounted camera.

The current implementation focuses specifically on the **static geometric geolocation stage**.

The estimator receives:

- Camera image
- Detected target pixel position
- UAV/gimbal/camera geometric state
- UAV GPS position

and estimates:

- Target latitude
- Target longitude

The target altitude is **not estimated** in the current implementation.

---

# 2. Current Estimation Method

The current estimator is a deterministic geometric method.

The complete estimation pipeline is:

```text
Camera
   ↓
Target Detection
   ↓
Largest Valid Contour
   ↓
Contour Centroid (u, v)
   ↓
Camera Intrinsics
   ↓
Camera Ray
   ↓
Camera / Pitch Link / Gimbal / UAV Transformations
   ↓
Gazebo World → NED
   ↓
Ground-Plane Ray Intersection
   ↓
Target Position Relative to UAV
   ↓
UAV GPS Georeferencing
   ↓
Estimated Target Latitude / Longitude
