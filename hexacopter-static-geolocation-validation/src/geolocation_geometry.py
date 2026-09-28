import math
import numpy as np


# ============================================================
# Camera intrinsics
# ============================================================

WIDTH = 640
HEIGHT = 480
HFOV = 2.0  # radians

FX = 205.46962738037109
FY = 205.46965599060059

CX = 320.0
CY = 240.0

GROUND_Z = 0.0

# ============================================================
# Rotation matrices
# ============================================================

def Rx(angle):
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=float)


def Ry(angle):
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=float)


def Rz(angle):
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=float)


# ============================================================
# Pixel -> camera ray
#
# Simplified camera convention used by our detector:
#   X = right
#   Y = down
#   Z = forward
# ============================================================

def pixel_to_camera_ray(u, v):

    ray = np.array([
        (u - CX) / FX,
        (v - CY) / FY,
        1.0
    ], dtype=float)

    norm = np.linalg.norm(ray)

    if norm == 0:
        raise ValueError("Invalid camera ray.")

    return ray / norm


# ============================================================
# Camera convention -> Gazebo camera convention
#
# Simplified:
#   X = right
#   Y = down
#   Z = forward
#
# Gazebo camera:
#   X = forward
#   Y = left
#   Z = up
# ============================================================

R_GAZEBO_FROM_CAMERA = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
], dtype=float)


# ============================================================
# Fixed transformations from the current SDF
# ============================================================

R_OUTER = (
    Rz(math.radians(90.0))
    @ Rx(math.radians(90.0))
)

R_SENSOR = (
    Ry(math.radians(-90.0))
    @ Rx(math.radians(-90.0))
)


# Gazebo body:
#   X = forward
#   Y = left
#   Z = up
#
# ArduPilot body:
#   X = forward
#   Y = right
#   Z = down

R_ARDUPILOT_FROM_GAZEBO = np.diag([
    1.0,
    -1.0,
    -1.0
])


# ============================================================
# Quaternion -> rotation matrix
# ============================================================

def quaternion_to_rotation_matrix(q):

    q = np.asarray(q, dtype=float)

    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,).")

    x, y, z, w = q

    norm = np.linalg.norm(q)

    if norm == 0:
        raise ValueError("Quaternion cannot be zero.")

    x, y, z, w = q / norm

    return np.array([
        [
            1 - 2 * (y*y + z*z),
            2 * (x*y - z*w),
            2 * (x*z + y*w)
        ],
        [
            2 * (x*y + z*w),
            1 - 2 * (x*x + z*z),
            2 * (y*z - x*w)
        ],
        [
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x*x + y*y)
        ]
    ], dtype=float)


# ============================================================
# STATIC TEST ONLY:
# extract the active pitch-link angle
#
# This is intentionally NOT our final 2-DOF gimbal solution.
# It is valid for the current static experiment because the
# measured pitch_link quaternion is essentially a pure X-axis
# rotation.
# ============================================================

def static_pitch_from_quaternion(q):

    q = np.asarray(q, dtype=float)

    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,).")

    x, _, _, w = q

    angle = 2.0 * math.atan2(x, w)

    return angle


# ============================================================
# Camera ray -> NED ray
#
# Current static experiment:
#   roll gimbal = fixed
#   yaw gimbal  = fixed
#   pitch       = measured pitch_link angle
# ============================================================

def camera_ray_to_ned_static(
    ray_camera,
    pitch_link_quaternion,
    roll_uav,
    pitch_uav,
    yaw_uav
):

    ray_camera = np.asarray(ray_camera, dtype=float)

    if ray_camera.shape != (3,):
        raise ValueError("ray_camera must have shape (3,).")

    # 1. Simplified camera -> Gazebo camera
    ray_gazebo_camera = (
        R_GAZEBO_FROM_CAMERA @ ray_camera
    )

    # 2. Current static gimbal pitch
    pitch_angle = static_pitch_from_quaternion(
        pitch_link_quaternion
    )

    R_pitch = Rx(pitch_angle)

    # 3. Camera -> Gazebo body
    R_gazebo_body_from_camera = (
        R_OUTER
        @ R_pitch
        @ R_SENSOR
    )

    ray_gazebo_body = (
        R_gazebo_body_from_camera
        @ ray_gazebo_camera
    )

    # 4. Gazebo body -> ArduPilot body
    ray_body = (
        R_ARDUPILOT_FROM_GAZEBO
        @ ray_gazebo_body
    )

    # 5. ArduPilot body -> NED
    R_ned_from_body = (
        Rz(yaw_uav)
        @ Ry(pitch_uav)
        @ Rx(roll_uav)
    )

    ray_ned = (
        R_ned_from_body
        @ ray_body
    )

    norm = np.linalg.norm(ray_ned)

    if norm == 0:
        raise ValueError("Invalid NED ray.")

    return ray_ned / norm


# ============================================================
# Ray -> ground intersection
# ============================================================

def ray_plane_intersection(ray_ned, vertical_distance_m):
    """
    Intersect a NED ray with a horizontal target plane.

    NED:
        Z = Down

    vertical_distance_m:
        Positive distance from UAV to the target plane.
    """

    ray_ned = np.asarray(ray_ned, dtype=float)

    if ray_ned.shape != (3,):
        raise ValueError("ray_ned must have shape (3,)")

    if vertical_distance_m <= 0:
        raise ValueError(
            "vertical_distance_m must be > 0."
        )

    if ray_ned[2] <= 0:
        raise ValueError(
            "Ray does not intersect the horizontal plane."
        )

    scale = (
        vertical_distance_m
        / ray_ned[2]
    )

    return scale * ray_ned


def ray_ground_intersection(ray_ned, height_m):
    """
    Backward-compatible wrapper for ground plane Z=0.
    """

    return ray_plane_intersection(
        ray_ned,
        height_m
    )

# ============================================================
# NED horizontal displacement -> GPS
# ============================================================

def ned_to_gps(
    lat_uav,
    lon_uav,
    north_m,
    east_m
):

    EARTH_RADIUS = 6378137.0

    lat_rad = math.radians(lat_uav)

    delta_lat = north_m / EARTH_RADIUS

    delta_lon = (
        east_m
        / (EARTH_RADIUS * math.cos(lat_rad))
    )

    target_lat = (
        lat_uav
        + math.degrees(delta_lat)
    )

    target_lon = (
        lon_uav
        + math.degrees(delta_lon)
    )

    return target_lat, target_lon


# ============================================================
# GPS error in local North/East meters
# ============================================================

def gps_error_m(
    estimated_lat,
    estimated_lon,
    reference_lat,
    reference_lon
):

    EARTH_RADIUS = 6378137.0

    north_error = (
        math.radians(
            estimated_lat - reference_lat
        )
        * EARTH_RADIUS
    )

    east_error = (
        math.radians(
            estimated_lon - reference_lon
        )
        * EARTH_RADIUS
        * math.cos(
            math.radians(reference_lat)
        )
    )

    horizontal_error = math.hypot(
        north_error,
        east_error
    )

    return (
        north_error,
        east_error,
        horizontal_error
    )
