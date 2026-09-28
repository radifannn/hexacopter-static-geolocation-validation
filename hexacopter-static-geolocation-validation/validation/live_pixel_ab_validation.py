import math
import time

import cv2
import numpy as np

import geolocation_geometry as geo
import static_batch_validation as batch
import camera_origin_diagnostic as cod


# ============================================================
# Experiment configuration
# ============================================================

N_SAMPLES = 100
TARGET_Z = 0.01

# Same camera stream used by red_target_detector.py
PIPELINE = (
    'udpsrc port=5600 caps="application/x-rtp, media=video, '
    'encoding-name=H264, payload=96" ! '
    'rtph264depay ! h264parse ! avdec_h264 ! '
    'videoconvert ! appsink'
)

# HSV thresholds copied from red_target_detector.py
LOWER_RED_1 = np.array([0, 100, 80])
UPPER_RED_1 = np.array([10, 255, 255])

LOWER_RED_2 = np.array([170, 100, 80])
UPPER_RED_2 = np.array([180, 255, 255])

KERNEL = np.ones((3, 3), np.uint8)


# ============================================================
# Helpers
# ============================================================

def find_target_pose(poses):

    for name, pose in poses.items():

        if "moving_target" in name:
            return pose

    raise RuntimeError(
        f"Target pose not found. Available: {list(poses)}"
    )


def largest_contour(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0.0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 10:
            continue

        if area > best_area:

            best = contour
            best_area = area

    return best


def get_measurements(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask1 = cv2.inRange(
        hsv,
        LOWER_RED_1,
        UPPER_RED_1
    )

    mask2 = cv2.inRange(
        hsv,
        LOWER_RED_2,
        UPPER_RED_2
    )

    raw_mask = cv2.bitwise_or(
        mask1,
        mask2
    )

    opened_mask = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_OPEN,
        KERNEL
    )

    closed_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        KERNEL
    )

    contour = largest_contour(
        closed_mask
    )

    if contour is None:
        return None

    x, y, w, h = cv2.boundingRect(
        contour
    )

    # Match the existing detector:
    # x + w // 2
    bbox_u = float(
        x + w // 2
    )

    bbox_v = float(
        y + h // 2
    )

    moments = cv2.moments(
        contour
    )

    if moments["m00"] == 0:
        return None

    centroid_u = (
        moments["m10"]
        / moments["m00"]
    )

    centroid_v = (
        moments["m01"]
        / moments["m00"]
    )

    return {
        "bbox": (x, y, w, h),
        "bbox_pixel": (
            bbox_u,
            bbox_v
        ),
        "centroid_pixel": (
            centroid_u,
            centroid_v
        ),
        "area": cv2.contourArea(contour)
    }


def get_camera_geometry():

    # --------------------------------------------------------
    # UAV
    # --------------------------------------------------------

    uav_text = cod.run_gz_model(
        "hexacopter_with_ardupilot"
    )

    uav_pos, uav_rpy = cod.parse_named_pose(
        uav_text,
        "hexacopter_with_ardupilot"
    )

    # --------------------------------------------------------
    # Gimbal
    # --------------------------------------------------------

    gimbal_text = cod.run_gz_model(
        "gimbal"
    )

    gimbal_pos, gimbal_rpy = cod.parse_named_pose(
        gimbal_text,
        "gimbal"
    )

    # --------------------------------------------------------
    # Pitch link
    # --------------------------------------------------------

    pitch_text = cod.run_gz_model(
        "gimbal",
        "pitch_link"
    )

    pitch_pos, pitch_rpy = cod.parse_named_pose(
        pitch_text,
        "pitch_link"
    )

    # --------------------------------------------------------
    # Rotation hierarchy
    # --------------------------------------------------------

    R_world_uav = cod.rpy_to_R(
        *uav_rpy
    )

    R_uav_gimbal = cod.rpy_to_R(
        *gimbal_rpy
    )

    R_gimbal_pitch = cod.rpy_to_R(
        *pitch_rpy
    )

    R_world_pitch = (
        R_world_uav
        @ R_uav_gimbal
        @ R_gimbal_pitch
    )

    # --------------------------------------------------------
    # Camera optical center
    #
    # Camera translation relative pitch_link = zero.
    # --------------------------------------------------------

    camera_pos_uav = (
        gimbal_pos
        + R_uav_gimbal @ pitch_pos
    )

    camera_world = (
        uav_pos
        + R_world_uav @ camera_pos_uav
    )

    return (
        uav_pos,
        camera_world,
        R_world_pitch,
        uav_rpy,
        gimbal_rpy,
        pitch_rpy
    )


def pixel_to_ned_ray(
    u,
    v,
    R_world_pitch
):

    # --------------------------------------------------------
    # Pixel -> Python camera frame
    # --------------------------------------------------------

    ray_camera = geo.pixel_to_camera_ray(
        u,
        v
    )

    # --------------------------------------------------------
    # Python camera -> Gazebo camera
    # --------------------------------------------------------

    ray_gazebo_camera = (
        geo.R_GAZEBO_FROM_CAMERA
        @ ray_camera
    )

    # --------------------------------------------------------
    # Gazebo camera -> pitch_link
    # --------------------------------------------------------

    ray_pitch = (
        geo.R_SENSOR
        @ ray_gazebo_camera
    )

    # --------------------------------------------------------
    # pitch_link -> world
    #
    # IMPORTANT:
    # Full local hierarchy.
    # No static_pitch_from_quaternion().
    # No separate MAVLink attitude.
    # --------------------------------------------------------

    ray_world = (
        R_world_pitch
        @ ray_pitch
    )

    # --------------------------------------------------------
    # Gazebo world -> NED
    # --------------------------------------------------------

    R_ned_from_gazebo = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ], dtype=float)

    ray_ned = (
        R_ned_from_gazebo
        @ ray_world
    )

    norm = np.linalg.norm(
        ray_ned
    )

    if norm == 0:
        raise RuntimeError(
            "Invalid NED ray."
        )

    return ray_ned / norm


def estimate_from_pixel(
    u,
    v,
    camera_world,
    uav_world,
    target_world,
    R_world_pitch
):

    ray_ned = pixel_to_ned_ray(
        u,
        v,
        R_world_pitch
    )

    if ray_ned[2] <= 0:
        return None

    # --------------------------------------------------------
    # Camera position relative UAV in NED
    # --------------------------------------------------------

    camera_delta = (
        camera_world
        - uav_world
    )

    camera_ned = np.array([
        camera_delta[1],
        camera_delta[0],
        -camera_delta[2]
    ])

    # --------------------------------------------------------
    # Camera -> target plane vertical distance
    # --------------------------------------------------------

    camera_to_plane_down = (
        camera_world[2]
        - TARGET_Z
    )

    if camera_to_plane_down <= 0:
        return None

    scale = (
        camera_to_plane_down
        / ray_ned[2]
    )

    estimated_ned = (
        camera_ned
        + scale * ray_ned
    )

    # --------------------------------------------------------
    # True target NED relative UAV
    # --------------------------------------------------------

    target_delta = (
        target_world
        - uav_world
    )

    true_target_ned = np.array([
        target_delta[1],
        target_delta[0],
        -target_delta[2]
    ])

    error_n = (
        estimated_ned[0]
        - true_target_ned[0]
    )

    error_e = (
        estimated_ned[1]
        - true_target_ned[1]
    )

    error_h = math.hypot(
        error_n,
        error_e
    )

    return {
        "ray_ned": ray_ned,
        "estimated": estimated_ned,
        "true": true_target_ned,
        "error_n": error_n,
        "error_e": error_e,
        "error_h": error_h
    }


# ============================================================
# Statistics
# ============================================================

def print_statistics(
    name,
    results,
    pixels
):

    errors_n = np.array([
        r["error_n"]
        for r in results
    ])

    errors_e = np.array([
        r["error_e"]
        for r in results
    ])

    errors_h = np.array([
        r["error_h"]
        for r in results
    ])

    pixel_array = np.array(
        pixels
    )

    print()
    print("=" * 65)
    print(name)
    print("=" * 65)

    print()
    print("Pixel statistics:")

    print(
        f"  Mean u = "
        f"{np.mean(pixel_array[:, 0]):.4f}"
    )

    print(
        f"  Mean v = "
        f"{np.mean(pixel_array[:, 1]):.4f}"
    )

    print(
        f"  Std u  = "
        f"{np.std(pixel_array[:, 0]):.4f}"
    )

    print(
        f"  Std v  = "
        f"{np.std(pixel_array[:, 1]):.4f}"
    )

    print()
    print("Position error:")

    print(
        f"  North bias = "
        f"{np.mean(errors_n):+.4f} m"
    )

    print(
        f"  East bias  = "
        f"{np.mean(errors_e):+.4f} m"
    )

    print(
        f"  North MAE  = "
        f"{np.mean(np.abs(errors_n)):.4f} m"
    )

    print(
        f"  East MAE   = "
        f"{np.mean(np.abs(errors_e)):.4f} m"
    )

    print(
        f"  North RMSE = "
        f"{math.sqrt(np.mean(errors_n ** 2)):.4f} m"
    )

    print(
        f"  East RMSE  = "
        f"{math.sqrt(np.mean(errors_e ** 2)):.4f} m"
    )

    print(
        f"  Horizontal MAE = "
        f"{np.mean(errors_h):.4f} m"
    )

    print(
        f"  Horizontal RMSE = "
        f"{math.sqrt(np.mean(errors_h ** 2)):.4f} m"
    )

    print(
        f"  Horizontal min = "
        f"{np.min(errors_h):.4f} m"
    )

    print(
        f"  Horizontal max = "
        f"{np.max(errors_h):.4f} m"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LIVE 100-SAMPLE PIXEL MEASUREMENT A/B VALIDATION")
    print("=" * 70)

    print()
    print("Requirements:")
    print("  UAV hovering at approximately 10 m")
    print("  Target stationary at (2, 5, 0.01)")
    print("  Gimbal approximately at RC7 = 1400")
    print()

    cap = cv2.VideoCapture(
        PIPELINE,
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera stream."
        )

    # Allow camera stream to initialize.
    time.sleep(1.0)

    bbox_results = []
    centroid_results = []

    bbox_pixels = []
    centroid_pixels = []

    attempts = 0
    valid_samples = 0

    print("Collecting samples...")

    while valid_samples < N_SAMPLES:

        ret, frame = cap.read()

        if not ret:
            continue

        attempts += 1

        measurement = get_measurements(
            frame
        )

        if measurement is None:
            continue

        try:

            poses = batch.read_gazebo_state()

            uav_pose = poses[
                "hexacopter_with_ardupilot"
            ]

            target_pose = find_target_pose(
                poses
            )

            uav_world = uav_pose["position"]
            target_world = target_pose["position"]

            (
                uav_check_pos,
                camera_world,
                R_world_pitch,
                _,
                _,
                _
            ) = get_camera_geometry()

            # ------------------------------------------------
            # BBOX CENTER
            # ------------------------------------------------

            bbox_u, bbox_v = (
                measurement["bbox_pixel"]
            )

            bbox_estimate = (
                estimate_from_pixel(
                    bbox_u,
                    bbox_v,
                    camera_world,
                    uav_check_pos,
                    target_world,
                    R_world_pitch
                )
            )

            # ------------------------------------------------
            # CONTOUR CENTROID
            # ------------------------------------------------

            centroid_u, centroid_v = (
                measurement["centroid_pixel"]
            )

            centroid_estimate = (
                estimate_from_pixel(
                    centroid_u,
                    centroid_v,
                    camera_world,
                    uav_check_pos,
                    target_world,
                    R_world_pitch
                )
            )

            if (
                bbox_estimate is None
                or centroid_estimate is None
            ):
                continue

        except Exception:

            continue

        bbox_results.append(
            bbox_estimate
        )

        centroid_results.append(
            centroid_estimate
        )

        bbox_pixels.append(
            (bbox_u, bbox_v)
        )

        centroid_pixels.append(
            (centroid_u, centroid_v)
        )

        valid_samples += 1

        if (
            valid_samples == 1
            or valid_samples % 10 == 0
        ):

            print(
                f"  Sample "
                f"{valid_samples:3d}/"
                f"{N_SAMPLES}"
                f" | bbox=({bbox_u:.2f},"
                f"{bbox_v:.2f})"
                f" | centroid=({centroid_u:.2f},"
                f"{centroid_v:.2f})"
            )

    cap.release()

    print()
    print(
        f"Collected {valid_samples} valid samples "
        f"after {attempts} frames."
    )

    if valid_samples < N_SAMPLES:

        raise RuntimeError(
            "Could not collect enough valid samples."
        )

    print_statistics(
        "A — BOUNDING-BOX CENTER",
        bbox_results,
        bbox_pixels
    )

    print_statistics(
        "B — CONTOUR CENTROID",
        centroid_results,
        centroid_pixels
    )


if __name__ == "__main__":
    main()
