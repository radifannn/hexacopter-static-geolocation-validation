import math
import time

import cv2
import numpy as np

import geolocation_geometry as geo
import static_batch_validation as batch
import camera_origin_diagnostic as cod


# ============================================================
# Configuration
# ============================================================

TARGET_Z = 0.01

PIPELINE = (
    'udpsrc port=5600 caps="application/x-rtp, media=video, '
    'encoding-name=H264, payload=96" ! '
    'rtph264depay ! h264parse ! avdec_h264 ! '
    'videoconvert ! appsink drop=true sync=false'
)

LOWER_RED_1 = np.array([0, 100, 80])
UPPER_RED_1 = np.array([10, 255, 255])

LOWER_RED_2 = np.array([170, 100, 80])
UPPER_RED_2 = np.array([180, 255, 255])

KERNEL = np.ones((3, 3), np.uint8)


# ============================================================
# Detection
# ============================================================

def detect_target(frame):

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

    mask = cv2.bitwise_or(
        mask1,
        mask2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        KERNEL
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        KERNEL
    )

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

    if best is None:
        return None

    x, y, w, h = cv2.boundingRect(best)

    bbox_center = (
        float(x + w // 2),
        float(y + h // 2)
    )

    moments = cv2.moments(best)

    if moments["m00"] == 0:
        return None

    centroid = (
        moments["m10"] / moments["m00"],
        moments["m01"] / moments["m00"]
    )

    return {
        "contour": best,
        "bbox": (x, y, w, h),
        "bbox_center": bbox_center,
        "centroid": centroid,
        "area": best_area
    }


# ============================================================
# Geometry
# ============================================================

def get_world_camera_geometry():

    # UAV
    uav_text = cod.run_gz_model(
        "hexacopter_with_ardupilot"
    )

    uav_pos_model, uav_rpy = cod.parse_named_pose(
        uav_text,
        "hexacopter_with_ardupilot"
    )

    # Gimbal relative to UAV
    gimbal_text = cod.run_gz_model(
        "gimbal"
    )

    gimbal_pos, gimbal_rpy = cod.parse_named_pose(
        gimbal_text,
        "gimbal"
    )

    # Pitch link relative to gimbal
    pitch_text = cod.run_gz_model(
        "gimbal",
        "pitch_link"
    )

    pitch_pos, pitch_rpy = cod.parse_named_pose(
        pitch_text,
        "pitch_link"
    )

    # Local rotation hierarchy
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

    # Camera position relative to UAV
    camera_uav = (
        gimbal_pos
        + R_uav_gimbal @ pitch_pos
    )

    camera_world = (
        uav_pos_model
        + R_world_uav @ camera_uav
    )

    return (
        uav_pos_model,
        camera_world,
        R_world_pitch
    )


def project_world_point_to_pixel(
    world_point,
    camera_world,
    R_world_pitch
):

    # World -> pitch_link
    point_pitch = (
        R_world_pitch.T
        @ (world_point - camera_world)
    )

    # pitch_link -> Gazebo camera
    point_gazebo_camera = (
        geo.R_SENSOR.T
        @ point_pitch
    )

    # Gazebo camera -> simplified camera convention
    point_camera = (
        geo.R_GAZEBO_FROM_CAMERA.T
        @ point_gazebo_camera
    )

    x, y, z = point_camera

    if z <= 0:
        return None

    u = (
        geo.FX * x / z
        + geo.CX
    )

    v = (
        geo.FY * y / z
        + geo.CY
    )

    return (
        float(u),
        float(v),
        float(z)
    )


def find_target_pose(poses):

    for name, pose in poses.items():

        if "moving_target" in name:
            return pose

    raise RuntimeError(
        "moving_target was not found."
    )


# ============================================================
# Drawing
# ============================================================

def draw_cross(
    image,
    point,
    color,
    label,
    size=8
):

    u, v = point

    ui = int(round(u))
    vi = int(round(v))

    cv2.line(
        image,
        (ui - size, vi),
        (ui + size, vi),
        color,
        2
    )

    cv2.line(
        image,
        (ui, vi - size),
        (ui, vi + size),
        color,
        2
    )

    cv2.putText(
        image,
        label,
        (ui + 8, vi - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PIXEL PROJECTION OVERLAY TEST")
    print("=" * 70)
    print()
    print("Press Q to quit.")
    print()

    cap = cv2.VideoCapture(
        PIPELINE,
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera stream."
        )

    time.sleep(1.0)

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        # --------------------------------------------
        # Detect target
        # --------------------------------------------

        detection = detect_target(
            frame
        )

        if detection is None:

            cv2.putText(
                frame,
                "TARGET NOT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(
                "Pixel Projection Overlay",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        # --------------------------------------------
        # Get current Gazebo state
        # --------------------------------------------

        try:

            poses = batch.read_gazebo_state()

            target_pose = find_target_pose(
                poses
            )

            target_world = target_pose[
                "position"
            ]

            (
                uav_world,
                camera_world,
                R_world_pitch
            ) = get_world_camera_geometry()

        except Exception:

            continue

        # --------------------------------------------
        # Theoretical projection
        # --------------------------------------------

        theoretical_pixel = (
            project_world_point_to_pixel(
                target_world,
                camera_world,
                R_world_pitch
            )
        )

        if theoretical_pixel is None:
            continue

        theoretical_u = theoretical_pixel[0]
        theoretical_v = theoretical_pixel[1]

        # --------------------------------------------
        # Measured pixels
        # --------------------------------------------

        bbox_u, bbox_v = (
            detection["bbox_center"]
        )

        centroid_u, centroid_v = (
            detection["centroid"]
        )

        # --------------------------------------------
        # Draw bounding box
        # --------------------------------------------

        x, y, w, h = detection["bbox"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # --------------------------------------------
        # Draw bbox center
        # --------------------------------------------

        draw_cross(
            frame,
            (bbox_u, bbox_v),
            (255, 255, 0),
            "BBox"
        )

        # --------------------------------------------
        # Draw contour centroid
        # --------------------------------------------

        draw_cross(
            frame,
            (centroid_u, centroid_v),
            (0, 255, 255),
            "Centroid"
        )

        # --------------------------------------------
        # Draw theoretical projection
        # --------------------------------------------

        draw_cross(
            frame,
            (theoretical_u, theoretical_v),
            (0, 0, 255),
            "Theory"
        )

        # --------------------------------------------
        # Connect centroid -> theoretical projection
        # --------------------------------------------

        cv2.line(
            frame,
            (
                int(round(centroid_u)),
                int(round(centroid_v))
            ),
            (
                int(round(theoretical_u)),
                int(round(theoretical_v))
            ),
            (255, 0, 255),
            1
        )

        # --------------------------------------------
        # Pixel differences
        # --------------------------------------------

        du = (
            centroid_u
            - theoretical_u
        )

        dv = (
            centroid_v
            - theoretical_v
        )

        pixel_error = math.hypot(
            du,
            dv
        )

        # --------------------------------------------
        # Information panel
        # --------------------------------------------

        lines = [
            f"BBox      : ({bbox_u:.2f}, {bbox_v:.2f})",
            f"Centroid  : ({centroid_u:.2f}, {centroid_v:.2f})",
            f"Theory    : ({theoretical_u:.2f}, {theoretical_v:.2f})",
            f"Delta     : ({du:+.2f}, {dv:+.2f}) px",
            f"Pixel err : {pixel_error:.2f} px",
        ]

        y_text = 25

        for line in lines:

            cv2.putText(
                frame,
                line,
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            y_text += 22

        # --------------------------------------------
        # Show
        # --------------------------------------------

        cv2.imshow(
            "Pixel Projection Overlay",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
