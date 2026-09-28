import re
import subprocess
import threading
import time

import cv2
import numpy as np
from pymavlink import mavutil

import geolocation_geometry as geo


# ============================================================
# Configuration
# ============================================================

CAMERA_PIPELINE = (
    'udpsrc port=5600 caps="application/x-rtp, media=video, '
    'encoding-name=H264, payload=96" ! '
    'rtph264depay ! h264parse ! avdec_h264 ! '
    'videoconvert ! appsink drop=true sync=false'
)

POSE_TOPIC = (
    "/world/hexacopter_runway/dynamic_pose/info"
)

MAVLINK_CONNECTION = (
    "tcp:127.0.0.1:5762"
)

TARGET_Z = 0.01

WINDOW_NAME = (
    "Realtime Static Geolocation"
)

# Red detector thresholds
LOWER_RED_1 = np.array([
    0, 100, 80
])

UPPER_RED_1 = np.array([
    10, 255, 255
])

LOWER_RED_2 = np.array([
    170, 100, 80
])

UPPER_RED_2 = np.array([
    180, 255, 255
])

KERNEL = np.ones(
    (3, 3),
    np.uint8
)


# ============================================================
# Shared state
# ============================================================

state_lock = threading.Lock()

state = {
    # Gazebo
    "uav_world": None,
    "camera_world": None,
    "R_world_pitch": None,
    "pose_timestamp": None,

    # MAVLink GPS
    "gps_lat": None,
    "gps_lon": None,
    "gps_timestamp": None,

    # Diagnostics
    "pose_error": None,
    "gps_error": None,
}

stop_event = threading.Event()


# ============================================================
# Generic pose parser
# ============================================================

def parse_pose_block(block):

    name_match = re.search(
        r'name:\s*"([^"]+)"',
        block
    )

    if not name_match:
        return None

    name = name_match.group(1)

    position_match = re.search(
        r'position\s*\{(.*?)\}',
        block,
        re.DOTALL
    )

    orientation_match = re.search(
        r'orientation\s*\{(.*?)\}',
        block,
        re.DOTALL
    )

    if (
        position_match is None
        or orientation_match is None
    ):
        return None

    position_text = (
        position_match.group(1)
    )

    orientation_text = (
        orientation_match.group(1)
    )

    def value(
        pattern,
        source,
        default=0.0
    ):

        match = re.search(
            pattern,
            source
        )

        if match:
            return float(
                match.group(1)
            )

        return default

    return {
        "name": name,

        "position": np.array([
            value(
                r'x:\s*([-+0-9.eE]+)',
                position_text
            ),
            value(
                r'y:\s*([-+0-9.eE]+)',
                position_text
            ),
            value(
                r'z:\s*([-+0-9.eE]+)',
                position_text
            )
        ]),

        "quaternion": np.array([
            value(
                r'x:\s*([-+0-9.eE]+)',
                orientation_text
            ),
            value(
                r'y:\s*([-+0-9.eE]+)',
                orientation_text
            ),
            value(
                r'z:\s*([-+0-9.eE]+)',
                orientation_text
            ),
            value(
                r'w:\s*([-+0-9.eE]+)',
                orientation_text,
                1.0
            )
        ])
    }


# ============================================================
# Pose stream update
# ============================================================

def parse_header_timestamp(block):

    sec_match = re.search(
        r'sec:\s*(\d+)',
        block
    )

    nsec_match = re.search(
        r'nsec:\s*(\d+)',
        block
    )

    if sec_match is None:
        return None

    sec = int(
        sec_match.group(1)
    )

    nsec = (
        int(nsec_match.group(1))
        if nsec_match
        else 0
    )

    return (
        sec
        + nsec * 1e-9
    )


# ============================================================
# Compose full camera pose
# ============================================================

def compose_camera_state(poses):

    required = {
        "hexacopter_with_ardupilot",
        "gimbal",
        "pitch_link",
    }

    if not required.issubset(
        poses.keys()
    ):
        missing = required - poses.keys()

        raise RuntimeError(
            "Missing pose(s): "
            + ", ".join(sorted(missing))
        )

    uav = poses[
        "hexacopter_with_ardupilot"
    ]

    gimbal = poses[
        "gimbal"
    ]

    pitch_link = poses[
        "pitch_link"
    ]

    # --------------------------------------------------------
    # Local rotations
    # --------------------------------------------------------

    R_world_uav = (
        geo.quaternion_to_rotation_matrix(
            uav["quaternion"]
        )
    )

    R_uav_gimbal = (
        geo.quaternion_to_rotation_matrix(
            gimbal["quaternion"]
        )
    )

    R_gimbal_pitch = (
        geo.quaternion_to_rotation_matrix(
            pitch_link["quaternion"]
        )
    )

    # Full hierarchy
    R_world_pitch = (
        R_world_uav
        @ R_uav_gimbal
        @ R_gimbal_pitch
    )

    # --------------------------------------------------------
    # Local positions
    # --------------------------------------------------------

    camera_uav = (
        gimbal["position"]
        + R_uav_gimbal
        @ pitch_link["position"]
    )

    camera_world = (
        uav["position"]
        + R_world_uav
        @ camera_uav
    )

    return (
        uav["position"],
        camera_world,
        R_world_pitch
    )


# ============================================================
# Gazebo state background thread
# ============================================================

def gazebo_state_worker():

    process = None

    try:

        process = subprocess.Popen(
            [
                "stdbuf",
                "-oL",
                "gz",
                "topic",
                "-e",
                "-t",
                POSE_TOPIC
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        current_pose_lines = []
        pose_depth = 0
        inside_pose = False

        current_header_lines = []
        header_depth = 0
        inside_header = False

        pending_poses = {}

        for line in process.stdout:

            if stop_event.is_set():
                break

            stripped = line.strip()

            # ------------------------------------------------
            # Start pose block
            # ------------------------------------------------

            if (
                not inside_pose
                and stripped == "pose {"
            ):

                inside_pose = True

                pose_depth = 1

                current_pose_lines = [
                    stripped
                ]

                continue

            # ------------------------------------------------
            # Read pose block
            # ------------------------------------------------

            if inside_pose:

                current_pose_lines.append(
                    stripped
                )

                pose_depth += (
                    stripped.count("{")
                    - stripped.count("}")
                )

                if pose_depth == 0:

                    inside_pose = False

                    block = "\n".join(
                        current_pose_lines
                    )

                    pose = parse_pose_block(
                        block
                    )

                    if pose is not None:

                        name = pose["name"]

                        if name in {
                            "hexacopter_with_ardupilot",
                            "gimbal",
                            "pitch_link",
                        }:

                            pending_poses[
                                name
                            ] = pose

                continue

            # ------------------------------------------------
            # Start header block
            # ------------------------------------------------

            if (
                not inside_header
                and stripped == "header {"
            ):

                inside_header = True

                header_depth = 1

                current_header_lines = [
                    stripped
                ]

                continue

            # ------------------------------------------------
            # Read header block
            # ------------------------------------------------

            if inside_header:

                current_header_lines.append(
                    stripped
                )

                header_depth += (
                    stripped.count("{")
                    - stripped.count("}")
                )

                if header_depth == 0:

                    inside_header = False

                    header_block = "\n".join(
                        current_header_lines
                    )

                    timestamp = (
                        parse_header_timestamp(
                            header_block
                        )
                    )

                    if (
                        timestamp is not None
                        and {
                            "hexacopter_with_ardupilot",
                            "gimbal",
                            "pitch_link",
                        }.issubset(
                            pending_poses.keys()
                        )
                    ):

                        try:

                            (
                                uav_world,
                                camera_world,
                                R_world_pitch
                            ) = compose_camera_state(
                                pending_poses
                            )

                            with state_lock:

                                state[
                                    "uav_world"
                                ] = uav_world

                                state[
                                    "camera_world"
                                ] = camera_world

                                state[
                                    "R_world_pitch"
                                ] = R_world_pitch

                                state[
                                    "pose_timestamp"
                                ] = time.monotonic()

                                state[
                                    "pose_error"
                                ] = None

                        except Exception as exc:

                            with state_lock:

                                state[
                                    "pose_error"
                                ] = str(exc)

                    pending_poses = {}

    except Exception as exc:

        with state_lock:

            state[
                "pose_error"
            ] = str(exc)

    finally:

        if process is not None:

            try:
                process.terminate()
                process.wait(
                    timeout=1.0
                )
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


# ============================================================
# MAVLink GPS background thread
# ============================================================

def gps_worker():

    master = None

    try:

        master = mavutil.mavlink_connection(
            MAVLINK_CONNECTION
        )

        master.wait_heartbeat(
            timeout=5
        )

        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            200000,
            0, 0, 0, 0, 0
        )

        while not stop_event.is_set():

            msg = master.recv_match(
                type="GLOBAL_POSITION_INT",
                blocking=True,
                timeout=0.25
            )

            if msg is None:
                continue

            lat = (
                msg.lat / 1e7
            )

            lon = (
                msg.lon / 1e7
            )

            with state_lock:

                state[
                    "gps_lat"
                ] = lat

                state[
                    "gps_lon"
                ] = lon

                state[
                    "gps_timestamp"
                ] = time.monotonic()

                state[
                    "gps_error"
                ] = None

    except Exception as exc:

        with state_lock:

            state[
                "gps_error"
            ] = str(exc)


# ============================================================
# Target detector
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

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        contour
    )

    if area < 10:
        return None

    M = cv2.moments(
        contour
    )

    if M["m00"] == 0:
        return None

    u = (
        M["m10"]
        / M["m00"]
    )

    v = (
        M["m01"]
        / M["m00"]
    )

    x, y, w, h = (
        cv2.boundingRect(
            contour
        )
    )

    return {
        "u": float(u),
        "v": float(v),
        "bbox": (
            x, y, w, h
        ),
        "area": float(area)
    }


# ============================================================
# Pixel -> NED ray
# ============================================================

def pixel_to_ned_ray(
    u,
    v,
    R_world_pitch
):

    ray_camera = (
        geo.pixel_to_camera_ray(
            u,
            v
        )
    )

    ray_gazebo_camera = (
        geo.R_GAZEBO_FROM_CAMERA
        @ ray_camera
    )

    ray_pitch = (
        geo.R_SENSOR
        @ ray_gazebo_camera
    )

    ray_world = (
        R_world_pitch
        @ ray_pitch
    )

    # Gazebo world -> NED
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

    return (
        ray_ned / norm
    )


# ============================================================
# Geolocation
# ============================================================

def estimate_target(
    u,
    v,
    uav_world,
    camera_world,
    R_world_pitch,
    gps_lat,
    gps_lon
):

    ray_ned = pixel_to_ned_ray(
        u,
        v,
        R_world_pitch
    )

    if ray_ned[2] <= 0:

        raise RuntimeError(
            "Ray does not point toward target plane."
        )

    # Camera relative to UAV in NED

    camera_delta = (
        camera_world
        - uav_world
    )

    camera_ned = np.array([
        camera_delta[1],
        camera_delta[0],
        -camera_delta[2]
    ])

    # Camera to target plane

    vertical_distance = (
        camera_world[2]
        - TARGET_Z
    )

    if vertical_distance <= 0:

        raise RuntimeError(
            "Target plane is not below camera."
        )

    # Ray-plane intersection

    scale = (
        vertical_distance
        / ray_ned[2]
    )

    target_ned = (
        camera_ned
        + scale * ray_ned
    )

    # NED -> GPS

    target_lat, target_lon = (
        geo.ned_to_gps(
            gps_lat,
            gps_lon,
            target_ned[0],
            target_ned[1]
        )
    )

    return {
        "target_ned": target_ned,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "ray_ned": ray_ned
    }


# ============================================================
# Drawing
# ============================================================

def draw_text(
    image,
    lines
):

    y = 24

    for line in lines:

        cv2.putText(
            image,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 21


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("REALTIME STATIC GEOLOCATION")
    print("=" * 70)
    print()
    print("State source:")
    print("  Gazebo dynamic_pose/info")
    print("  MAVLink GLOBAL_POSITION_INT")
    print()
    print("Detector:")
    print("  Red contour centroid")
    print()
    print("Target plane:")
    print(f"  Z = {TARGET_Z:.3f} m")
    print()
    print("Press Q to quit.")
    print()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_PIPELINE,
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera stream."
        )

    # --------------------------------------------------------
    # Start background workers
    # --------------------------------------------------------

    gazebo_thread = threading.Thread(
        target=gazebo_state_worker,
        daemon=True
    )

    gps_thread = threading.Thread(
        target=gps_worker,
        daemon=True
    )

    gazebo_thread.start()
    gps_thread.start()

    last_terminal_print = 0.0

    try:

        while not stop_event.is_set():

            # =================================================
            # Camera
            # =================================================

            ret, frame = cap.read()

            if not ret:
                continue

            display = frame.copy()

            # =================================================
            # Detection
            # =================================================

            detection = detect_target(
                frame
            )

            # =================================================
            # Read latest state
            # =================================================

            with state_lock:

                uav_world = state[
                    "uav_world"
                ]

                camera_world = state[
                    "camera_world"
                ]

                R_world_pitch = state[
                    "R_world_pitch"
                ]

                pose_timestamp = state[
                    "pose_timestamp"
                ]

                gps_lat = state[
                    "gps_lat"
                ]

                gps_lon = state[
                    "gps_lon"
                ]

                gps_timestamp = state[
                    "gps_timestamp"
                ]

                pose_error = state[
                    "pose_error"
                ]

                gps_error = state[
                    "gps_error"
                ]

            # =================================================
            # State ages
            # =================================================

            now = time.monotonic()

            if pose_timestamp is None:

                pose_age = float("inf")

            else:

                pose_age = (
                    now
                    - pose_timestamp
                )

            if gps_timestamp is None:

                gps_age = float("inf")

            else:

                gps_age = (
                    now
                    - gps_timestamp
                )

            # =================================================
            # Detection visualization
            # =================================================

            estimate = None

            if detection is not None:

                x, y, w, h = (
                    detection["bbox"]
                )

                u = detection["u"]
                v = detection["v"]

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.drawMarker(
                    display,
                    (
                        int(round(u)),
                        int(round(v))
                    ),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    12,
                    2
                )

                cv2.putText(
                    display,
                    "Centroid",
                    (
                        int(round(u)) + 8,
                        int(round(v)) - 8
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

                # =================================================
                # Geolocation
                # =================================================

                if (
                    uav_world is not None
                    and camera_world is not None
                    and R_world_pitch is not None
                    and gps_lat is not None
                    and gps_lon is not None
                ):

                    try:

                        estimate = (
                            estimate_target(
                                u,
                                v,
                                uav_world,
                                camera_world,
                                R_world_pitch,
                                gps_lat,
                                gps_lon
                            )
                        )

                    except Exception as exc:

                        pose_error = (
                            f"Estimate: {exc}"
                        )

            # =================================================
            # GUI text
            # =================================================

            if detection is None:

                lines = [
                    "TARGET NOT DETECTED"
                ]

            elif estimate is None:

                lines = [
                    "WAITING FOR STATE"
                ]

            else:

                target_ned = (
                    estimate["target_ned"]
                )

                lines = [
                    "GEOLOCATION ACTIVE",
                    (
                        f"Centroid: "
                        f"({detection['u']:.2f}, "
                        f"{detection['v']:.2f}) px"
                    ),
                    (
                        f"North: "
                        f"{target_ned[0]:+.3f} m"
                    ),
                    (
                        f"East : "
                        f"{target_ned[1]:+.3f} m"
                    ),
                    (
                        f"Down : "
                        f"{target_ned[2]:+.3f} m"
                    ),
                    (
                        f"Target lat: "
                        f"{estimate['target_lat']:.8f}"
                    ),
                    (
                        f"Target lon: "
                        f"{estimate['target_lon']:.8f}"
                    )
                ]

            # Always display state ages

            lines.append(
                f"Pose age: {pose_age:.3f} s"
            )

            lines.append(
                f"GPS age : {gps_age:.3f} s"
            )

            if pose_error is not None:

                lines.append(
                    f"Pose error: {pose_error}"
                )

            if gps_error is not None:

                lines.append(
                    f"GPS error: {gps_error}"
                )

            draw_text(
                display,
                lines
            )

            # =================================================
            # Terminal output
            # =================================================

            if (
                estimate is not None
                and now - last_terminal_print
                >= 1.0
            ):

                n = (
                    estimate["target_ned"][0]
                )

                e = (
                    estimate["target_ned"][1]
                )

                print(
                    f"Target estimate | "
                    f"N={n:+.3f} m | "
                    f"E={e:+.3f} m | "
                    f"Pose age={pose_age:.3f} s | "
                    f"GPS age={gps_age:.3f} s"
                )

                last_terminal_print = now

            # =================================================
            # Display
            # =================================================

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

    finally:

        stop_event.set()

        gazebo_thread.join(
            timeout=1.0
        )

        gps_thread.join(
            timeout=1.0
        )

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
