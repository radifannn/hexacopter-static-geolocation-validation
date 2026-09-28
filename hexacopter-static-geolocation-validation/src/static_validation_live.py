import csv
import math
import re
import subprocess
import threading
import time

import cv2
import numpy as np
from pymavlink import mavutil

import geolocation_geometry as geo


# ============================================================
# CONFIGURATION
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

TARGET_GPS_TOPIC = (
    "/target/gps"
)

MAVLINK_CONNECTION = (
    "tcp:127.0.0.1:5762"
)

TARGET_Z = 0.01

N_SAMPLES = 100

SAMPLE_INTERVAL = 0.10

OUTPUT_CSV = (
    "/home/rifqi-radifan/gz_ws/geolocation/"
    "static_validation_live.csv"
)


# ============================================================
# DETECTOR CONFIGURATION
# ============================================================

LOWER_RED_1 = np.array([
    0,
    100,
    80
])

UPPER_RED_1 = np.array([
    10,
    255,
    255
])

LOWER_RED_2 = np.array([
    170,
    100,
    80
])

UPPER_RED_2 = np.array([
    180,
    255,
    255
])

KERNEL = np.ones(
    (3, 3),
    np.uint8
)


# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.Lock()

stop_event = threading.Event()

state = {

    # --------------------------------------------------------
    # Gazebo / camera geometry
    # --------------------------------------------------------

    "uav_world": None,
    "camera_world": None,
    "R_world_pitch": None,

    # Ground truth target world position
    "target_world": None,

    # Pose timing
    "pose_timestamp": None,

    # --------------------------------------------------------
    # UAV GPS
    # --------------------------------------------------------

    "gps_lat": None,
    "gps_lon": None,
    "gps_timestamp": None,

    # --------------------------------------------------------
    # Target GPS ground truth
    # --------------------------------------------------------

    "target_gps_lat": None,
    "target_gps_lon": None,
    "target_gps_alt": None,
    "target_gps_timestamp": None,

    # --------------------------------------------------------
    # Worker diagnostics
    # --------------------------------------------------------

    "pose_error": None,
    "gps_error": None,
    "target_gps_error": None
}


# ============================================================
# GENERIC PARSER
# ============================================================

def parse_number(
    pattern,
    text,
    default=None
):

    match = re.search(
        pattern,
        text
    )

    if match is None:
        return default

    return float(
        match.group(1)
    )


# ============================================================
# GAZEBO POSE PARSER
# ============================================================

def parse_pose_block(block):

    name_match = re.search(
        r'name:\s*"([^"]+)"',
        block
    )

    if name_match is None:
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
        or
        orientation_match is None
    ):

        return None

    position_text = (
        position_match.group(1)
    )

    orientation_text = (
        orientation_match.group(1)
    )

    return {

        "name":
            name,

        "position":
            np.array([

                parse_number(
                    r'x:\s*([-+0-9.eE]+)',
                    position_text,
                    0.0
                ),

                parse_number(
                    r'y:\s*([-+0-9.eE]+)',
                    position_text,
                    0.0
                ),

                parse_number(
                    r'z:\s*([-+0-9.eE]+)',
                    position_text,
                    0.0
                )
            ]),

        "quaternion":
            np.array([

                parse_number(
                    r'x:\s*([-+0-9.eE]+)',
                    orientation_text,
                    0.0
                ),

                parse_number(
                    r'y:\s*([-+0-9.eE]+)',
                    orientation_text,
                    0.0
                ),

                parse_number(
                    r'z:\s*([-+0-9.eE]+)',
                    orientation_text,
                    0.0
                ),

                parse_number(
                    r'w:\s*([-+0-9.eE]+)',
                    orientation_text,
                    1.0
                )
            ])
    }


# ============================================================
# HEADER TIMESTAMP
# ============================================================

def parse_header_timestamp(
    block
):

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
        int(
            nsec_match.group(1)
        )
        if nsec_match
        else 0
    )

    return (
        sec
        +
        nsec * 1e-9
    )


# ============================================================
# CAMERA POSE COMPOSITION
# ============================================================

def compose_camera_state(
    poses
):

    required = {
        "hexacopter_with_ardupilot",
        "gimbal",
        "pitch_link"
    }

    if not required.issubset(
        poses.keys()
    ):

        return None

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
    # Rotation hierarchy
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

    R_world_pitch = (
        R_world_uav
        @ R_uav_gimbal
        @ R_gimbal_pitch
    )

    # --------------------------------------------------------
    # Position hierarchy
    # --------------------------------------------------------

    camera_uav = (
        gimbal["position"]
        +
        R_uav_gimbal
        @ pitch_link["position"]
    )

    camera_world = (
        uav["position"]
        +
        R_world_uav
        @ camera_uav
    )

    return (
        uav["position"],
        camera_world,
        R_world_pitch
    )


# ============================================================
# GAZEBO PERSISTENT POSE STREAM
# ============================================================

def gazebo_pose_worker():

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

        pose_lines = []
        pose_depth = 0
        inside_pose = False

        header_lines = []
        header_depth = 0
        inside_header = False

        pending_poses = {}

        required = {
            "hexacopter_with_ardupilot",
            "gimbal",
            "pitch_link",
            "moving_target"
        }

        for line in process.stdout:

            if stop_event.is_set():
                break

            stripped = line.strip()

            # ------------------------------------------------
            # POSE
            # ------------------------------------------------

            if (
                not inside_pose
                and
                stripped == "pose {"
            ):

                inside_pose = True

                pose_depth = 1

                pose_lines = [
                    stripped
                ]

                continue

            if inside_pose:

                pose_lines.append(
                    stripped
                )

                pose_depth += (
                    stripped.count("{")
                    -
                    stripped.count("}")
                )

                if pose_depth == 0:

                    inside_pose = False

                    block = "\n".join(
                        pose_lines
                    )

                    pose = (
                        parse_pose_block(
                            block
                        )
                    )

                    if pose is not None:

                        if (
                            pose["name"]
                            in
                            required
                        ):

                            pending_poses[
                                pose["name"]
                            ] = pose

                continue

            # ------------------------------------------------
            # HEADER
            # ------------------------------------------------

            if (
                not inside_header
                and
                stripped == "header {"
            ):

                inside_header = True

                header_depth = 1

                header_lines = [
                    stripped
                ]

                continue

            if inside_header:

                header_lines.append(
                    stripped
                )

                header_depth += (
                    stripped.count("{")
                    -
                    stripped.count("}")
                )

                if header_depth == 0:

                    inside_header = False

                    timestamp = (
                        parse_header_timestamp(
                            "\n".join(
                                header_lines
                            )
                        )
                    )

                    if (
                        timestamp is not None
                        and
                        required.issubset(
                            pending_poses.keys()
                        )
                    ):

                        try:

                            result = (
                                compose_camera_state(
                                    pending_poses
                                )
                            )

                            if result is None:
                                pending_poses = {}
                                continue

                            (
                                uav_world,
                                camera_world,
                                R_world_pitch
                            ) = result

                            target_world = (
                                pending_poses[
                                    "moving_target"
                                ]["position"]
                                .copy()
                            )

                            with state_lock:

                                state[
                                    "uav_world"
                                ] = (
                                    uav_world
                                )

                                state[
                                    "camera_world"
                                ] = (
                                    camera_world
                                )

                                state[
                                    "R_world_pitch"
                                ] = (
                                    R_world_pitch
                                )

                                state[
                                    "target_world"
                                ] = (
                                    target_world
                                )

                                state[
                                    "pose_timestamp"
                                ] = (
                                    time.monotonic()
                                )

                                state[
                                    "pose_error"
                                ] = None

                        except Exception as exc:

                            with state_lock:

                                state[
                                    "pose_error"
                                ] = str(
                                    exc
                                )

                    pending_poses = {}

    except Exception as exc:

        with state_lock:

            state[
                "pose_error"
            ] = str(
                exc
            )

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
# UAV GPS WORKER
# ============================================================

def gps_worker():

    try:

        master = (
            mavutil.mavlink_connection(
                MAVLINK_CONNECTION
            )
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
            0,
            0,
            0,
            0,
            0
        )

        while not stop_event.is_set():

            msg = (
                master.recv_match(
                    type="GLOBAL_POSITION_INT",
                    blocking=True,
                    timeout=0.25
                )
            )

            if msg is None:
                continue

            with state_lock:

                state[
                    "gps_lat"
                ] = (
                    msg.lat / 1e7
                )

                state[
                    "gps_lon"
                ] = (
                    msg.lon / 1e7
                )

                state[
                    "gps_timestamp"
                ] = (
                    time.monotonic()
                )

                state[
                    "gps_error"
                ] = None

    except Exception as exc:

        with state_lock:

            state[
                "gps_error"
            ] = str(
                exc
            )


# ============================================================
# TARGET GPS GROUND TRUTH WORKER
# ============================================================

def target_gps_worker():

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
                TARGET_GPS_TOPIC
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        current_lat = None
        current_lon = None
        current_alt = None

        for line in process.stdout:

            if stop_event.is_set():
                break

            stripped = line.strip()

            # ------------------------------------------------
            # Latitude
            # ------------------------------------------------

            lat_match = re.search(
                r'latitude_deg:\s*([-+0-9.eE]+)',
                stripped
            )

            if lat_match:

                current_lat = float(
                    lat_match.group(1)
                )

            # ------------------------------------------------
            # Longitude
            # ------------------------------------------------

            lon_match = re.search(
                r'longitude_deg:\s*([-+0-9.eE]+)',
                stripped
            )

            if lon_match:

                current_lon = float(
                    lon_match.group(1)
                )

            # ------------------------------------------------
            # Altitude
            # ------------------------------------------------

            alt_match = re.search(
                r'altitude:\s*([-+0-9.eE]+)',
                stripped
            )

            if alt_match:

                current_alt = float(
                    alt_match.group(1)
                )

            # ------------------------------------------------
            # Complete target GPS sample
            # ------------------------------------------------

            if (
                current_lat is not None
                and
                current_lon is not None
            ):

                with state_lock:

                    state[
                        "target_gps_lat"
                    ] = (
                        current_lat
                    )

                    state[
                        "target_gps_lon"
                    ] = (
                        current_lon
                    )

                    state[
                        "target_gps_alt"
                    ] = (
                        current_alt
                    )

                    state[
                        "target_gps_timestamp"
                    ] = (
                        time.monotonic()
                    )

                    state[
                        "target_gps_error"
                    ] = None

                current_lat = None
                current_lon = None
                current_alt = None

    except Exception as exc:

        with state_lock:

            state[
                "target_gps_error"
            ] = str(
                exc
            )

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
# TARGET DETECTOR
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

    moments = cv2.moments(
        contour
    )

    if moments["m00"] == 0:
        return None

    u = (
        moments["m10"]
        /
        moments["m00"]
    )

    v = (
        moments["m01"]
        /
        moments["m00"]
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
            x,
            y,
            w,
            h
        ),
        "area": float(area)
    }


# ============================================================
# PIXEL -> NED RAY
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
        ray_ned
        /
        norm
    )


# ============================================================
# GEOMETRIC TARGET ESTIMATION
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

    ray_ned = (
        pixel_to_ned_ray(
            u,
            v,
            R_world_pitch
        )
    )

    if ray_ned[2] <= 0:

        raise RuntimeError(
            "Ray does not intersect target plane."
        )

    # --------------------------------------------------------
    # Camera position relative to UAV
    # --------------------------------------------------------

    camera_delta = (
        camera_world
        -
        uav_world
    )

    camera_ned = np.array([

        camera_delta[1],

        camera_delta[0],

        -camera_delta[2]
    ])

    # --------------------------------------------------------
    # Ground plane
    # --------------------------------------------------------

    vertical_distance = (
        camera_world[2]
        -
        TARGET_Z
    )

    if vertical_distance <= 0:

        raise RuntimeError(
            "Target plane is not below camera."
        )

    scale = (
        vertical_distance
        /
        ray_ned[2]
    )

    target_ned = (
        camera_ned
        +
        scale * ray_ned
    )

    # --------------------------------------------------------
    # NED -> GPS
    # --------------------------------------------------------

    target_lat, target_lon = (
        geo.ned_to_gps(
            gps_lat,
            gps_lon,
            target_ned[0],
            target_ned[1]
        )
    )

    return {

        "target_ned":
            target_ned,

        "target_lat":
            target_lat,

        "target_lon":
            target_lon,

        "ray_ned":
            ray_ned
    }


# ============================================================
# GPS ERROR
# ============================================================

def gps_difference_meters(
    estimated_lat,
    estimated_lon,
    true_lat,
    true_lon
):

    earth_radius = 6378137.0

    delta_lat = math.radians(
        estimated_lat
        -
        true_lat
    )

    delta_lon = math.radians(
        estimated_lon
        -
        true_lon
    )

    mean_lat = math.radians(
        (
            estimated_lat
            +
            true_lat
        )
        /
        2.0
    )

    north = (
        earth_radius
        *
        delta_lat
    )

    east = (
        earth_radius
        *
        math.cos(mean_lat)
        *
        delta_lon
    )

    horizontal = math.hypot(
        north,
        east
    )

    return (
        north,
        east,
        horizontal
    )


# ============================================================
# SAVE + REPORT
# ============================================================

def save_and_report(
    rows
):

    if not rows:

        print(
            "No samples collected."
        )

        return

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    gps_n = np.array([
        row[
            "gps_error_north"
        ]
        for row in rows
    ])

    gps_e = np.array([
        row[
            "gps_error_east"
        ]
        for row in rows
    ])

    gps_h = np.array([
        row[
            "gps_horizontal_error"
        ]
        for row in rows
    ])

    rel_n = np.array([
        row[
            "error_north"
        ]
        for row in rows
    ])

    rel_e = np.array([
        row[
            "error_east"
        ]
        for row in rows
    ])

    rel_h = np.array([
        row[
            "horizontal_error"
        ]
        for row in rows
    ])

    u_values = np.array([
        row[
            "u"
        ]
        for row in rows
    ])

    v_values = np.array([
        row[
            "v"
        ]
        for row in rows
    ])

    pose_age = np.array([
        row[
            "pose_age"
        ]
        for row in rows
    ])

    uav_gps_age = np.array([
        row[
            "uav_gps_age"
        ]
        for row in rows
    ])

    target_gps_age = np.array([
        row[
            "target_gps_age"
        ]
        for row in rows
    ])

    est_lat = np.array([
        row[
            "estimated_target_lat"
        ]
        for row in rows
    ])

    est_lon = np.array([
        row[
            "estimated_target_lon"
        ]
        for row in rows
    ])

    true_lat = np.array([
        row[
            "true_target_lat"
        ]
        for row in rows
    ])

    true_lon = np.array([
        row[
            "true_target_lon"
        ]
        for row in rows
    ])

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STATIC VALIDATION LIVE — FINAL RESULTS")
    print("=" * 70)

    print()
    print(
        f"Valid samples = "
        f"{len(rows)}"
    )

    print()
    print(
        "Estimated target GPS:"
    )

    print(
        f"  Latitude  = "
        f"{np.mean(est_lat):.10f}"
    )

    print(
        f"  Longitude = "
        f"{np.mean(est_lon):.10f}"
    )

    print()
    print(
        "True target GPS:"
    )

    print(
        f"  Latitude  = "
        f"{np.mean(true_lat):.10f}"
    )

    print(
        f"  Longitude = "
        f"{np.mean(true_lon):.10f}"
    )

    print()
    print(
        "Direct GPS position error:"
    )

    print(
        f"  North bias = "
        f"{np.mean(gps_n):+.4f} m"
    )

    print(
        f"  East bias  = "
        f"{np.mean(gps_e):+.4f} m"
    )

    print(
        f"  North RMSE = "
        f"{math.sqrt(np.mean(gps_n ** 2)):.4f} m"
    )

    print(
        f"  East RMSE  = "
        f"{math.sqrt(np.mean(gps_e ** 2)):.4f} m"
    )

    print(
        f"  Horizontal RMSE = "
        f"{math.sqrt(np.mean(gps_h ** 2)):.4f} m"
    )

    print(
        f"  Horizontal MAE = "
        f"{np.mean(gps_h):.4f} m"
    )

    print(
        f"  Minimum error = "
        f"{np.min(gps_h):.4f} m"
    )

    print(
        f"  Maximum error = "
        f"{np.max(gps_h):.4f} m"
    )

    print()
    print(
        "Relative NED diagnostic:"
    )

    print(
        f"  North bias = "
        f"{np.mean(rel_n):+.4f} m"
    )

    print(
        f"  East bias = "
        f"{np.mean(rel_e):+.4f} m"
    )

    print(
        f"  Horizontal RMSE = "
        f"{math.sqrt(np.mean(rel_h ** 2)):.4f} m"
    )

    print()
    print(
        "Pixel measurement:"
    )

    print(
        f"  Mean u = "
        f"{np.mean(u_values):.4f} px"
    )

    print(
        f"  Mean v = "
        f"{np.mean(v_values):.4f} px"
    )

    print(
        f"  Std u = "
        f"{np.std(u_values):.4f} px"
    )

    print(
        f"  Std v = "
        f"{np.std(v_values):.4f} px"
    )

    print()
    print(
        "State freshness:"
    )

    print(
        f"  Pose age mean = "
        f"{np.mean(pose_age):.4f} s"
    )

    print(
        f"  Pose age max = "
        f"{np.max(pose_age):.4f} s"
    )

    print(
        f"  UAV GPS age mean = "
        f"{np.mean(uav_gps_age):.4f} s"
    )

    print(
        f"  UAV GPS age max = "
        f"{np.max(uav_gps_age):.4f} s"
    )

    print(
        f"  Target GPS age mean = "
        f"{np.mean(target_gps_age):.4f} s"
    )

    print(
        f"  Target GPS age max = "
        f"{np.max(target_gps_age):.4f} s"
    )

    print()
    print(
        "CSV:"
    )

    print(
        f"  {OUTPUT_CSV}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STATIC VALIDATION LIVE")
    print("=" * 70)
    print()
    print(
        "One camera stream:"
    )
    print(
        "  Live geolocation + 100-sample validation"
    )
    print()
    print(
        "Controls:"
    )
    print(
        "  R = start/restart 100-sample validation"
    )
    print(
        "  Q = quit"
    )
    print()

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    stop_event.clear()

    # --------------------------------------------------------
    # Open camera ONCE
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_PIPELINE,
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera stream. "
            "Make sure camera streaming is enabled."
        )

    # --------------------------------------------------------
    # Start workers
    # --------------------------------------------------------

    pose_thread = threading.Thread(
        target=gazebo_pose_worker,
        daemon=True
    )

    gps_thread = threading.Thread(
        target=gps_worker,
        daemon=True
    )

    target_gps_thread = threading.Thread(
        target=target_gps_worker,
        daemon=True
    )

    pose_thread.start()

    gps_thread.start()

    target_gps_thread.start()

    # --------------------------------------------------------
    # Sampling state
    # --------------------------------------------------------

    sampling = False

    rows = []

    last_sample_time = 0.0

    try:

        while not stop_event.is_set():

            # =================================================
            # Read camera
            # =================================================

            ret, frame = cap.read()

            if not ret:

                continue

            display = frame.copy()

            detection = detect_target(
                frame
            )

            estimate = None

            pose_age = None
            uav_gps_age = None
            target_gps_age = None

            # =================================================
            # Snapshot shared state
            # =================================================

            with state_lock:

                uav_world = (
                    state[
                        "uav_world"
                    ].copy()
                    if
                    state[
                        "uav_world"
                    ] is not None
                    else None
                )

                camera_world = (
                    state[
                        "camera_world"
                    ].copy()
                    if
                    state[
                        "camera_world"
                    ] is not None
                    else None
                )

                R_world_pitch = (
                    state[
                        "R_world_pitch"
                    ].copy()
                    if
                    state[
                        "R_world_pitch"
                    ] is not None
                    else None
                )

                target_world = (
                    state[
                        "target_world"
                    ].copy()
                    if
                    state[
                        "target_world"
                    ] is not None
                    else None
                )

                gps_lat = (
                    state[
                        "gps_lat"
                    ]
                )

                gps_lon = (
                    state[
                        "gps_lon"
                    ]
                )

                pose_timestamp = (
                    state[
                        "pose_timestamp"
                    ]
                )

                gps_timestamp = (
                    state[
                        "gps_timestamp"
                    ]
                )

                true_target_lat = (
                    state[
                        "target_gps_lat"
                    ]
                )

                true_target_lon = (
                    state[
                        "target_gps_lon"
                    ]
                )

                true_target_alt = (
                    state[
                        "target_gps_alt"
                    ]
                )

                target_gps_timestamp = (
                    state[
                        "target_gps_timestamp"
                    ]
                )

            # =================================================
            # State age
            # =================================================

            now = time.monotonic()

            if pose_timestamp is not None:

                pose_age = (
                    now
                    -
                    pose_timestamp
                )

            if gps_timestamp is not None:

                uav_gps_age = (
                    now
                    -
                    gps_timestamp
                )

            if target_gps_timestamp is not None:

                target_gps_age = (
                    now
                    -
                    target_gps_timestamp
                )

            # =================================================
            # Detector visualization
            # =================================================

            if detection is not None:

                u = detection["u"]
                v = detection["v"]

                x, y, w, h = (
                    detection["bbox"]
                )

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
                        int(
                            round(u)
                        ),
                        int(
                            round(v)
                        )
                    ),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    12,
                    2
                )

                cv2.putText(
                    display,
                    (
                        f"Centroid: "
                        f"({u:.2f}, {v:.2f})"
                    ),
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                # =================================================
                # Live geolocation
                # =================================================

                if (
                    uav_world is not None
                    and
                    camera_world is not None
                    and
                    R_world_pitch is not None
                    and
                    gps_lat is not None
                    and
                    gps_lon is not None
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

                        target_ned = (
                            estimate[
                                "target_ned"
                            ]
                        )

                        target_lat = (
                            estimate[
                                "target_lat"
                            ]
                        )

                        target_lon = (
                            estimate[
                                "target_lon"
                            ]
                        )

                        # ----------------------------------------
                        # Live text
                        # ----------------------------------------

                        cv2.putText(
                            display,
                            "GEOLOCATION ACTIVE",
                            (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        cv2.putText(
                            display,
                            (
                                f"N: "
                                f"{target_ned[0]:+.3f} m"
                            ),
                            (10, 78),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        cv2.putText(
                            display,
                            (
                                f"E: "
                                f"{target_ned[1]:+.3f} m"
                            ),
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        cv2.putText(
                            display,
                            (
                                f"Lat: "
                                f"{target_lat:.8f}"
                            ),
                            (10, 122),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        cv2.putText(
                            display,
                            (
                                f"Lon: "
                                f"{target_lon:.8f}"
                            ),
                            (10, 144),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        if pose_age is not None:

                            cv2.putText(
                                display,
                                (
                                    f"Pose age: "
                                    f"{pose_age:.3f} s"
                                ),
                                (10, 166),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.48,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )

                        if uav_gps_age is not None:

                            cv2.putText(
                                display,
                                (
                                    f"UAV GPS age: "
                                    f"{uav_gps_age:.3f} s"
                                ),
                                (10, 187),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.48,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )

                        if target_gps_age is not None:

                            cv2.putText(
                                display,
                                (
                                    f"Target GPS age: "
                                    f"{target_gps_age:.3f} s"
                                ),
                                (10, 208),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.48,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )

                        # ----------------------------------------
                        # Validation error on live window
                        # ----------------------------------------

                        if (
                            true_target_lat is not None
                            and
                            true_target_lon is not None
                        ):

                            (
                                _,
                                _,
                                live_gps_error
                            ) = (
                                gps_difference_meters(
                                    target_lat,
                                    target_lon,
                                    true_target_lat,
                                    true_target_lon
                                )
                            )

                            cv2.putText(
                                display,
                                (
                                    f"GPS error: "
                                    f"{live_gps_error:.3f} m"
                                ),
                                (10, 230),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.50,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )

                    except Exception as exc:

                        cv2.putText(
                            display,
                            (
                                "Estimation error: "
                                +
                                str(exc)[:70]
                            ),
                            (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

            else:

                cv2.putText(
                    display,
                    "TARGET NOT DETECTED",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # =================================================
            # Sampling display
            # =================================================

            if sampling:

                cv2.putText(
                    display,
                    (
                        f"SAMPLING: "
                        f"{len(rows)}/{N_SAMPLES}"
                    ),
                    (410, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                if (
                    detection is not None
                    and
                    estimate is not None
                    and
                    true_target_lat is not None
                    and
                    true_target_lon is not None
                    and
                    target_world is not None
                    and
                    uav_world is not None
                    and
                    pose_timestamp is not None
                    and
                    gps_timestamp is not None
                    and
                    target_gps_timestamp is not None
                    and
                    now - last_sample_time
                    >= SAMPLE_INTERVAL
                ):

                    estimated_ned = (
                        estimate[
                            "target_ned"
                        ]
                    )

                    target_delta = (
                        target_world
                        -
                        uav_world
                    )

                    true_ned = np.array([

                        target_delta[1],

                        target_delta[0],

                        -target_delta[2]
                    ])

                    # -----------------------------------------
                    # Relative NED error
                    # -----------------------------------------

                    error_n = (
                        estimated_ned[0]
                        -
                        true_ned[0]
                    )

                    error_e = (
                        estimated_ned[1]
                        -
                        true_ned[1]
                    )

                    error_d = (
                        estimated_ned[2]
                        -
                        true_ned[2]
                    )

                    horizontal_error = math.hypot(
                        error_n,
                        error_e
                    )

                    # -----------------------------------------
                    # Direct GPS error
                    # -----------------------------------------

                    (
                        gps_error_n,
                        gps_error_e,
                        gps_horizontal_error
                    ) = (
                        gps_difference_meters(
                            estimate["target_lat"],
                            estimate["target_lon"],
                            true_target_lat,
                            true_target_lon
                        )
                    )

                    # -----------------------------------------
                    # Store
                    # -----------------------------------------

                    rows.append({

                        "sample":
                            len(rows) + 1,

                        "u":
                            u,

                        "v":
                            v,

                        "true_north":
                            true_ned[0],

                        "true_east":
                            true_ned[1],

                        "true_down":
                            true_ned[2],

                        "estimated_north":
                            estimated_ned[0],

                        "estimated_east":
                            estimated_ned[1],

                        "estimated_down":
                            estimated_ned[2],

                        "error_north":
                            error_n,

                        "error_east":
                            error_e,

                        "error_down":
                            error_d,

                        "horizontal_error":
                            horizontal_error,

                        "uav_lat":
                            gps_lat,

                        "uav_lon":
                            gps_lon,

                        "true_target_lat":
                            true_target_lat,

                        "true_target_lon":
                            true_target_lon,

                        "true_target_alt":
                            true_target_alt,

                        "estimated_target_lat":
                            estimate["target_lat"],

                        "estimated_target_lon":
                            estimate["target_lon"],

                        "gps_error_north":
                            gps_error_n,

                        "gps_error_east":
                            gps_error_e,

                        "gps_horizontal_error":
                            gps_horizontal_error,

                        "pose_age":
                            pose_age,

                        "uav_gps_age":
                            uav_gps_age,

                        "target_gps_age":
                            target_gps_age
                    })

                    last_sample_time = now

                    print(
                        f"  [{len(rows):03d}/{N_SAMPLES}] "
                        f"pixel=({u:.2f},{v:.2f}) "
                        f"est_gps=("
                        f"{estimate['target_lat']:.8f},"
                        f"{estimate['target_lon']:.8f}) "
                        f"gps_error="
                        f"{gps_horizontal_error:.4f} m"
                    )

                    # -----------------------------------------
                    # Finished
                    # -----------------------------------------

                    if len(rows) >= N_SAMPLES:

                        sampling = False

                        print()
                        print(
                            "100 samples collected."
                        )

                        save_and_report(
                            rows
                        )

            else:

                cv2.putText(
                    display,
                    (
                        "Press R to start "
                        "100-sample validation"
                    ),
                    (330, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # =================================================
            # Show
            # =================================================

            cv2.imshow(
                "Static Validation Live",
                display
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            if key == ord("q"):

                break

            # ------------------------------------------------
            # Start / restart sampling
            # ------------------------------------------------

            if key == ord("r"):

                rows = []

                sampling = True

                last_sample_time = 0.0

                print()
                print(
                    "Starting a new "
                    "100-sample validation run..."
                )

    finally:

        stop_event.set()

        cap.release()

        cv2.destroyAllWindows()

        pose_thread.join(
            timeout=1.0
        )

        gps_thread.join(
            timeout=1.0
        )

        target_gps_thread.join(
            timeout=1.0


        )


if __name__ == "__main__":

    main()
