import re
import subprocess
import time

import numpy as np

import geolocation_geometry as geo


TOPIC = "/world/hexacopter_runway/dynamic_pose/info"

TARGET_NAMES = {
    "hexacopter_with_ardupilot",
    "gimbal",
    "pitch_link",
}


# ============================================================
# Parse one complete pose block
# ============================================================

def parse_pose_block(block):

    name_match = re.search(
        r'name:\s*"([^"]+)"',
        block
    )

    if not name_match:
        return None

    name = name_match.group(1)

    if name not in TARGET_NAMES:
        return None

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

    if not position_match or not orientation_match:
        return None

    position_text = position_match.group(1)
    orientation_text = orientation_match.group(1)

    def get_value(pattern, source, default=0.0):

        match = re.search(
            pattern,
            source
        )

        if match:
            return float(match.group(1))

        return default

    position = np.array([
        get_value(
            r'x:\s*([-+0-9.eE]+)',
            position_text
        ),
        get_value(
            r'y:\s*([-+0-9.eE]+)',
            position_text
        ),
        get_value(
            r'z:\s*([-+0-9.eE]+)',
            position_text
        )
    ])

    quaternion = np.array([
        get_value(
            r'x:\s*([-+0-9.eE]+)',
            orientation_text
        ),
        get_value(
            r'y:\s*([-+0-9.eE]+)',
            orientation_text
        ),
        get_value(
            r'z:\s*([-+0-9.eE]+)',
            orientation_text
        ),
        get_value(
            r'w:\s*([-+0-9.eE]+)',
            orientation_text,
            1.0
        )
    ])

    return {
        "name": name,
        "position": position,
        "quaternion": quaternion
    }


# ============================================================
# Parse complete header block
# ============================================================

def parse_header_block(block):

    sec_match = re.search(
        r'sec:\s*(\d+)',
        block
    )

    nsec_match = re.search(
        r'nsec:\s*(\d+)',
        block
    )

    if not sec_match:
        return None

    sec = int(
        sec_match.group(1)
    )

    nsec = int(
        nsec_match.group(1)
    ) if nsec_match else 0

    return sec + nsec * 1e-9


# ============================================================
# Compose camera pose
# ============================================================

def compose_camera_pose(poses):

    uav = poses[
        "hexacopter_with_ardupilot"
    ]

    gimbal = poses[
        "gimbal"
    ]

    pitch_link = poses[
        "pitch_link"
    ]

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

    # --------------------------------------------------------
    # Rotation hierarchy
    # --------------------------------------------------------

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
        + R_uav_gimbal
        @ pitch_link["position"]
    )

    camera_world = (
        uav["position"]
        + R_world_uav
        @ camera_uav
    )

    return (
        uav,
        gimbal,
        pitch_link,
        camera_world,
        R_world_pitch
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GAZEBO DYNAMIC POSE STREAM TEST")
    print("=" * 70)
    print()
    print(f"Topic: {TOPIC}")
    print()
    print("Press Ctrl+C to stop.")
    print()

    process = subprocess.Popen(
        [
            "stdbuf",
            "-oL",
            "gz",
            "topic",
            "-e",
            "-t",
            TOPIC
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    current_pose_lines = []
    current_pose_depth = 0
    inside_pose = False

    current_header_lines = []
    current_header_depth = 0
    inside_header = False

    pending_poses = {}

    last_timestamp = None
    last_receive_time = None
    update_count = 0

    try:

        for line in process.stdout:

            stripped = line.strip()

            # =================================================
            # Pose block
            # =================================================

            if not inside_pose and stripped == "pose {":

                inside_pose = True
                current_pose_depth = 1
                current_pose_lines = [
                    stripped
                ]
                continue

            if inside_pose:

                current_pose_lines.append(
                    stripped
                )

                current_pose_depth += (
                    stripped.count("{")
                    - stripped.count("}")
                )

                if current_pose_depth == 0:

                    inside_pose = False

                    block = "\n".join(
                        current_pose_lines
                    )

                    pose = parse_pose_block(
                        block
                    )

                    if pose is not None:

                        pending_poses[
                            pose["name"]
                        ] = pose

                continue

            # =================================================
            # Header block
            # =================================================

            if not inside_header and stripped == "header {":

                inside_header = True
                current_header_depth = 1
                current_header_lines = [
                    stripped
                ]
                continue

            if inside_header:

                current_header_lines.append(
                    stripped
                )

                current_header_depth += (
                    stripped.count("{")
                    - stripped.count("}")
                )

                if current_header_depth == 0:

                    inside_header = False

                    block = "\n".join(
                        current_header_lines
                    )

                    timestamp = parse_header_block(
                        block
                    )

                    # ------------------------------------------------
                    # A complete dynamic_pose/info update is now
                    # available.
                    # ------------------------------------------------

                    if (
                        timestamp is not None
                        and TARGET_NAMES.issubset(
                            pending_poses.keys()
                        )
                    ):

                        (
                            uav,
                            gimbal,
                            pitch_link,
                            camera_world,
                            R_world_pitch
                        ) = compose_camera_pose(
                            pending_poses
                        )

                        receive_time = (
                            time.monotonic()
                        )

                        update_count += 1

                        if last_timestamp is None:

                            print(
                                f"Update #{update_count}"
                            )

                        else:

                            sim_dt = (
                                timestamp
                                - last_timestamp
                            )

                            receive_dt = (
                                receive_time
                                - last_receive_time
                            )

                            if sim_dt > 0:

                                frequency = (
                                    1.0
                                    / sim_dt
                                )

                            else:

                                frequency = 0.0

                            print(
                                f"Update "
                                f"#{update_count:5d}"
                                f" | sim dt="
                                f"{sim_dt:.4f} s"
                                f" | "
                                f"freq="
                                f"{frequency:6.1f} Hz"
                                f" | receive dt="
                                f"{receive_dt:.4f} s"
                            )

                        if update_count == 1:

                            print()
                            print(
                                "First valid state:"
                            )

                            print(
                                "  UAV = "
                                f"["
                                f"{uav['position'][0]:+.6f}, "
                                f"{uav['position'][1]:+.6f}, "
                                f"{uav['position'][2]:+.6f}"
                                f"]"
                            )

                            print(
                                "  Gimbal = "
                                f"["
                                f"{gimbal['position'][0]:+.6f}, "
                                f"{gimbal['position'][1]:+.6f}, "
                                f"{gimbal['position'][2]:+.6f}"
                                f"]"
                            )

                            print(
                                "  Pitch link = "
                                f"["
                                f"{pitch_link['position'][0]:+.6f}, "
                                f"{pitch_link['position'][1]:+.6f}, "
                                f"{pitch_link['position'][2]:+.6f}"
                                f"]"
                            )

                            print(
                                "  Camera world = "
                                f"["
                                f"{camera_world[0]:+.6f}, "
                                f"{camera_world[1]:+.6f}, "
                                f"{camera_world[2]:+.6f}"
                                f"]"
                            )

                            print()

                        last_timestamp = timestamp
                        last_receive_time = receive_time

                    pending_poses = {}

    except KeyboardInterrupt:

        print()
        print("Stopping...")

    finally:

        process.terminate()

        try:
            process.wait(
                timeout=1.0
            )
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
