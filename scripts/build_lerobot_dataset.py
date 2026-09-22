#!/usr/bin/env python3
"""
Convert a converted vdego-c2 stereo capture into a LeRobot v2.1 dataset,
matching the structure of the supplied reference zip (lerobot_dataset.zip).

WHAT THE REFERENCE LOOKS LIKE, AND WHERE THIS DEPARTS FROM IT
--------------------------------------------------------------
The reference is a MONOCULAR egocentric dataset: one camera key
(observation.images.cam_egocentric), one episode, 5400 frames, with
observation.state and action both 6-DoF [tx,ty,tz,rx,ry,rz] filled
entirely with ZEROS. Its own CAMERA_PARAMETERS doc marks the intrinsics
and extrinsics "SAMPLE *  replace with calibrated values before final
submission", and says to run COLMAP or ORB-SLAM3 for per-frame poses.

This capture has more than the reference could express, so three
deliberate departures, each recorded in meta/info.json under
"_mytron_notes":

  1. TWO camera keys (cam_left, cam_right) instead of one. The capture is
     stereo; collapsing it to one key would discard a camera.
  2. REAL state/action from trajectory.csv instead of zero placeholders.
     state = VIO pose at frame t, action = pose at frame t+1 (the usual
     next-state-as-action convention for passive human recordings; a human
     wearing a headset issues no commanded action). The final frame's
     action repeats its own state, since there is no t+1.
  3. observation.imu added as an extra 6-element feature. The reference
     has no IMU slot; extra observation keys are common in LeRobot
     datasets and readers ignore keys they do not know.

Everything else follows the reference exactly: file layout, info.json key
order and feature block shape, episodes.jsonl / tasks.jsonl line format,
camera.json (MTX / DIST / hand_eye), h264 + yuv420p video, chunks_size
1000, and the same parquet column set and dtypes.

ROTATION CONVENTION
-------------------
The reference names the last three components rx, ry, rz without saying
what they are. trajectory.csv stores orientation as a quaternion, so this
script converts to an AXIS-ANGLE rotation vector (the rotation axis scaled
by the angle in radians) -- the same thing OpenCV's rvec and scipy's
as_rotvec produce. This is recorded explicitly in info.json rather than
left implicit, because Euler angles would also fit the name "rx,ry,rz"
and would be a different quantity.

VIDEO RE-ENCODE
---------------
The converted capture is HEVC (hevc_nvenc, hvc1). The reference declares
h264 / yuv420p, so videos are re-encoded to h264. That is a second lossy
generation on top of the crop; CRF 18 keeps it visually near-transparent,
but note it in any delivery. --copy-video skips the re-encode if you would
rather keep HEVC and edit the declared codec yourself.

USAGE
-----
    python build_lerobot_dataset.py \\
        --input  "Z:\\MytronLabs Internship\\Stereo_1080p" \\
        --output "Z:\\MytronLabs Internship\\lerobot_dataset" \\
        --task-name factory_worker_activity \\
        --task-description "Egocentric video of factory worker performing activities."

The script auto-detects column names in trajectory.csv / imu.csv /
video_frames_timestamp.csv and prints what it matched. If it cannot find a
required column it stops and prints the actual header, rather than guessing.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas pyarrow")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow is required:  pip install pyarrow")

FOOTER_MARKER = "Finished downloading"
CHUNKS_SIZE = 1000
CAMERA_KEYS = {"cam_left": "cam_left.mp4", "cam_right": "cam_right.mp4"}

WARNINGS: list[str] = []


def warn(msg: str):
    WARNINGS.append(msg)
    print(f"  [!] {msg}")


# ----------------------------------------------------------------- helpers

def read_json_stripped(path: Path) -> dict:
    lines = [
        line
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if FOOTER_MARKER not in line
    ]
    return json.loads("\n".join(lines))


def read_csv_stripped(path: Path) -> pd.DataFrame:
    """Read a CSV, dropping this pipeline's trailing non-CSV footer line."""
    df = pd.read_csv(path, encoding="utf-8-sig", engine="python",
                     on_bad_lines="skip")
    first = df.columns[0]
    mask = df[first].astype(str).str.contains(FOOTER_MARKER, na=False)
    if mask.any():
        df = df[~mask]
    return df.reset_index(drop=True)


def _norm(name: str) -> str:
    """Normalise a column name for matching.

    Strips case, separators and bracketed unit suffixes' punctuation, so
    'timestamp(ns)' -> 'timestampns' and 'accel_x (m/s^2)' -> 'accelxms2'.
    """
    out = name.lower()
    for ch in "_ .()[]{}<>-/\\^,:;'\"#*":
        out = out.replace(ch, "")
    return out


def pick_column(df: pd.DataFrame, candidates, required=True, label=""):
    """Find a column by name, tolerating separators and unit suffixes.

    Two passes: exact normalised match first, then a prefix match so a column
    carrying its units ('accel_x(m/s^2)') still matches the bare name
    ('accel_x'). Prefix matches are announced, because a wrong one would be
    silently wrong data rather than a crash.
    """
    norm = {_norm(c): c for c in df.columns}

    for cand in candidates:
        key = _norm(cand)
        if key in norm:
            return norm[key]

    for cand in candidates:
        key = _norm(cand)
        if len(key) < 2:
            continue  # too short to prefix-match safely
        hits = sorted((n for n in norm if n.startswith(key)), key=len)
        if hits:
            chosen = norm[hits[0]]
            print(f"    matched {label or cand!r} -> column {chosen!r} (by prefix)")
            return chosen

    if required:
        sys.exit(
            f"Could not find a column for {label or candidates[0]!r}.\n"
            f"Tried: {candidates}\n"
            f"Actual columns: {list(df.columns)}\n"
            "Add the real name to the candidate list in this script."
        )
    return None


def quat_to_rotvec(qx, qy, qz, qw):
    """Quaternion -> axis-angle rotation vector (same as cv2 rvec)."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    if qw < 0:  # shortest arc
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    s = math.sqrt(qx * qx + qy * qy + qz * qz)
    if s < 1e-12:
        return 0.0, 0.0, 0.0
    angle = 2.0 * math.atan2(s, qw)
    k = angle / s
    return qx * k, qy * k, qz * k


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def ffprobe(path: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt",
        "-of", "json", str(path),
    ], text=True)
    return json.loads(out)["streams"][0]


def fps_of(info: dict) -> float:
    num, _, den = info.get("r_frame_rate", "30/1").partition("/")
    try:
        return int(num) / int(den)
    except (ValueError, ZeroDivisionError):
        return 30.0


# ------------------------------------------------------------ source loading

def load_timestamps(folder: Path, n_frames: int) -> np.ndarray:
    """Per-frame capture times in nanoseconds."""
    path = folder / "video_frames_timestamp.csv"
    if not path.exists():
        warn(f"{folder.name}: no video_frames_timestamp.csv; synthesising from fps.")
        return np.array([], dtype=np.int64)

    df = read_csv_stripped(path)
    col = pick_column(
        df,
        ["timestamp_ns", "timestamp", "ts", "time_ns", "frame_timestamp",
         "capture_time", "time"],
        label="frame timestamp",
    )
    ts = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
    print(f"  timestamps: column {col!r}, {len(ts)} rows")
    if len(ts) < n_frames:
        warn(f"{folder.name}: {len(ts)} timestamps for {n_frames} video frames; "
             "trailing frames fall back to synthesised times.")
    return ts.astype(np.int64)


def load_trajectory(folder: Path) -> pd.DataFrame | None:
    path = folder / "trajectory.csv"
    if not path.exists():
        warn(f"{folder.name}: no trajectory.csv; state/action fall back to zeros.")
        return None

    df = read_csv_stripped(path)
    # EuRoC/Basalt names (p_RS_R_x [m], q_RS_w []) are listed first, since this
    # capture's trajectory.csv uses them.
    cols = {
        "t":  pick_column(df, ["#timestamp", "timestamp_ns", "timestamp", "ts",
                               "time_ns", "time"], label="trajectory timestamp"),
        "px": pick_column(df, ["p_RS_R_x", "px", "tx", "pos_x", "position_x", "t_x", "x"],
                          label="px"),
        "py": pick_column(df, ["p_RS_R_y", "py", "ty", "pos_y", "position_y", "t_y", "y"],
                          label="py"),
        "pz": pick_column(df, ["p_RS_R_z", "pz", "tz", "pos_z", "position_z", "t_z", "z"],
                          label="pz"),
        "qx": pick_column(df, ["q_RS_x", "qx", "quat_x", "q_x", "rot_x"], label="qx"),
        "qy": pick_column(df, ["q_RS_y", "qy", "quat_y", "q_y", "rot_y"], label="qy"),
        "qz": pick_column(df, ["q_RS_z", "qz", "quat_z", "q_z", "rot_z"], label="qz"),
        "qw": pick_column(df, ["q_RS_w", "qw", "quat_w", "q_w", "rot_w"], label="qw"),
    }
    print(f"  trajectory: {len(df)} rows, columns "
          f"{cols['px']}/{cols['py']}/{cols['pz']} + "
          f"{cols['qx']}/{cols['qy']}/{cols['qz']}/{cols['qw']}")

    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in cols.items()})
    return out.dropna().reset_index(drop=True)


def load_imu(folder: Path) -> pd.DataFrame | None:
    path = folder / "imu.csv"
    if not path.exists():
        warn(f"{folder.name}: no imu.csv; observation.imu will be zeros.")
        return None

    df = read_csv_stripped(path)
    # EuRoC/Basalt names first: a_RS_S_* is the accelerometer, w_RS_S_* the gyro.
    cols = {
        "t":  pick_column(df, ["#timestamp", "timestamp_ns", "timestamp", "ts",
                               "time_ns", "time"], label="imu timestamp"),
        "ax": pick_column(df, ["a_RS_S_x", "ax", "accel_x", "acc_x",
                               "linear_acceleration_x", "accelerometer_x", "a_x"],
                          label="ax"),
        "ay": pick_column(df, ["a_RS_S_y", "ay", "accel_y", "acc_y",
                               "linear_acceleration_y", "accelerometer_y", "a_y"],
                          label="ay"),
        "az": pick_column(df, ["a_RS_S_z", "az", "accel_z", "acc_z",
                               "linear_acceleration_z", "accelerometer_z", "a_z"],
                          label="az"),
        "gx": pick_column(df, ["w_RS_S_x", "gx", "gyro_x", "gyr_x",
                               "angular_velocity_x", "gyroscope_x", "omega_x", "g_x"],
                          label="gx"),
        "gy": pick_column(df, ["w_RS_S_y", "gy", "gyro_y", "gyr_y",
                               "angular_velocity_y", "gyroscope_y", "omega_y", "g_y"],
                          label="gy"),
        "gz": pick_column(df, ["w_RS_S_z", "gz", "gyro_z", "gyr_z",
                               "angular_velocity_z", "gyroscope_z", "omega_z", "g_z"],
                          label="gz"),
    }
    print(f"  imu: {len(df)} rows, columns {cols['ax']}..{cols['gz']}")
    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in cols.items()})
    return out.dropna().reset_index(drop=True)


def resample_to_frames(src: pd.DataFrame, value_cols, frame_ts: np.ndarray, n_frames: int):
    """Nearest-neighbour sample of a high-rate stream at each frame time.

    The IMU runs far faster than 30 fps and the VIO stream at its own rate, so
    each frame takes the reading closest in time. Nearest-neighbour rather than
    interpolation: interpolating a quaternion componentwise is wrong, and
    interpolated IMU samples would be readings that were never measured.
    """
    out = np.zeros((n_frames, len(value_cols)), dtype=np.float64)
    if src is None or len(src) == 0:
        return out, None

    src_t = src["t"].to_numpy(dtype=np.float64)
    order = np.argsort(src_t)
    src_t = src_t[order]
    vals = src[list(value_cols)].to_numpy(dtype=np.float64)[order]

    if len(frame_ts) >= n_frames:
        query = frame_ts[:n_frames].astype(np.float64)
    else:
        # No usable frame timestamps: spread frames evenly across the stream.
        query = np.linspace(src_t[0], src_t[-1], n_frames)

    idx = np.searchsorted(src_t, query)
    idx = np.clip(idx, 1, len(src_t) - 1)
    left, right = src_t[idx - 1], src_t[idx]
    choose = np.where(np.abs(query - left) <= np.abs(right - query), idx - 1, idx)
    out = vals[choose]

    gap_ns = float(np.max(np.abs(query - src_t[choose]))) if len(query) else 0.0
    return out, gap_ns


# ------------------------------------------------------------------ video

def convert_video(src: Path, dst: Path, copy_video: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_video:
        shutil.copy2(src, dst)
        print(f"    copied {src.name} -> {dst.relative_to(dst.parents[3])}")
        return

    # The reference declares h264 / yuv420p. The source is HEVC, so this is a
    # real re-encode -- a second lossy generation after the crop.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]
    print(f"    encoding {src.name} -> h264 ...")
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------- build

def build_episode(folder: Path, ep_idx: int, out_root: Path, args, index_offset: int):
    left_video = folder / "cam_left.mp4"
    if not left_video.exists():
        return None

    print(f"\n--- episode {ep_idx:06d}  <-  {folder} ---")

    info = ffprobe(left_video)
    width, height = int(info["width"]), int(info["height"])
    fps = fps_of(info)
    n_frames = int(info["nb_frames"]) if info.get("nb_frames") else None
    if not n_frames:
        sys.exit(f"{left_video}: ffprobe reported no frame count.")
    print(f"  video: {width}x{height} @ {fps:g} fps, {n_frames} frames, "
          f"codec {info.get('codec_name')}")

    # Right camera must have the same frame count or the two streams desync.
    right_video = folder / "cam_right.mp4"
    if right_video.exists():
        r = ffprobe(right_video)
        if int(r["nb_frames"]) != n_frames:
            warn(f"{folder.name}: cam_right has {r['nb_frames']} frames vs "
                 f"cam_left {n_frames}. They will not stay in sync.")
    else:
        warn(f"{folder.name}: no cam_right.mp4; this episode is left-only.")

    frame_ts = load_timestamps(folder, n_frames)
    traj = load_trajectory(folder)
    imu = load_imu(folder)

    # --- state / action -------------------------------------------------
    if traj is not None and len(traj):
        pose, gap = resample_to_frames(
            traj, ["px", "py", "pz", "qx", "qy", "qz", "qw"], frame_ts, n_frames
        )
        state = np.zeros((n_frames, 6), dtype=np.float64)
        state[:, 0:3] = pose[:, 0:3]
        for i in range(n_frames):
            state[i, 3:6] = quat_to_rotvec(*pose[i, 3:7])
        if gap is not None and gap > 5e7:  # 50 ms
            warn(f"{folder.name}: worst frame-to-pose time gap {gap/1e6:.1f} ms "
                 "-- trajectory may not cover the whole video.")
    else:
        state = np.zeros((n_frames, 6), dtype=np.float64)

    # action = next frame's state; the last frame repeats its own (no t+1).
    action = np.vstack([state[1:], state[-1:]]) if n_frames > 1 else state.copy()

    # --- imu ------------------------------------------------------------
    if imu is not None and len(imu) and not args.no_imu:
        imu_arr, imu_gap = resample_to_frames(
            imu, ["ax", "ay", "az", "gx", "gy", "gz"], frame_ts, n_frames
        )
        if imu_gap is not None and imu_gap > 5e7:
            warn(f"{folder.name}: worst frame-to-IMU time gap {imu_gap/1e6:.1f} ms.")
    else:
        imu_arr = np.zeros((n_frames, 6), dtype=np.float64)

    # --- timestamps -----------------------------------------------------
    if len(frame_ts) >= n_frames:
        t0 = frame_ts[0]
        timestamps = (frame_ts[:n_frames] - t0) / 1e9
    else:
        timestamps = np.arange(n_frames) / fps

    # --- parquet --------------------------------------------------------
    chunk = ep_idx // CHUNKS_SIZE
    table = pa.table({
        "timestamp": pa.array(timestamps, type=pa.float32()),
        "frame_index": pa.array(np.arange(n_frames), type=pa.int64()),
        "episode_index": pa.array(np.full(n_frames, ep_idx), type=pa.int64()),
        "index": pa.array(np.arange(index_offset, index_offset + n_frames),
                          type=pa.int64()),
        "task_index": pa.array(np.zeros(n_frames), type=pa.int64()),
        "next.done": pa.array([i == n_frames - 1 for i in range(n_frames)],
                              type=pa.bool_()),
        "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float64())),
        "action": pa.array(action.tolist(), type=pa.list_(pa.float64())),
        "observation.imu": pa.array(imu_arr.tolist(), type=pa.list_(pa.float64())),
    })
    if args.no_imu:
        table = table.drop_columns(["observation.imu"])

    data_path = out_root / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_path)
    print(f"  wrote {data_path.relative_to(out_root)}  ({n_frames} rows)")

    # --- videos ---------------------------------------------------------
    videos_map = {}
    for key, fname in CAMERA_KEYS.items():
        src = folder / fname
        if not src.exists():
            continue
        rel = f"videos/chunk-{chunk:03d}/observation.images.{key}/episode_{ep_idx:06d}.mp4"
        convert_video(src, out_root / rel, args.copy_video)
        videos_map[f"observation.images.{key}"] = rel

    # --- camera parameters ----------------------------------------------
    cam_params = build_camera_entry(folder, width, height, traj)

    return {
        "episode_index": ep_idx,
        "length": n_frames,
        "videos": videos_map,
        "fps": fps,
        "width": width,
        "height": height,
        "camera": cam_params,
        "n_frames": n_frames,
    }


def build_camera_entry(folder: Path, width: int, height: int, traj):
    """camera.json entry: real calibrated MTX where available, plus frame-0 pose."""
    mcap = folder / "mcap_files"
    entry = {"camera_ids": {}, "camera_parameters": {}, "hand_eye": {}}
    res_key = f"{height}P"

    for key, side in (("cam_left", "left"), ("cam_right", "right")):
        if not (folder / CAMERA_KEYS[key]).exists():
            continue
        entry["camera_ids"][key] = f"{folder.name}-{side}"

        f_intr = mcap / f"{side}_rectified_intrinsics.json"
        if f_intr.exists():
            d = read_json_stripped(f_intr)
            K = d["K"]
            fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
            undistorted = all(abs(float(x)) < 1e-9 for x in d.get("D", [0, 0, 0, 0]))
            if int(d.get("image_height", height)) != height:
                warn(f"{folder.name}/{side}: intrinsics declare "
                     f"{d.get('image_width')}x{d.get('image_height')} but the video is "
                     f"{width}x{height} -- camera.json would describe the wrong pixels.")
        else:
            warn(f"{folder.name}/{side}: no rectified intrinsics found; "
                 "camera.json falls back to a nominal pinhole guess.")
            fx = fy = float(width) * 0.55
            cx, cy = width / 2.0, height / 2.0
            undistorted = True

        entry["camera_parameters"][key] = {
            res_key: {
                "MTX": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "DIST": [0.0, 0.0, 0.0, 0.0, 0.0],
                "DIST_MODEL": "radial-tangential",
                "DIST_ORDER": ["k1", "k2", "p1", "p2", "k3"],
                "image_width": width,
                "image_height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "undistorted": bool(undistorted),
            }
        }

    # hand_eye: real frame-0 VIO pose instead of the reference's identity
    # placeholder. Per-frame poses live in the parquet's observation.state.
    if traj is not None and len(traj):
        r0 = traj.iloc[0]
        R = quat_to_R(r0["qx"], r0["qy"], r0["qz"], r0["qw"])
        T = [float(r0["px"]), float(r0["py"]), float(r0["pz"])]
        note = ("Frame-0 VIO pose from trajectory.csv. Per-frame poses are in the "
                "parquet observation.state column, not repeated here.")
    else:
        R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        T = [0.0, 0.0, 0.0]
        note = "No trajectory.csv available; identity placeholder."

    entry["hand_eye"]["cam_left_in_world"] = {
        "note": note,
        "R": R,
        "T": T,
        "T_4x4": [R[0] + [T[0]], R[1] + [T[1]], R[2] + [T[2]], [0.0, 0.0, 0.0, 1.0]],
        "convention": "camera_to_world",
        "convention_note": (
            "trajectory.csv stores the camera/IMU pose IN the world frame, so this is "
            "camera_to_world. The reference zip declared world_to_camera for an "
            "identity placeholder, where the two are indistinguishable. Invert if a "
            "consumer expects world_to_camera."
        ),
    }
    return entry


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="Converted 1080p capture root")
    ap.add_argument("--output", required=True, help="LeRobot dataset root to create")
    ap.add_argument("--task-name", default="factory_worker_activity")
    ap.add_argument("--task-label", default="factory worker activity recording",
                    help="The string that appears in episodes.jsonl 'tasks'")
    ap.add_argument("--task-category", default="industrial_egocentric")
    ap.add_argument("--task-description",
                    default="Egocentric video of a worker performing activities.")
    ap.add_argument("--robot-type", default="egocentric_camera")
    ap.add_argument("--copy-video", action="store_true",
                    help="Copy videos as-is (HEVC) instead of re-encoding to h264. "
                         "The reference declares h264, so this leaves info.json "
                         "describing a codec the files do not use unless you edit it.")
    ap.add_argument("--no-imu", action="store_true",
                    help="Omit observation.imu, matching the reference feature set exactly.")
    ap.add_argument("--include-root", action="store_true",
                    help="Also make an episode from the capture root (raw fisheye); "
                         "by default only segments/ are used.")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    if not src.is_dir():
        sys.exit(f"Not a directory: {src}")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"Output directory is not empty: {out}\nRemove it or pick another path.")

    folders = []
    if args.include_root and (src / "cam_left.mp4").exists():
        folders.append(src)
        warn("Including the capture root as an episode. Those videos are the raw "
             "fisheye pair, not the rectified segments -- their geometry differs from "
             "the pinhole intrinsics written into camera.json.")
    seg = src / "segments"
    if seg.is_dir():
        folders.extend(sorted(d for d in seg.iterdir() if d.is_dir()))
    if not folders:
        sys.exit(f"No cam_left.mp4 found under {src}")

    out.mkdir(parents=True, exist_ok=True)

    episodes = []
    index_offset = 0
    for ep_idx, folder in enumerate(folders):
        ep = build_episode(folder, ep_idx, out, args, index_offset)
        if ep:
            episodes.append(ep)
            index_offset += ep["n_frames"]

    if not episodes:
        sys.exit("No episodes were built.")

    write_metadata(out, episodes, args)

    print("\n" + "=" * 72)
    print(f"DONE. LeRobot v2.1 dataset at: {out}")
    print(f"{len(episodes)} episode(s), {index_offset} frames total")
    if WARNINGS:
        print(f"\n{len(WARNINGS)} item(s) need your attention:\n")
        for w in dict.fromkeys(WARNINGS):
            print(f"  - {w}")
    print("=" * 72)


def write_metadata(out: Path, episodes, args):
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    total_frames = sum(e["length"] for e in episodes)
    total_videos = sum(len(e["videos"]) for e in episodes)
    fps = episodes[0]["fps"]
    height, width = episodes[0]["height"], episodes[0]["width"]
    codec = "hevc" if args.copy_video else "h264"

    features = {}
    for key in CAMERA_KEYS:
        if any(f"observation.images.{key}" in e["videos"] for e in episodes):
            features[f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

    pose_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    features["observation.state"] = {"dtype": "float32", "shape": [6], "names": pose_names}
    features["action"] = {"dtype": "float32", "shape": [6], "names": pose_names}
    if not args.no_imu:
        features["observation.imu"] = {
            "dtype": "float32", "shape": [6],
            "names": ["ax", "ay", "az", "gx", "gy", "gz"],
        }
    for name, dtype in (("timestamp", "float32"), ("frame_index", "int64"),
                        ("episode_index", "int64"), ("index", "int64"),
                        ("task_index", "int64")):
        features[name] = {"dtype": dtype, "shape": [1], "names": None}
    features["next.done"] = {"dtype": "bool", "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": (len(episodes) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": int(round(fps)),
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "_mytron_notes": {
            "source": "vdego-c2 stereo egocentric capture, cropped 1920x1200 -> 1920x1080",
            "departures_from_reference": [
                "Two camera keys (cam_left, cam_right); the reference was monocular.",
                "observation.state / action carry real VIO poses from trajectory.csv "
                "instead of the reference's zero placeholders. state = pose at frame t, "
                "action = pose at frame t+1; the final frame's action repeats its state.",
                "observation.imu added as an extra 6-element feature; the reference has "
                "no IMU slot."
                if not args.no_imu else
                "observation.imu omitted (--no-imu), matching the reference feature set.",
            ],
            "rotation_convention": (
                "rx, ry, rz are an AXIS-ANGLE rotation vector in radians (axis scaled by "
                "angle), converted from the quaternion in trajectory.csv -- the same "
                "convention as cv2 rvec. They are NOT Euler angles."
            ),
            "state_units": "tx, ty, tz in metres; rx, ry, rz in radians.",
            "imu_units": "ax, ay, az in m/s^2; gx, gy, gz in rad/s. Nearest-neighbour "
                         "sampled at each video frame time from the higher-rate stream.",
            "video_provenance": (
                "Re-encoded from HEVC to h264 to match the reference's declared codec; "
                "a second lossy generation after the resolution crop."
                if not args.copy_video else
                "Copied as HEVC without re-encoding (--copy-video)."
            ),
            "not_included": (
                "hand_keypoints.json (MediaPipe 2D + triangulated 3D) is not part of the "
                "reference format and is not carried into this dataset. episode_00.mcap "
                "was excluded as a duplicate of the video and IMU streams."
            ),
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps({
                "episode_index": e["episode_index"],
                "length": e["length"],
                "tasks": [args.task_label],
                "videos": e["videos"],
            }) + "\n")

    with (meta / "tasks.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "task_index": 0,
            "task_name": args.task_name,
            "category": args.task_category,
            "description": args.task_description,
            "language": "en",
        }) + "\n")

    camera = {f"episode_{e['episode_index']:06d}": e["camera"] for e in episodes}
    (meta / "camera.json").write_text(json.dumps(camera, indent=2) + "\n", encoding="utf-8")

    print(f"\n  wrote meta/info.json, episodes.jsonl, tasks.jsonl, camera.json")


if __name__ == "__main__":
    main()
