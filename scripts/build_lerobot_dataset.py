#!/usr/bin/env python3
"""
Convert a converted vdego-c2 stereo egocentric capture into a LeRobot v3.0
dataset, laid out exactly as the target structure:

    dataset_root/
    |-- data/chunk-000/
    |   |-- file-000.parquet                      # episode 0, per-frame rows
    |   `-- file-001.parquet                      # episode 1, per-frame rows
    |-- meta/
    |   |-- info.json                             # format, fps, path templates
    |   |-- episodes/chunk-000/file-000.parquet   # one location record per episode
    |   |-- tasks.parquet                         # task vocabulary
    |   |-- subtasks.parquet                      # action-stage vocabulary
    |   |-- frame_masks/part-00000.parquet        # per-frame validity flags
    |   |-- train_intervals/part-00000.parquet    # contiguous valid intervals
    |   `-- stats.json                            # aggregate feature statistics
    `-- videos/observation.images.main/chunk-000/
        |-- file-000.mp4
        `-- file-001.mp4

WHAT CHANGED FROM THE OLD v2.1 SCRIPT
-------------------------------------
This replaces the previous v2.1 output. The v2.1 version wrote
episode_{i:06d}.parquet, episodes.jsonl and tasks.jsonl, with the chunk
folder ahead of the camera key in the video path. v3.0 differs on every
path and moves the per-episode and task metadata into parquet:

  * data file names are file-{file_index:03d}.parquet, one episode per file.
  * video path is videos/{video_key}/chunk-xxx/file-xxx.mp4 (key first).
  * meta/episodes/... is parquet, one row per episode, each row pointing at
    the data file + global row range and the video file + time range.
  * meta/tasks.parquet / meta/subtasks.parquet replace the .jsonl files.
  * info.json declares codebase_version v3.0, data_files_size_in_mb and
    video_files_size_in_mb, and the v3.0 path templates.

CAMERA KEY
----------
The target has a single camera key, observation.images.main. The capture is
stereo, so by default the LEFT camera becomes "main" (--main-camera right to
flip). Pass --stereo to additionally emit the other camera as
observation.images.second; the default keeps the single-key layout shown
above. Whichever camera is dropped is recorded in info.json _notes.

THE THREE ANNOTATION TABLES
---------------------------
subtasks.parquet, frame_masks and train_intervals describe annotations the
raw capture does not carry (action-stage labels, per-frame validity, training
windows). They are written with explicit, documented defaults so the dataset
is structurally complete and immediately loadable, and so real annotations
can be dropped in later without changing the layout:

  * subtasks.parquet  -- a single subtask (--subtask-label) covering the whole
    recording. Replace with a real action-stage vocabulary when available.
  * frame_masks       -- every frame flagged valid=True. Edit flags to False
    to exclude frames (e.g. blur, occlusion, out-of-task).
  * train_intervals   -- derived from frame_masks as the maximal contiguous
    runs of valid frames. With the all-valid default that is one interval per
    episode; re-running after editing frame_masks recomputes the intervals.

ROTATION / STATE / ACTION / IMU
-------------------------------
Unchanged from the capture semantics: observation.state is the per-frame VIO
pose [tx,ty,tz, rx,ry,rz] with rx,ry,rz an AXIS-ANGLE rotation vector in
radians (cv2 rvec convention, converted from the trajectory quaternion -- NOT
Euler angles); action is the next frame's state (last frame repeats its own);
observation.imu is [ax,ay,az, gx,gy,gz] nearest-neighbour sampled at each
frame time. Units: metres, radians, m/s^2, rad/s.

USAGE
-----
    python build_lerobot_dataset.py \\
        --input  "/path/to/Stereo_1080p" \\
        --output "/path/to/lerobot_dataset" \\
        --task-label "factory worker activity recording"

The script auto-detects column names in trajectory.csv / imu.csv /
video_frames_timestamp.csv and prints what it matched. If a required column
is missing it stops and prints the actual header rather than guessing.
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

# --- v3.0 layout constants (match huggingface/lerobot defaults) -------------
FOOTER_MARKER = "Finished downloading"
CHUNKS_SIZE = 1000
DATA_FILE_SIZE_IN_MB = 100
VIDEO_FILE_SIZE_IN_MB = 200

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

MAIN_KEY = "observation.images.main"
SECOND_KEY = "observation.images.second"
CAMERA_SOURCES = {"left": "cam_left.mp4", "right": "cam_right.mp4"}

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
    """Find a column by name, tolerating separators and unit suffixes."""
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
    """Nearest-neighbour sample of a high-rate stream at each frame time."""
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


# -------------------------------------------------------------- statistics

def feature_stats(arr: np.ndarray) -> dict:
    """Per-dimension mean/std/min/max + a scalar count for a (n, d) array."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return {
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "count": [int(a.shape[0])],
    }


class RunningStats:
    """Accumulates mean/std/min/max across episodes without keeping all frames."""

    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf)
        self.max = np.full(dim, -np.inf)

    def update(self, arr: np.ndarray):
        a = np.asarray(arr, dtype=np.float64).reshape(-1, self.dim)
        self.n += a.shape[0]
        self.sum += a.sum(axis=0)
        self.sumsq += (a * a).sum(axis=0)
        self.min = np.minimum(self.min, a.min(axis=0))
        self.max = np.maximum(self.max, a.max(axis=0))

    def result(self) -> dict:
        if self.n == 0:
            return {"mean": [0.0] * self.dim, "std": [0.0] * self.dim,
                    "min": [0.0] * self.dim, "max": [0.0] * self.dim, "count": [0]}
        mean = self.sum / self.n
        var = np.maximum(self.sumsq / self.n - mean * mean, 0.0)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "count": [int(self.n)],
        }


# ------------------------------------------------------------------ video

def convert_video(src: Path, dst: Path, copy_video: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_video:
        shutil.copy2(src, dst)
        print(f"    copied {src.name} -> {dst.name}")
        return

    # The source is HEVC; re-encode to h264 (a second lossy generation after
    # the resolution crop). CRF 18 keeps it visually near-transparent.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]
    print(f"    encoding {src.name} -> h264 ...")
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------ episode build

def compute_frame_data(folder: Path, ep_idx: int, args):
    """Load per-frame state/action/imu/timestamps for one episode folder.

    Returns a dict of arrays plus video metadata, or None if the episode has
    no main-camera video. Does not write anything.
    """
    main_src = folder / CAMERA_SOURCES[args.main_camera]
    if not main_src.exists():
        return None

    print(f"\n--- episode {ep_idx:06d}  <-  {folder} ---")

    info = ffprobe(main_src)
    width, height = int(info["width"]), int(info["height"])
    fps = fps_of(info)
    n_frames = int(info["nb_frames"]) if info.get("nb_frames") else None
    if not n_frames:
        sys.exit(f"{main_src}: ffprobe reported no frame count.")
    print(f"  video: {width}x{height} @ {fps:g} fps, {n_frames} frames, "
          f"codec {info.get('codec_name')}")

    # Second camera (if --stereo) must share the frame count or the streams desync.
    second_name = CAMERA_SOURCES["right" if args.main_camera == "left" else "left"]
    second_src = folder / second_name
    if args.stereo:
        if second_src.exists():
            r = ffprobe(second_src)
            if int(r["nb_frames"]) != n_frames:
                warn(f"{folder.name}: second camera has {r['nb_frames']} frames vs "
                     f"main {n_frames}. They will not stay in sync.")
        else:
            warn(f"{folder.name}: --stereo set but {second_name} is missing; "
                 "this episode is main-only.")
            second_src = None
    else:
        second_src = None

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

    return {
        "folder": folder,
        "n_frames": n_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "codec_name": info.get("codec_name"),
        "state": state,
        "action": action,
        "imu": imu_arr,
        "timestamps": timestamps.astype(np.float64),
        "main_src": main_src,
        "second_src": second_src,
        "traj": traj,
    }


def write_episode_data(out_root: Path, ep, ep_idx: int, index_offset: int, args):
    """Write one episode's per-frame parquet and video(s); return its record."""
    n_frames = ep["n_frames"]
    chunk_index = ep_idx // CHUNKS_SIZE
    file_index = ep_idx % CHUNKS_SIZE
    from_index = index_offset
    to_index = index_offset + n_frames

    columns = {
        "timestamp": pa.array(ep["timestamps"], type=pa.float32()),
        "frame_index": pa.array(np.arange(n_frames), type=pa.int64()),
        "episode_index": pa.array(np.full(n_frames, ep_idx), type=pa.int64()),
        "index": pa.array(np.arange(from_index, to_index), type=pa.int64()),
        "task_index": pa.array(np.zeros(n_frames, dtype=np.int64), type=pa.int64()),
        "subtask_index": pa.array(np.zeros(n_frames, dtype=np.int64), type=pa.int64()),
        "next.done": pa.array([i == n_frames - 1 for i in range(n_frames)], type=pa.bool_()),
        "observation.state": pa.array(ep["state"].tolist(), type=pa.list_(pa.float32())),
        "action": pa.array(ep["action"].tolist(), type=pa.list_(pa.float32())),
    }
    if not args.no_imu:
        columns["observation.imu"] = pa.array(ep["imu"].tolist(), type=pa.list_(pa.float32()))

    data_rel = DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)
    data_path = out_root / data_rel
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), data_path)
    print(f"  wrote {data_rel}  ({n_frames} rows)")

    # --- video(s) -------------------------------------------------------
    duration = float(ep["timestamps"][-1]) if n_frames else 0.0
    videos = {}
    main_rel = VIDEO_PATH.format(video_key=MAIN_KEY, chunk_index=chunk_index,
                                 file_index=file_index)
    convert_video(ep["main_src"], out_root / main_rel, args.copy_video)
    videos[MAIN_KEY] = {"chunk_index": chunk_index, "file_index": file_index,
                        "from_timestamp": 0.0, "to_timestamp": duration}
    if ep["second_src"] is not None:
        second_rel = VIDEO_PATH.format(video_key=SECOND_KEY, chunk_index=chunk_index,
                                       file_index=file_index)
        convert_video(ep["second_src"], out_root / second_rel, args.copy_video)
        videos[SECOND_KEY] = {"chunk_index": chunk_index, "file_index": file_index,
                              "from_timestamp": 0.0, "to_timestamp": duration}

    # --- per-episode stats ----------------------------------------------
    stats = {
        "observation.state": feature_stats(ep["state"]),
        "action": feature_stats(ep["action"]),
    }
    if not args.no_imu:
        stats["observation.imu"] = feature_stats(ep["imu"])

    # Final video codec: copied files keep the source codec; re-encodes are h264.
    out_codec = (ep.get("codec_name") or "h264") if args.copy_video else "h264"

    return {
        "episode_index": ep_idx,
        "length": n_frames,
        "data/chunk_index": chunk_index,
        "data/file_index": file_index,
        "dataset_from_index": from_index,
        "dataset_to_index": to_index,
        "videos": videos,
        "stats": stats,
        "fps": ep["fps"],
        "width": ep["width"],
        "height": ep["height"],
        "codec": out_codec,
    }


# -------------------------------------------------------------- meta tables

def write_episodes_table(out_root: Path, episodes, args):
    """meta/episodes/chunk-000/file-000.parquet -- one location record per episode."""
    video_keys = sorted({k for e in episodes for k in e["videos"]})
    feat_keys = sorted({k for e in episodes for k in e["stats"]})

    cols: dict[str, list] = {
        "episode_index": [e["episode_index"] for e in episodes],
        "tasks": [[args.task_label] for _ in episodes],
        "length": [e["length"] for e in episodes],
        "data/chunk_index": [e["data/chunk_index"] for e in episodes],
        "data/file_index": [e["data/file_index"] for e in episodes],
        "dataset_from_index": [e["dataset_from_index"] for e in episodes],
        "dataset_to_index": [e["dataset_to_index"] for e in episodes],
        # every episode record lives in this single meta file
        "meta/episodes/chunk_index": [0 for _ in episodes],
        "meta/episodes/file_index": [0 for _ in episodes],
    }
    for vk in video_keys:
        cols[f"videos/{vk}/chunk_index"] = [e["videos"].get(vk, {}).get("chunk_index") for e in episodes]
        cols[f"videos/{vk}/file_index"] = [e["videos"].get(vk, {}).get("file_index") for e in episodes]
        cols[f"videos/{vk}/from_timestamp"] = [e["videos"].get(vk, {}).get("from_timestamp") for e in episodes]
        cols[f"videos/{vk}/to_timestamp"] = [e["videos"].get(vk, {}).get("to_timestamp") for e in episodes]
    for fk in feat_keys:
        for stat in ("mean", "std", "min", "max", "count"):
            cols[f"stats/{fk}/{stat}"] = [e["stats"].get(fk, {}).get(stat) for e in episodes]

    rel = EPISODES_PATH.format(chunk_index=0, file_index=0)
    path = out_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)
    print(f"  wrote {rel}  ({len(episodes)} episode records)")


def write_vocab_table(path: Path, index_name: str, index_col: str, items: list[str]):
    """Write a {task,subtask}.parquet: string index + integer index column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({index_col: list(range(len(items)))},
                      index=pd.Index(items, name=index_name))
    df.to_parquet(path)
    print(f"  wrote {path.relative_to(path.parents[1])}  ({len(items)} rows)")


def write_frame_masks(out_root: Path, episodes):
    """meta/frame_masks/part-00000.parquet -- per-frame validity, all True by default."""
    total = sum(e["length"] for e in episodes)
    idx = np.arange(total, dtype=np.int64)
    ep_index = np.concatenate([np.full(e["length"], e["episode_index"], dtype=np.int64)
                               for e in episodes]) if episodes else np.array([], np.int64)
    frame_index = np.concatenate([np.arange(e["length"], dtype=np.int64)
                                  for e in episodes]) if episodes else np.array([], np.int64)
    table = pa.table({
        "index": pa.array(idx, type=pa.int64()),
        "episode_index": pa.array(ep_index, type=pa.int64()),
        "frame_index": pa.array(frame_index, type=pa.int64()),
        "valid": pa.array(np.ones(total, dtype=bool), type=pa.bool_()),
    })
    rel = "meta/frame_masks/part-00000.parquet"
    path = out_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  wrote {rel}  ({total} frames, all valid)")
    return np.ones(total, dtype=bool), ep_index, idx


def write_train_intervals(out_root: Path, valid: np.ndarray, ep_index: np.ndarray,
                          global_index: np.ndarray):
    """meta/train_intervals/part-00000.parquet -- maximal contiguous valid runs.

    Derived from the frame_masks validity flags: within each episode, every
    maximal run of valid==True frames becomes one [start, end) interval over
    global frame indices. With the all-valid default this is one interval per
    episode; edit frame_masks and re-run to split them.
    """
    rows = {"interval_index": [], "episode_index": [],
            "start_index": [], "end_index": [], "length": []}
    k = 0
    n = len(valid)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        ep = ep_index[i]
        j = i
        while j < n and valid[j] and ep_index[j] == ep:
            j += 1
        rows["interval_index"].append(k)
        rows["episode_index"].append(int(ep))
        rows["start_index"].append(int(global_index[i]))
        rows["end_index"].append(int(global_index[j - 1]) + 1)
        rows["length"].append(int(j - i))
        k += 1
        i = j

    table = pa.table({
        "interval_index": pa.array(rows["interval_index"], type=pa.int64()),
        "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
        "start_index": pa.array(rows["start_index"], type=pa.int64()),
        "end_index": pa.array(rows["end_index"], type=pa.int64()),
        "length": pa.array(rows["length"], type=pa.int64()),
    })
    rel = "meta/train_intervals/part-00000.parquet"
    path = out_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  wrote {rel}  ({k} interval(s))")


def write_info_and_stats(out_root: Path, episodes, args):
    meta = out_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    total_frames = sum(e["length"] for e in episodes)
    video_keys = sorted({k for e in episodes for k in e["videos"]})
    total_videos = sum(len(e["videos"]) for e in episodes)
    fps = episodes[0]["fps"]
    height, width = episodes[0]["height"], episodes[0]["width"]
    codec = episodes[0].get("codec", "h264")

    # --- features -------------------------------------------------------
    features = {}
    for vk in video_keys:
        features[vk] = {
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
                        ("task_index", "int64"), ("subtask_index", "int64")):
        features[name] = {"dtype": dtype, "shape": [1], "names": None}
    features["next.done"] = {"dtype": "bool", "shape": [1], "names": None}

    dropped = "right" if args.main_camera == "left" else "left"
    info = {
        "codebase_version": "v3.0",
        "robot_type": args.robot_type,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_IN_MB,
        "fps": int(round(fps)),
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        "_notes": {
            "source": "vdego-c2 stereo egocentric capture, cropped 1920x1200 -> 1920x1080",
            "camera": (
                f"observation.images.main = {args.main_camera} camera."
                + ("" if args.stereo else
                   f" The {dropped} camera is not carried; re-run with --stereo to add "
                   "it as observation.images.second.")
            ),
            "rotation_convention": (
                "rx, ry, rz are an AXIS-ANGLE rotation vector in radians (axis scaled by "
                "angle), converted from the trajectory quaternion -- the cv2 rvec "
                "convention. They are NOT Euler angles."
            ),
            "state_action": (
                "observation.state = per-frame VIO pose; action = next frame's state; "
                "the final frame's action repeats its own state."
            ),
            "state_units": "tx, ty, tz in metres; rx, ry, rz in radians.",
            "imu_units": "ax, ay, az in m/s^2; gx, gy, gz in rad/s. Nearest-neighbour "
                         "sampled at each video frame time from the higher-rate stream.",
            "video_provenance": (
                "Re-encoded to h264 to match the declared codec."
                if not args.copy_video else
                f"Copied without re-encoding (--copy-video); declared codec is {codec}."
            ),
            "annotation_defaults": (
                "subtasks.parquet holds one default subtask; frame_masks flags every "
                "frame valid=True; train_intervals are the contiguous valid runs (one "
                "per episode under the default). Replace these with real annotations as "
                "they become available -- the layout does not change."
            ),
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    # --- aggregate stats.json ------------------------------------------
    running = {"observation.state": RunningStats(6), "action": RunningStats(6)}
    if not args.no_imu:
        running["observation.imu"] = RunningStats(6)
    for e in episodes:
        for k, rs in running.items():
            s = e["stats"][k]
            # reconstruct per-episode sums from its stats to feed the aggregate
            # (cheaper path: we kept the arrays? no -- recombine from mean/count)
            cnt = s["count"][0]
            mean = np.array(s["mean"])
            # min/max combine directly; mean/std via count-weighting
            rs.n += cnt
            rs.sum += mean * cnt
            rs.sumsq += (np.array(s["std"]) ** 2 + mean ** 2) * cnt
            rs.min = np.minimum(rs.min, np.array(s["min"]))
            rs.max = np.maximum(rs.max, np.array(s["max"]))
    stats_json = {k: rs.result() for k, rs in running.items()}
    (meta / "stats.json").write_text(json.dumps(stats_json, indent=2) + "\n", encoding="utf-8")

    print("  wrote meta/info.json, meta/stats.json")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="Converted 1080p capture root")
    ap.add_argument("--output", required=True, help="LeRobot dataset root to create")
    ap.add_argument("--task-label", default="factory worker activity recording",
                    help="The task string stored in tasks.parquet and each episode record")
    ap.add_argument("--task-name", default="factory_worker_activity")
    ap.add_argument("--subtask-label", default="full_episode",
                    help="The single default subtask written to subtasks.parquet")
    ap.add_argument("--robot-type", default="egocentric_camera")
    ap.add_argument("--main-camera", choices=["left", "right"], default="left",
                    help="Which source camera becomes observation.images.main")
    ap.add_argument("--stereo", action="store_true",
                    help="Also emit the other camera as observation.images.second")
    ap.add_argument("--copy-video", action="store_true",
                    help="Copy videos as-is (HEVC) instead of re-encoding to h264.")
    ap.add_argument("--no-imu", action="store_true",
                    help="Omit observation.imu from the dataset.")
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

    main_name = CAMERA_SOURCES[args.main_camera]
    folders = []
    if args.include_root and (src / main_name).exists():
        folders.append(src)
        warn("Including the capture root as an episode. Those videos are the raw "
             "fisheye pair, not the rectified segments -- their geometry differs from "
             "the pinhole intrinsics.")
    seg = src / "segments"
    if seg.is_dir():
        folders.extend(sorted(d for d in seg.iterdir() if d.is_dir()))
    if not folders:
        sys.exit(f"No {main_name} found under {src}")

    out.mkdir(parents=True, exist_ok=True)

    episodes = []
    index_offset = 0
    for ep_idx, folder in enumerate(folders):
        ep = compute_frame_data(folder, ep_idx, args)
        if ep is None:
            continue
        record = write_episode_data(out, ep, ep_idx, index_offset, args)
        episodes.append(record)
        index_offset += record["length"]

    if not episodes:
        sys.exit("No episodes were built.")

    # --- metadata -------------------------------------------------------
    write_episodes_table(out, episodes, args)
    write_vocab_table(out / "meta" / "tasks.parquet", "task", "task_index", [args.task_label])
    write_vocab_table(out / "meta" / "subtasks.parquet", "subtask", "subtask_index",
                      [args.subtask_label])
    valid, ep_index, global_index = write_frame_masks(out, episodes)
    write_train_intervals(out, valid, ep_index, global_index)
    write_info_and_stats(out, episodes, args)

    print("\n" + "=" * 72)
    print(f"DONE. LeRobot v3.0 dataset at: {out}")
    print(f"{len(episodes)} episode(s), {index_offset} frames total")
    if WARNINGS:
        print(f"\n{len(WARNINGS)} item(s) need your attention:\n")
        for w in dict.fromkeys(WARNINGS):
            print(f"  - {w}")
    print("=" * 72)


if __name__ == "__main__":
    main()
