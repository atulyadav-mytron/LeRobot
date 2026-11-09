#!/usr/bin/env python3
"""
Build a LeRobot v3.0 *hand-manipulation* dataset (robot_type: human_hand) from
raw EgoCapture-1 clips, matching the nvwa_2full_episodes reference schema exactly.

This supersedes build_lerobot_dataset.py for this target: that script produced a
camera-pose/IMU schema; the real reference is hand-centric. The two schemas share
only the v3.0 skeleton (chunk/file paths, episodes/tasks/subtasks parquet,
frame_masks, train_intervals).

WHAT THIS FILLS NOW (no extra inputs needed)
--------------------------------------------
  * observation.images.main          fisheye -> pinhole, undistorted to 640x640
  * observation.state.camera_intrinsics [9]   the 3x3 K of that 640 view
  * timestamp / frame_index / episode_index / index / task_index / subtask_index
  * the complete meta tree: info.json (incl. window_contract / frame_filter /
    video_policy), episodes, tasks, subtasks, frame_masks (19 cols), train_intervals
  * task / subtask / scene text pulled from each clip's <stem>.json when present

WHAT IS A TYPED PLACEHOLDER UNTIL UPSTREAM DATA ARRIVES
------------------------------------------------------
Each is written with the correct dtype/shape so the dataset loads, and is filled
for real by pointing the matching --*-dir at per-clip files:

  * observation.state.hand_{right,left}_cam            [63]  zeros
  * observation.state.hand_{right,left}_cam_rotation   [84]  identity quats
        -> --hands-dir   : per-clip <stem>.npz with arrays
                           hand_right_cam[N,63], hand_right_cam_rotation[N,84],
                           hand_left_cam[N,63],  hand_left_cam_rotation[N,84]
           (produce with a 3D hand-pose pipeline: MediaPipe/HaMeR/WiLoR)
  * observation.state.camera_position [3] / camera_rotation [4]   zeros / identity
        -> --pose-dir    : per-clip <stem>.(tum|csv) SLAM trajectory
  * frame_masks quality flags                           all True / train_keep True
        -> --quality-dir : per-clip <stem>.parquet with the bool columns
           (produce with the hand-kinematics + VLM validation pipeline)
  * subtask_index per frame + subtask vocabulary        single subtask
        -> --anno        : JSON mapping clip -> tasks/subtasks/segmentation

USAGE
    python build_hand_dataset.py --input <clips_dir_or_clip> --output <dataset> \
        [--hands-dir D] [--pose-dir D] [--quality-dir D] [--anno anno.json] \
        [--hfov 110] [--limit N]
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow is required:  pip install pyarrow")
try:
    import cv2
except ImportError:
    sys.exit("opencv is required:  pip install opencv-python-headless")

# --- v3.0 layout ------------------------------------------------------------
CHUNKS_SIZE = 1000
VIDEO_KEY = "observation.images.main"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

# --- target geometry --------------------------------------------------------
OUT = 640
HAND_J = 21
STATE_DIMS = {
    "camera_position": 3, "camera_rotation": 4, "camera_intrinsics": 9,
    "hand_right_cam": 63, "hand_right_cam_rotation": 84,
    "hand_left_cam": 63, "hand_left_cam_rotation": 84,
}

# --- EgoCapture-1 fisheye calibration (calib_all_s30-camchain.yaml) ----------
FISHEYE_K = np.array([[707.2808075315367, 0.0, 956.0608949841028],
                      [0.0, 707.6480183518742, 550.7429189294107],
                      [0.0, 0.0, 1.0]])
FISHEYE_D = np.array([0.011734004280818429, 9.279535348401179e-06,
                      -0.00015706402699172342, -0.0002178771185773861])

# frame-mask boolean columns, in reference order. Quality flags default True;
# the three below default False (matching the reference's row 0).
MASK_FLAGS = [
    "is_hand_bmc_valid", "is_hand_velocity_valid", "is_hand_acceleration_valid",
    "is_hand_jerk_valid", "is_hand_coordinate_range_valid", "is_hand_bone_length_valid",
    "is_camera_rotation_valid", "is_vlm_video_quality_valid", "is_vlm_informative_valid",
    "is_vlm_execution_valid", "is_vlm_hand_position_valid", "is_vlm_hand_left_right_valid",
    "is_vlm_hand_shape_valid", "keep_all_quality", "validation_50h", "validation_online",
    "train_keep", "train100_keep",
]
MASK_DEFAULT_FALSE = {"validation_50h", "validation_online", "train100_keep"}


# ----------------------------------------------------------------- helpers

def ffprobe(path: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", str(path)], text=True)
    return json.loads(out)["streams"][0]


def fps_of(stream: dict) -> float:
    num, _, den = stream.get("r_frame_rate", "30/1").partition("/")
    try:
        return int(num) / int(den)
    except (ValueError, ZeroDivisionError):
        return 30.0


def new_K(hfov_deg: float) -> np.ndarray:
    """Pinhole K for the OUTxOUT undistorted view at the given horizontal FOV."""
    fp = (OUT / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    return np.array([[fp, 0.0, OUT / 2.0], [0.0, fp, OUT / 2.0], [0.0, 0.0, 1.0]])


def undistort_to_640(src: Path, dst: Path, fps: float, K640: np.ndarray) -> int:
    """Fisheye -> pinhole 640x640, streamed through ffmpeg; returns frame count."""
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        FISHEYE_K, FISHEYE_D, np.eye(3), K640, (OUT, OUT), cv2.CV_16SC2)
    stream = ffprobe(src)
    sw, sh = int(stream["width"]), int(stream["height"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbytes = sw * sh * 3
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{OUT}x{OUT}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264",
         "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", "-an", str(dst)],
        stdin=subprocess.PIPE)
    n = 0
    try:
        while True:
            raw = dec.stdout.read(nbytes)
            if len(raw) < nbytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(sh, sw, 3)
            out = cv2.remap(frame, map1, map2, cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT)
            enc.stdin.write(out.tobytes())
            n += 1
            if n % 900 == 0:
                print(f"    undistort {n} frames", flush=True)
    finally:
        if enc.stdin:
            enc.stdin.close()
        dec.wait()
        enc.wait()
    print(f"    undistorted {n} frames -> {dst.name}", flush=True)
    return n


def fsl(arr2d: np.ndarray, dim: int) -> pa.Array:
    """numpy (n, dim) float -> fixed_size_list<float>[dim]."""
    a = np.asarray(arr2d, dtype=np.float32).reshape(-1, dim)
    return pa.array(a.tolist(), type=pa.list_(pa.float32(), dim))


def identity_quats(n: int, joints: int) -> np.ndarray:
    """(n, joints*4) filled with the identity quaternion [0,0,0,1] per joint."""
    q = np.zeros((n, joints, 4), dtype=np.float32)
    q[..., 3] = 1.0
    return q.reshape(n, joints * 4)


# ----------------------------------------------------- optional real inputs

def load_hands(hands_dir, stem, n):
    if not hands_dir:
        return None
    p = Path(hands_dir) / f"{stem}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: np.asarray(z[k], dtype=np.float32) for k in
            ("hand_right_cam", "hand_right_cam_rotation",
             "hand_left_cam", "hand_left_cam_rotation") if k in z}


def load_pose(pose_dir, stem, n, fps):
    if not pose_dir:
        return None
    for ext in (".tum", ".csv", ".txt"):
        p = Path(pose_dir) / f"{stem}{ext}"
        if p.exists():
            break
    else:
        return None
    pos = np.zeros((n, 3), np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))
    rows = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) >= 8:
            rows.append([float(x) for x in parts[:8]])
    if rows:
        arr = np.array(rows)  # t tx ty tz qx qy qz qw
        m = min(n, len(arr))
        pos[:m] = arr[:m, 1:4]
        quat[:m] = arr[:m, 4:8]
    return pos, quat


def load_quality(quality_dir, stem, n):
    if not quality_dir:
        return None
    p = Path(quality_dir) / f"{stem}.parquet"
    if not p.exists():
        return None
    return pq.read_table(p).to_pandas()


def load_anno(anno_path, stem, meta_json):
    """Return (task_text, scene, scene_label, subtask_names, subtask_per_frame?).

    Priority: --anno JSON entry > clip <stem>.json metadata > generic default.
    subtask_per_frame is None unless the anno provides segmentation.
    """
    task = "activity recording"
    scene = scene_label = "unknown"
    subtasks = ["full recording"]
    seg = None
    if meta_json:
        task = meta_json.get("task_description", task)
        scene = meta_json.get("environment_l3", meta_json.get("environment-id", scene))
        scene_label = scene
    if anno_path and Path(anno_path).exists():
        anno = json.loads(Path(anno_path).read_text())
        e = anno.get(stem) or anno.get(stem.split("-")[-1]) or {}
        task = e.get("task", task)
        scene = e.get("scene", scene)
        scene_label = e.get("scene_label", scene_label)
        subtasks = e.get("subtasks", subtasks)
        seg = e.get("subtask_per_frame")  # list[int] length n, optional
    return task, scene, scene_label, subtasks, seg


# ------------------------------------------------------------------ build

def build_clip(stem_path: Path, ep_idx: int, out_root: Path, index_offset: int,
               K640: np.ndarray, args, task_table, subtask_table):
    mp4 = stem_path.with_suffix(".mp4")
    if not mp4.exists():
        return None
    stem = stem_path.name
    print(f"\n--- episode {ep_idx:06d}  <-  {stem} ---", flush=True)

    stream = ffprobe(mp4)
    fps = fps_of(stream)
    chunk_index, file_index = ep_idx // CHUNKS_SIZE, ep_idx % CHUNKS_SIZE

    # video: reuse a pre-undistorted 640 clip if --video-dir provides one (the
    # batch pipeline undistorts once and reuses it for VO + the dataset), else
    # undistort from the source here.
    video_rel = VIDEO_PATH.format(video_key=VIDEO_KEY,
                                  chunk_index=chunk_index, file_index=file_index)
    dst = out_root / video_rel
    pre = (Path(args.video_dir) / f"{stem}.mp4") if args.video_dir else None
    if pre is not None and pre.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pre, dst)
        vs = ffprobe(dst)
        n = int(vs["nb_frames"]) if vs.get("nb_frames") else 0
        fps = fps_of(vs)
        print(f"  reused 640 video {pre.name} ({n} frames)")
    else:
        n = undistort_to_640(mp4, dst, fps, K640)

    # annotations / metadata
    meta_json = None
    jp = stem_path.with_suffix(".json")
    if jp.exists():
        try:
            meta_json = json.loads(jp.read_text())
        except json.JSONDecodeError:
            meta_json = None
    task_text, scene, scene_label, subtask_names, seg = load_anno(args.anno, stem, meta_json)
    ep_uuid = (meta_json or {}).get("episode-uuid", f"episode_{ep_idx:06d}")

    # task / subtask indices (dedup into the shared vocab tables)
    task_index = task_table.setdefault(task_text, len(task_table))
    local_sub_idx = []
    for s in subtask_names:
        local_sub_idx.append(subtask_table.setdefault(s, len(subtask_table)))
    if seg is not None and len(seg) >= n:
        subtask_per_frame = np.array(
            [local_sub_idx[min(v, len(local_sub_idx) - 1)] for v in seg[:n]], np.int64)
    else:
        subtask_per_frame = np.full(n, local_sub_idx[0], np.int64)

    # --- per-frame state --------------------------------------------------
    intr = np.tile(K640.astype(np.float32).reshape(9), (n, 1))     # real
    hands = load_hands(args.hands_dir, stem, n) or {}
    pose = load_pose(args.pose_dir, stem, n, fps)
    if pose is not None:
        cam_pos, cam_rot = pose
    else:
        cam_pos = np.zeros((n, 3), np.float32)
        cam_rot = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))

    def hand(key, dim, rot=False):
        if key in hands and hands[key].shape[0] >= n:
            return hands[key][:n]
        return identity_quats(n, dim // 4) if rot else np.zeros((n, dim), np.float32)

    timestamps = (np.arange(n) / fps).astype(np.float32)  # per-episode, starts at 0
    cols = {
        "timestamp": pa.array(timestamps, type=pa.float32()),
        "frame_index": pa.array(np.arange(n), type=pa.int64()),
        "episode_index": pa.array(np.full(n, ep_idx), type=pa.int64()),
        "index": pa.array(np.arange(index_offset, index_offset + n), type=pa.int64()),
        "task_index": pa.array(np.full(n, task_index), type=pa.int64()),
        "subtask_index": pa.array(subtask_per_frame, type=pa.int64()),
        "observation.state.camera_position": fsl(cam_pos, 3),
        "observation.state.camera_rotation": fsl(cam_rot, 4),
        "observation.state.camera_intrinsics": fsl(intr, 9),
        "observation.state.hand_right_cam": fsl(hand("hand_right_cam", 63), 63),
        "observation.state.hand_right_cam_rotation": fsl(hand("hand_right_cam_rotation", 84, True), 84),
        "observation.state.hand_left_cam": fsl(hand("hand_left_cam", 63), 63),
        "observation.state.hand_left_cam_rotation": fsl(hand("hand_left_cam_rotation", 84, True), 84),
    }
    data_rel = DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)
    (out_root / data_rel).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out_root / data_rel)
    print(f"  wrote {data_rel}  ({n} rows)")

    # frame-mask values (real if quality file given, else defaults)
    q = load_quality(args.quality_dir, stem, n)
    return {
        "episode_index": ep_idx, "episode_uuid": ep_uuid,
        "scene": scene, "scene_label": scene_label,
        "task_text": task_text, "task_index": task_index,
        "subtask_indices": sorted(set(int(x) for x in subtask_per_frame.tolist())),
        "length": n, "fps": fps,
        "dataset_from_index": index_offset, "dataset_to_index": index_offset + n,
        "data/chunk_index": chunk_index, "data/file_index": file_index,
        "video_from": 0.0, "video_to": float(n / fps),
        "subtask_per_frame": subtask_per_frame, "quality": q,
    }


# -------------------------------------------------------------- meta tables

def write_episodes(out_root, eps):
    cols = {
        "episode_index": [e["episode_index"] for e in eps],
        "episode_uuid": [e["episode_uuid"] for e in eps],
        "scene": [e["scene"] for e in eps],
        "scene_label": [e["scene_label"] for e in eps],
        "tasks": [[e["task_text"]] for e in eps],
        "subtask_indices": [e["subtask_indices"] for e in eps],
        "length": [e["length"] for e in eps],
        "dataset_from_index": [e["dataset_from_index"] for e in eps],
        "dataset_to_index": [e["dataset_to_index"] for e in eps],
        "data/chunk_index": [e["data/chunk_index"] for e in eps],
        "data/file_index": [e["data/file_index"] for e in eps],
        "meta/episodes/chunk_index": [0 for _ in eps],
        "meta/episodes/file_index": [0 for _ in eps],
        f"videos/{VIDEO_KEY}/chunk_index": [e["data/chunk_index"] for e in eps],
        f"videos/{VIDEO_KEY}/file_index": [e["data/file_index"] for e in eps],
        f"videos/{VIDEO_KEY}/from_timestamp": [e["video_from"] for e in eps],
        f"videos/{VIDEO_KEY}/to_timestamp": [e["video_to"] for e in eps],
    }
    schema = pa.schema([
        ("episode_index", pa.int64()), ("episode_uuid", pa.string()),
        ("scene", pa.string()), ("scene_label", pa.string()),
        ("tasks", pa.list_(pa.string())), ("subtask_indices", pa.list_(pa.int64())),
        ("length", pa.int64()), ("dataset_from_index", pa.int64()),
        ("dataset_to_index", pa.int64()), ("data/chunk_index", pa.int64()),
        ("data/file_index", pa.int64()), ("meta/episodes/chunk_index", pa.int64()),
        ("meta/episodes/file_index", pa.int64()),
        (f"videos/{VIDEO_KEY}/chunk_index", pa.int64()),
        (f"videos/{VIDEO_KEY}/file_index", pa.int64()),
        (f"videos/{VIDEO_KEY}/from_timestamp", pa.float64()),
        (f"videos/{VIDEO_KEY}/to_timestamp", pa.float64()),
    ])
    rel = EPISODES_PATH.format(chunk_index=0, file_index=0)
    (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols, schema=schema), out_root / rel)
    print(f"  wrote {rel}  ({len(eps)} episode records)")


def write_vocab(out_root, fname, idx_col, val_col, table_dict):
    items = sorted(table_dict.items(), key=lambda kv: kv[1])
    t = pa.table({idx_col: pa.array([i for _, i in items], pa.int64()),
                  val_col: pa.array([k for k, _ in items], pa.string())})
    p = out_root / "meta" / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, p)
    print(f"  wrote meta/{fname}  ({len(items)} rows)")


def write_frame_masks(out_root, eps):
    total = sum(e["length"] for e in eps)
    ep_index, frame_index = [], []
    for e in eps:
        ep_index.extend([e["episode_index"]] * e["length"])
        frame_index.extend(range(e["length"]))
    cols = {"episode_index": pa.array(ep_index, pa.int64()),
            "frame_index": pa.array(frame_index, pa.int64())}
    for flag in MASK_FLAGS:
        vals = None
        # real per-frame flags if a quality table supplied them
        parts = []
        have_real = all(e["quality"] is not None and flag in e["quality"].columns for e in eps)
        if have_real:
            for e in eps:
                parts.append(e["quality"][flag].to_numpy()[:e["length"]])
            vals = np.concatenate(parts).astype(bool)
        else:
            default = flag not in MASK_DEFAULT_FALSE
            vals = np.full(total, default, dtype=bool)
        cols[flag] = pa.array(vals, pa.bool_())
    rel = "meta/frame_masks/part-00000.parquet"
    (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out_root / rel)
    print(f"  wrote {rel}  ({total} frames)")
    return int(np.sum(cols["train_keep"].to_numpy(zero_copy_only=False)))


def write_train_intervals(out_root, eps):
    """One interval per contiguous (episode, subtask) run, over per-episode frames."""
    cols = {"episode_index": [], "start_frame": [], "end_frame": [],
            "task_index": [], "subtask_index": []}
    for e in eps:
        seg = e["subtask_per_frame"]
        i, nseg = 0, len(seg)
        while i < nseg:
            j = i
            while j < nseg and seg[j] == seg[i]:
                j += 1
            cols["episode_index"].append(e["episode_index"])
            cols["start_frame"].append(int(i))
            cols["end_frame"].append(int(j))
            cols["task_index"].append(e["task_index"])
            cols["subtask_index"].append(int(seg[i]))
            i = j
    t = pa.table({k: pa.array(v, pa.int64()) for k, v in cols.items()})
    rel = "meta/train_intervals/part-00000.parquet"
    (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, out_root / rel)
    print(f"  wrote {rel}  ({t.num_rows} intervals)")


def write_info(out_root, eps, n_tasks, n_subtasks, valid_frames, args):
    total_frames = sum(e["length"] for e in eps)
    fps = eps[0]["fps"]

    def state_feat(dim):
        return {"dtype": "float32", "shape": [dim], "names": None}

    def scalar(dtype):
        return {"dtype": dtype, "shape": [1], "names": None}

    features = {
        "observation.state.camera_position": state_feat(3),
        "observation.state.camera_rotation": state_feat(4),
        "observation.state.hand_right_cam": state_feat(63),
        "observation.state.hand_right_cam_rotation": state_feat(84),
        "observation.state.hand_left_cam": state_feat(63),
        "observation.state.hand_left_cam_rotation": state_feat(84),
        "observation.images.main": {
            "dtype": "video", "shape": [OUT, OUT, 3],
            "names": ["height", "width", "rgb"],
            "info": {"video.fps": float(fps), "video.height": OUT, "video.width": OUT,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False},
        },
        "timestamp": scalar("float32"), "frame_index": scalar("int64"),
        "episode_index": scalar("int64"), "index": scalar("int64"),
        "task_index": scalar("int64"), "subtask_index": scalar("int64"),
        "observation.state.camera_intrinsics": state_feat(9),
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "human_hand",
        "fps": float(fps),
        "chunks_size": CHUNKS_SIZE,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        "window_contract": {
            "source_fps": 30, "model_fps": 15, "history_steps": 16,
            "observation_frames": 1, "target_steps": 32, "source_stride": 2,
            "sample_stride": 32, "required_source_frames": 97, "sampled_frames": 49,
            "future_horizon_seconds": 2.1333333333333333,
            "action_representation": "MCP_wrist", "action_reference": "last_observation",
            "use_state": True, "task_sampling": "uniform",
            "source_window_rule": ("All 97 source frames are quality-valid and continuous "
                                   "within one original recording, scene, task and subtask."),
        },
        "dataset_role": "train",
        "total_episodes": len(eps),
        "total_frames": total_frames,
        "total_tasks": n_tasks,
        "total_subtasks": n_subtasks,
        "total_videos": len(eps),
        "total_chunks": (len(eps) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "splits": {"train": f"0:{len(eps)}"},
        "total_valid_frames": int(valid_frames),
        "valid_hours": valid_frames / fps / 3600.0,
        "frame_filter": {
            "format": "source_frame_intervals_v1",
            "intervals_path": "meta/train_intervals",
            "mask_path": "meta/frame_masks",
            "mask_column": "train_keep",
            "source_frames": total_frames,
            "valid_frames": int(valid_frames),
        },
        "video_policy": ("Each clip is one full original recording with its own data/video "
                         "file; per-episode timestamps start at zero."),
        "data_provenance": {
            "note": "How each field in this build was produced (real vs placeholder).",
            "camera_intrinsics": ("explicit 640x640 K (--intrinsics)" if args.intrinsics
                                  else f"calibrated 640x640 pinhole (hfov={args.hfov:g})"),
            "hand_keypoints": ("mediapipe_world_landmarks (metres, wrist-relative)"
                               if args.hands_dir else "placeholder: zeros"),
            "hand_rotations": ("derived per-joint quaternions from bone directions"
                               if args.hands_dir else "placeholder: identity quats"),
            "camera_pose": ("monocular_vo: UP-TO-SCALE and drifting, not metric"
                            if args.pose_dir else "placeholder: identity"),
            "frame_mask_flags": ("validation pipeline"
                                 if args.quality_dir else
                                 "placeholder: all-valid, train_keep=True"),
            "subtasks": ("annotation" if args.anno else "placeholder: single subtask"),
        },
    }
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    print("  wrote meta/info.json")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--hands-dir", default=None)
    ap.add_argument("--pose-dir", default=None)
    ap.add_argument("--quality-dir", default=None)
    ap.add_argument("--anno", default=None)
    ap.add_argument("--hfov", type=float, default=110.0,
                    help="Horizontal FOV of the 640 pinhole view (sets intrinsics)")
    ap.add_argument("--intrinsics", default=None,
                    help="Exact 'fx,fy,cx,cy' for the 640 view: the video is "
                         "undistorted to this K and it is stored verbatim in "
                         "camera_intrinsics. Overrides --hfov (use to byte-match a "
                         "reference's intrinsics).")
    ap.add_argument("--video-dir", default=None,
                    help="Dir of pre-undistorted 640 videos (<stem>.mp4) to reuse "
                         "instead of undistorting from source (batch pipeline).")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"Output not empty: {out}")
    if inp.is_file():
        clips = [inp.with_suffix("")]
    else:
        clips = sorted({p.with_suffix("") for p in inp.glob("*.mp4")})
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        sys.exit(f"No .mp4 clips under {inp}")
    out.mkdir(parents=True, exist_ok=True)

    if args.intrinsics:
        fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
        K640 = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        print(f"640 intrinsics (explicit): fx={fx:.6f} fy={fy:.6f} cx={cx:.6f} cy={cy:.6f}")
    else:
        K640 = new_K(args.hfov)
        print(f"640 pinhole intrinsics (hfov={args.hfov:g}): "
              f"fx={K640[0,0]:.3f} cx={K640[0,2]:.1f}")

    task_table, subtask_table = {}, {}
    eps, offset = [], 0
    for i, stem in enumerate(clips):
        rec = build_clip(stem, i, out, offset, K640, args, task_table, subtask_table)
        if rec:
            eps.append(rec)
            offset += rec["length"]
    if not eps:
        sys.exit("No episodes built.")

    write_episodes(out, eps)
    write_vocab(out, "tasks.parquet", "task_index", "task", task_table)
    write_vocab(out, "subtasks.parquet", "subtask_index", "subtask", subtask_table)
    valid = write_frame_masks(out, eps)
    write_train_intervals(out, eps)
    write_info(out, eps, len(task_table), len(subtask_table), valid, args)

    print("\n" + "=" * 72)
    print(f"DONE. LeRobot v3.0 hand dataset at: {out}")
    print(f"{len(eps)} episode(s), {offset} frames, {len(task_table)} task(s), "
          f"{len(subtask_table)} subtask(s)")
    print("=" * 72)


if __name__ == "__main__":
    main()
