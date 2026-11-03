#!/usr/bin/env python3
"""
Ingest raw EgoCapture-1 clips (the ots-egocentric-inventory bucket layout) into
the per-segment folder layout that build_lerobot_dataset.py consumes, then hand
off to that builder to emit the LeRobot v3.0 dataset.

Per clip the bucket gives four files that share a stem, e.g. MTL-BPT01426-001:
    <stem>.mp4   monocular fisheye video, HEVC 1920x1080 @ 30 fps
    <stem>.txt   IMU, JSON-Lines: {"t_us":int, "acc":[x,y,z], "gyro":[x,y,z]}
    <stem>.json  metadata (fps, resolution, codec, task, ...)
    <stem>.aac   audio (not used by the dataset)

What this script does for each clip:
  1. Undistorts the 185 deg fisheye to a pinhole view using the team's Kalibr
     calibration (equidistant model; same K/D/FOV as 5_scripts/
     undistort_to_pinhole.py). Output is the SLAM-ready pinhole image that the
     eventual VIO/SLAM poses will be expressed in.
  2. Converts the IMU .txt (SI units already: m/s^2, rad/s) to imu.csv.
  3. Writes video_frames_timestamp.csv from the IMU microsecond clock (the IMU
     is 1:1 with video frames at 30 Hz).
  4. Optionally copies a per-clip trajectory (if --trajectory-dir is given) into
     trajectory.csv so observation.state / action carry real poses. Without it,
     the builder zero-fills state/action -- that is the one external input this
     pipeline cannot produce on its own (see --trajectory-dir).

Then it runs build_lerobot_dataset.py on the staged tree.

USAGE
    python ingest_egocapture.py \
        --input   /path/to/clips_dir_or_single_clip \
        --staging /path/to/staging \
        --output  /path/to/lerobot_dataset \
        [--trajectory-dir /path/to/poses]      # <stem>.csv|.tum per clip
        [--no-undistort]                       # keep raw fisheye instead
        [--limit N]                            # only the first N clips

TRAJECTORY INPUT (the missing piece)
    Poses are NOT in the bucket. Produce them with monocular(-inertial) SLAM on
    the undistorted pinhole video (see --emit-slam-inputs, which writes a ROS-
    bag-ready frames/ + the pinhole intrinsics), then drop one file per clip in
    --trajectory-dir. Accepted formats (auto-detected):
      * TUM   : "t tx ty tz qx qy qz qw" per line (whitespace-separated)
      * CSV   : a header with timestamp + px,py,pz + qx,qy,qz,qw (any of the
                name variants build_lerobot_dataset.load_trajectory accepts)
    Mark in --output's info.json whether the poses are metric or up-to-scale.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv is required:  pip install opencv-python-headless")

# --- fisheye calibration (calib_all_s30-camchain.yaml, 1920x1080) -----------
# Identical to 5_scripts/undistort_to_pinhole.py so the pinhole view matches the
# team's deliverable exactly. Override by parsing a camchain with --calib later.
FISHEYE_K = np.array([[707.2808075315367, 0.0, 956.0608949841028],
                      [0.0, 707.6480183518742, 550.7429189294107],
                      [0.0, 0.0, 1.0]])
FISHEYE_D = np.array([0.011734004280818429, 9.279535348401179e-06,
                      -0.00015706402699172342, -0.0002178771185773861])
SRC_W, SRC_H = 1920, 1080
OUT_W, OUT_H = 1920, 1080
HFOV_DEG = 90.0


def pinhole_K(out_w, out_h, hfov_deg):
    import math
    fp = (out_w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    return np.array([[fp, 0.0, out_w / 2.0],
                     [0.0, fp, out_h / 2.0],
                     [0.0, 0.0, 1.0]]), fp


def ffprobe_stream(path: Path) -> dict:
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


def undistort_video(src: Path, dst: Path, fps: float):
    """Fisheye -> pinhole, streamed through ffmpeg (decode) and ffmpeg (encode).

    cv2 only does the remap; all video I/O is ffmpeg, so this does not depend on
    opencv being built with a working video backend.
    """
    newK, _ = pinhole_K(OUT_W, OUT_H, HFOV_DEG)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        FISHEYE_K, FISHEYE_D, np.eye(3), newK, (OUT_W, OUT_H), cv2.CV_16SC2)

    dst.parent.mkdir(parents=True, exist_ok=True)
    frame_bytes = SRC_W * SRC_H * 3

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps:g}", "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        stdin=subprocess.PIPE)

    n = 0
    try:
        while True:
            raw = dec.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(SRC_H, SRC_W, 3)
            out = cv2.remap(frame, map1, map2, cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT)
            enc.stdin.write(out.tobytes())
            n += 1
            if n % 600 == 0:
                print(f"    undistort {n} frames", flush=True)
    finally:
        if enc.stdin:
            enc.stdin.close()
        dec.wait()
        enc.wait()
    print(f"    undistorted {n} frames -> {dst.name}", flush=True)
    return n


def copy_video(src: Path, dst: Path):
    """Keep the raw fisheye, just transcode HEVC -> h264 (no geometry change)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


def load_imu_txt(path: Path):
    """Parse JSON-Lines IMU; returns (t_us, acc[N,3], gyro[N,3])."""
    t, acc, gyro = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t.append(int(r["t_us"]))
        acc.append([float(x) for x in r["acc"]])
        gyro.append([float(x) for x in r["gyro"]])
    return (np.asarray(t, dtype=np.int64),
            np.asarray(acc, dtype=np.float64),
            np.asarray(gyro, dtype=np.float64))


def write_imu_csv(path: Path, t_us, acc, gyro):
    # SI units straight from the device: acc m/s^2, gyro rad/s. No rescaling.
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "acc_x", "acc_y", "acc_z",
                    "gyro_x", "gyro_y", "gyro_z"])
        for i in range(len(t_us)):
            w.writerow([int(t_us[i] * 1000),
                        acc[i, 0], acc[i, 1], acc[i, 2],
                        gyro[i, 0], gyro[i, 1], gyro[i, 2]])


def write_frame_timestamps(path: Path, t_us, n_frames):
    """One row per video frame. IMU is 1:1 with frames; if counts differ, the
    builder resamples, so we write whichever we have and let it align."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence_num", "timestamp_ns"])
        m = min(len(t_us), n_frames) if n_frames else len(t_us)
        for i in range(m):
            w.writerow([i + 1, int(t_us[i] * 1000)])


def convert_trajectory(src: Path, dst: Path):
    """Normalise a SLAM trajectory (TUM or CSV) to columns the builder reads."""
    text = src.read_text(encoding="utf-8").splitlines()
    rows = []
    # TUM: 8 whitespace-separated numbers, '#'-comments. Detect by first data line.
    is_tum = False
    for line in text:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) == 8:
            try:
                [float(p) for p in parts]
                is_tum = True
            except ValueError:
                is_tum = False
        break

    if is_tum:
        for line in text:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            t, tx, ty, tz, qx, qy, qz, qw = s.split()
            # TUM timestamps are seconds -> ns
            rows.append([int(round(float(t) * 1e9)), tx, ty, tz, qx, qy, qz, qw])
    else:
        import pandas as pd
        df = pd.read_csv(src)
        # reuse the builder's tolerant column picker
        import importlib.util
        lb = _load_builder()
        cols = {
            "t": lb.pick_column(df, ["timestamp_ns", "#timestamp", "timestamp", "t", "time"], label="t"),
            "px": lb.pick_column(df, ["px", "tx", "p_RS_R_x", "x"], label="px"),
            "py": lb.pick_column(df, ["py", "ty", "p_RS_R_y", "y"], label="py"),
            "pz": lb.pick_column(df, ["pz", "tz", "p_RS_R_z", "z"], label="pz"),
            "qx": lb.pick_column(df, ["qx", "q_RS_x"], label="qx"),
            "qy": lb.pick_column(df, ["qy", "q_RS_y"], label="qy"),
            "qz": lb.pick_column(df, ["qz", "q_RS_z"], label="qz"),
            "qw": lb.pick_column(df, ["qw", "q_RS_w"], label="qw"),
        }
        for _, r in df.iterrows():
            rows.append([r[cols["t"]], r[cols["px"]], r[cols["py"]], r[cols["pz"]],
                         r[cols["qx"]], r[cols["qy"]], r[cols["qz"]], r[cols["qw"]]])

    with dst.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "px", "py", "pz", "qx", "qy", "qz", "qw"])
        w.writerows(rows)
    print(f"    trajectory: {len(rows)} poses -> {dst.name}")


def _load_builder():
    import importlib.util
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "build_lerobot_dataset", here / "build_lerobot_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_clips(input_path: Path):
    """Return sorted list of clip stems (Path without suffix) that have a .mp4."""
    if input_path.is_file():
        return [input_path.with_suffix("")]
    stems = sorted({p.with_suffix("") for p in input_path.glob("*.mp4")})
    return stems


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="Directory of EgoCapture clips, or one clip's .mp4/stem")
    ap.add_argument("--staging", required=True, help="Where to stage segments/")
    ap.add_argument("--output", required=True, help="LeRobot dataset root to create")
    ap.add_argument("--trajectory-dir", default=None,
                    help="Dir with <stem>.csv/.tum SLAM poses (one per clip)")
    ap.add_argument("--no-undistort", action="store_true",
                    help="Keep the raw fisheye (still transcoded to h264)")
    ap.add_argument("--limit", type=int, default=0, help="Only the first N clips")
    ap.add_argument("--task-label", default="garment worker activity recording")
    ap.add_argument("--robot-type", default="egocentric_camera")
    ap.add_argument("--skip-build", action="store_true",
                    help="Only stage; do not run build_lerobot_dataset.py")
    args = ap.parse_args()

    inp = Path(args.input)
    staging = Path(args.staging)
    out = Path(args.output)
    traj_dir = Path(args.trajectory_dir) if args.trajectory_dir else None

    clips = find_clips(inp)
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        sys.exit(f"No .mp4 clips found under {inp}")

    seg_root = staging / "segments"
    seg_root.mkdir(parents=True, exist_ok=True)
    print(f"Staging {len(clips)} clip(s) -> {seg_root}")

    for stem in clips:
        name = stem.name
        mp4 = stem.with_suffix(".mp4")
        txt = stem.with_suffix(".txt")
        if not mp4.exists():
            print(f"  [skip] {name}: no .mp4")
            continue
        print(f"\n--- {name} ---", flush=True)

        seg = seg_root / name
        seg.mkdir(parents=True, exist_ok=True)

        stream = ffprobe_stream(mp4)
        fps = fps_of(stream)
        n_frames = int(stream["nb_frames"]) if stream.get("nb_frames") else 0
        print(f"  video: {stream['width']}x{stream['height']} @ {fps:g} fps, "
              f"{n_frames} frames", flush=True)

        # 1. video (undistort to pinhole, or keep fisheye)
        if args.no_undistort:
            copy_video(mp4, seg / "cam_left.mp4")
        else:
            undistort_video(mp4, seg / "cam_left.mp4", fps)

        # 2/3. IMU + frame timestamps
        if txt.exists():
            t_us, acc, gyro = load_imu_txt(txt)
            print(f"  imu: {len(t_us)} samples", flush=True)
            write_imu_csv(seg / "imu.csv", t_us, acc, gyro)
            write_frame_timestamps(seg / "video_frames_timestamp.csv", t_us, n_frames)
        else:
            print(f"  [!] {name}: no .txt IMU", flush=True)

        # 4. trajectory (optional)
        if traj_dir:
            cand = None
            for ext in (".csv", ".tum", ".txt"):
                if (traj_dir / f"{name}{ext}").exists():
                    cand = traj_dir / f"{name}{ext}"
                    break
            if cand:
                convert_trajectory(cand, seg / "trajectory.csv")
            else:
                print(f"  [!] {name}: no trajectory in {traj_dir}; "
                      "state/action will be zero-filled", flush=True)

    if args.skip_build:
        print(f"\nStaged only (--skip-build). Segments at {seg_root}")
        return

    # hand off to the v3.0 builder
    builder = Path(__file__).resolve().parent / "build_lerobot_dataset.py"
    cmd = [sys.executable, str(builder),
           "--input", str(staging), "--output", str(out),
           "--main-camera", "left",
           "--task-label", args.task_label,
           "--robot-type", args.robot_type,
           "--copy-video"]  # video is already h264; don't re-encode a 2nd time
    print("\n" + "=" * 72)
    print("Running builder:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
