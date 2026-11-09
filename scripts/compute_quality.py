#!/usr/bin/env python3
"""
Compute per-frame quality flags (the --quality-dir input for build_hand_dataset.py)
from the extracted hand keypoints and camera trajectory.

Output: one parquet per clip with the frame_masks columns, so train_keep then
reflects real hand presence + kinematic plausibility instead of all-True.

WHAT IS COMPUTED vs. PLACEHOLDER
--------------------------------
Computed here from the data (thresholds are DEFAULTS -- reconcile with the
reference validation pipeline):
  is_hand_coordinate_range_valid  keypoints within +/- --coord-range m of wrist
  is_hand_bone_length_valid       finger bone lengths within [--bl-min,--bl-max] m
  is_hand_bmc_valid               proxy: bone-length AND coordinate-range valid
  is_hand_velocity_valid          wrist speed < --v-max  m/s
  is_hand_acceleration_valid      wrist accel < --a-max  m/s^2
  is_hand_jerk_valid              wrist jerk  < --j-max  m/s^3
  is_camera_rotation_valid        camera angular speed < --cam-w  deg/s
A hand-based flag is False on frames with no detected hand; where >=1 hand is
present it must pass for every present hand.

Placeholder (need the team's VLM + prompts -- left True, override with --vlm-dir):
  is_vlm_video_quality_valid, is_vlm_informative_valid, is_vlm_execution_valid,
  is_vlm_hand_position_valid, is_vlm_hand_left_right_valid, is_vlm_hand_shape_valid

Derived:
  keep_all_quality = AND of every is_* flag above
  train_keep       = keep_all_quality
  validation_50h, validation_online, train100_keep = False (dataset-split flags)

USAGE
    python compute_quality.py --hands hands_dir/<stem>.npz \
        --pose pose_dir/<stem>.tum --stem <id> --out quality_dir/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow is required")

# MediaPipe finger bones (parent, child) for bone-length checks
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),
         (11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

VLM_FLAGS = ["is_vlm_video_quality_valid", "is_vlm_informative_valid",
             "is_vlm_execution_valid", "is_vlm_hand_position_valid",
             "is_vlm_hand_left_right_valid", "is_vlm_hand_shape_valid"]


def quat_angular_speed(quat, fps):
    """Angular speed (deg/s) between consecutive camera-to-world quaternions."""
    q = quat / np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-9, None)
    dot = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(-1, 1)
    ang = 2 * np.arccos(dot)  # rad between frames
    speed = np.degrees(ang) * fps
    return np.concatenate([[0.0], speed])


def hand_checks(pos, present, args):
    """Per-frame (coord_range, bone_length, velocity, accel, jerk) for one hand."""
    n = len(present)
    coord = np.zeros(n, bool); bone = np.zeros(n, bool)
    vel = np.zeros(n, bool); acc = np.zeros(n, bool); jerk = np.zeros(n, bool)
    p = pos.reshape(n, 21, 3)
    wrist = p[:, 0, :]
    # static checks
    for i in range(n):
        if not present[i]:
            continue
        rel = p[i] - wrist[i]
        coord[i] = np.all(np.abs(rel) <= args.coord_range)
        bl = np.array([np.linalg.norm(p[i, c] - p[i, a]) for a, c in BONES])
        bone[i] = np.all((bl >= args.bl_min) & (bl <= args.bl_max))
    # kinematics along present runs
    dt = 1.0 / args.fps
    v = np.full(n, np.nan); a = np.full(n, np.nan); j = np.full(n, np.nan)
    for i in range(1, n):
        if present[i] and present[i-1]:
            v[i] = np.linalg.norm(wrist[i] - wrist[i-1]) / dt
    for i in range(1, n):
        if not np.isnan(v[i]) and not np.isnan(v[i-1]):
            a[i] = abs(v[i] - v[i-1]) / dt
    for i in range(1, n):
        if not np.isnan(a[i]) and not np.isnan(a[i-1]):
            j[i] = abs(a[i] - a[i-1]) / dt
    for i in range(n):
        vel[i]  = present[i] and (np.isnan(v[i]) or v[i] < args.v_max)
        acc[i]  = present[i] and (np.isnan(a[i]) or a[i] < args.a_max)
        jerk[i] = present[i] and (np.isnan(j[i]) or j[i] < args.j_max)
    return {"coord": coord, "bone": bone, "vel": vel, "acc": acc, "jerk": jerk}


def combine(rp, lp, rc, lc, key):
    """Frame valid if >=1 hand present and every present hand passes `key`."""
    n = len(rp)
    out = np.zeros(n, bool)
    for i in range(n):
        present = []
        if rp[i]: present.append(rc[key][i])
        if lp[i]: present.append(lc[key][i])
        out[i] = len(present) > 0 and all(present)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", required=True)
    ap.add_argument("--pose", default=None)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--coord-range", type=float, default=0.3)   # m from wrist
    ap.add_argument("--bl-min", type=float, default=0.003)      # m
    ap.add_argument("--bl-max", type=float, default=0.13)       # m (hand bones up to ~13cm)
    ap.add_argument("--v-max", type=float, default=3.0)         # m/s
    ap.add_argument("--a-max", type=float, default=60.0)        # m/s^2
    ap.add_argument("--j-max", type=float, default=2000.0)      # m/s^3
    ap.add_argument("--cam-w", type=float, default=200.0)       # deg/s
    args = ap.parse_args()

    z = np.load(args.hands)
    rp, lp = z["hand_right_present"], z["hand_left_present"]
    n = len(rp)
    rc = hand_checks(z["hand_right_cam"], rp, args)
    lc = hand_checks(z["hand_left_cam"], lp, args)

    coord = combine(rp, lp, rc, lc, "coord")
    bone = combine(rp, lp, rc, lc, "bone")
    vel = combine(rp, lp, rc, lc, "vel")
    acc = combine(rp, lp, rc, lc, "acc")
    jerk = combine(rp, lp, rc, lc, "jerk")
    bmc = bone & coord

    if args.pose and Path(args.pose).exists():
        rows = [[float(x) for x in ln.split()] for ln in open(args.pose)
                if ln.strip() and not ln.startswith("#")]
        quat = np.array(rows)[:, 4:8]
        cam_ok = quat_angular_speed(quat, args.fps)[:n] < args.cam_w
        if len(cam_ok) < n:
            cam_ok = np.concatenate([cam_ok, np.ones(n - len(cam_ok), bool)])
    else:
        cam_ok = np.ones(n, bool)

    cols = {
        "episode_index": pa.array(np.zeros(n, np.int64)),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "is_hand_bmc_valid": pa.array(bmc),
        "is_hand_velocity_valid": pa.array(vel),
        "is_hand_acceleration_valid": pa.array(acc),
        "is_hand_jerk_valid": pa.array(jerk),
        "is_hand_coordinate_range_valid": pa.array(coord),
        "is_hand_bone_length_valid": pa.array(bone),
        "is_camera_rotation_valid": pa.array(cam_ok),
    }
    for f in VLM_FLAGS:
        cols[f] = pa.array(np.ones(n, bool))            # placeholder (no VLM)

    hand_quality = bmc & vel & acc & jerk & coord & bone
    keep_all = hand_quality & cam_ok & np.ones(n, bool)  # VLM all True
    cols["keep_all_quality"] = pa.array(keep_all)
    cols["validation_50h"] = pa.array(np.zeros(n, bool))
    cols["validation_online"] = pa.array(np.zeros(n, bool))
    cols["train_keep"] = pa.array(keep_all)
    cols["train100_keep"] = pa.array(np.zeros(n, bool))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out / f"{args.stem}.parquet")
    print(f"{args.stem}: {n} frames | hand-present {int((rp|lp).sum())} | "
          f"train_keep {int(keep_all.sum())} ({100*keep_all.mean():.1f}%) "
          f"-> {out / (args.stem + '.parquet')}")


if __name__ == "__main__":
    main()
