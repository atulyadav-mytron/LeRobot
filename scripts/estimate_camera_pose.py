#!/usr/bin/env python3
"""
Estimate a per-frame camera trajectory (the --pose-dir input for
build_hand_dataset.py) with monocular visual odometry on the undistorted video.

Output: one TUM file per clip -- "<t> <tx> <ty> <tz> <qx> <qy> <qz> <qw>" per
frame -- that build_hand_dataset.py reads into observation.state.camera_position
and observation.state.camera_rotation.

HONEST LIMITATIONS (read before trusting the numbers)
-----------------------------------------------------
This is ORB + essential-matrix monocular VO, not full VIO/SLAM:
  * Scale is NOT metric. Monocular VO has no metric scale. Each step's
    translation direction is recoverPose's unit vector, scaled by the de-rotated
    median parallax so step length tracks real motion (self-consistent up-to-
    scale) instead of a constant. Absolute metres still need the team's VIO.
  * Rotation-gating: frames whose de-rotated parallax is below --px-min (static
    or near-pure rotation) take rotation only, no translation -- this removes the
    main drift source on egocentric footage. Rotation is still applied.
  * It still drifts over long runs; no loop closure or global bundle adjustment.
  * Frames with too few matches/inliers carry the previous pose unchanged.
It exists so observation.state.camera_* carries a real motion estimate rather
than zeros. Replace it with the team's calibrated VIO (same --pose-dir contract,
camchain + IMU noise model already in the calibration deliverable) for metric,
drift-corrected poses. Poses are camera-to-world; mark metric-vs-up-to-scale in
the dataset's info.json.

USAGE
    python estimate_camera_pose.py --video undistorted.mp4 --stem <id> \
        --fx 224.066 --cx 320 --cy 320 --out pose_dir/
    # fx/cx/cy must match the video's intrinsics (build_hand_dataset prints them)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv is required:  pip install opencv-python-headless")


def ffprobe(path):
    import json
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", str(path)], text=True)
    return json.loads(out)["streams"][0]


def fps_of(s):
    num, _, den = s.get("r_frame_rate", "30/1").partition("/")
    try:
        return int(num) / int(den)
    except (ValueError, ZeroDivisionError):
        return 30.0


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0 + R[0,0]-R[1,1]-R[2,2]) * 2
        w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1]-R[0,0]-R[2,2]) * 2
        w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = np.sqrt(1.0 + R[2,2]-R[0,0]-R[1,1]) * 2
        w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    return np.array([x, y, z, w], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--min-inliers", type=int, default=30,
                    help="Below this many pose inliers, hold translation (rotation only)")
    ap.add_argument("--px-min", type=float, default=0.7,
                    help="De-rotated median parallax (px) below which a frame is "
                         "treated as rotation-only (no translation step)")
    ap.add_argument("--px-cap", type=float, default=60.0,
                    help="Parallax (px) is capped here before scaling the step")
    ap.add_argument("--px-fail", type=float, default=300.0,
                    help="Above this parallax the pose estimate is untrusted -> "
                         "rotation-only (likely a bad R from blur/outliers)")
    args = ap.parse_args()

    stream = ffprobe(args.video)
    w, h = int(stream["width"]), int(stream["height"])
    fps = fps_of(stream)
    fy = args.fy if args.fy else args.fx
    K = np.array([[args.fx, 0, args.cx], [0, fy, args.cy], [0, 0, 1]], dtype=np.float64)
    Kinv = np.linalg.inv(K)

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # world->camera pose (cur_R, cur_t); frame 0 is the world origin
    cur_R = np.eye(3)
    cur_t = np.zeros((3, 1))
    rows = []

    def emit(i, R, t):
        cam_center = (-R.T @ t).reshape(3)            # camera-to-world position
        q = mat_to_quat(R.T)                           # camera-to-world rotation
        rows.append((i / fps, *cam_center.tolist(), *q.tolist()))

    emit(0, cur_R, cur_t)

    def derotated_parallax(p0, p1, R):
        """Median residual pixel motion after removing R's rotational flow.

        ~0 for pure rotation / static; grows with translation parallax. Used both
        to gate translation and to scale each step, so step length tracks real
        motion instead of a meaningless constant (recoverPose's unit vector)."""
        ones = np.ones((len(p0), 1))
        rays = (Kinv @ np.hstack([p0, ones]).T)          # 3xN
        rot = R @ rays
        proj = K @ rot
        p0rot = (proj[:2] / proj[2]).T                    # Nx2
        return float(np.median(np.linalg.norm(p1 - p0rot, axis=1)))

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(args.video), "-f", "rawvideo",
         "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    nbytes = w * h
    prev_kp = prev_des = None
    idx = 0
    held = 0      # frames with no pose at all
    rot_only = 0  # frames gated to rotation-only (static / pure rotation)
    moving = 0
    while True:
        raw = dec.stdout.read(nbytes)
        if len(raw) < nbytes:
            break
        gray = np.frombuffer(raw, np.uint8).reshape(h, w)
        kp, des = orb.detectAndCompute(gray, None)

        if idx > 0:
            R_rel, t_step, ok = np.eye(3), np.zeros((3, 1)), False
            if des is not None and prev_des is not None and len(kp) > 8 and len(prev_kp) > 8:
                matches = bf.knnMatch(prev_des, des, k=2)
                good = [m for m, n in (mn for mn in matches if len(mn) == 2)
                        if m.distance < 0.75 * n.distance]
                if len(good) >= args.min_matches:
                    p0 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                    p1 = np.float32([kp[m.trainIdx].pt for m in good])
                    E, mask = cv2.findEssentialMat(p1, p0, K, cv2.RANSAC, 0.999, 1.0)
                    if E is not None and E.shape == (3, 3):
                        n_in, R_rel, t_dir, pmask = cv2.recoverPose(E, p1, p0, K, mask=mask)
                        ok = True
                        inl = (pmask.ravel() > 0)
                        if inl.sum() >= args.min_inliers:
                            px = derotated_parallax(p0[inl], p1[inl], R_rel)
                            if args.px_min <= px <= args.px_fail:
                                # direction from recoverPose; magnitude = parallax,
                                # capped and normalised by focal length (angular ~
                                # translation/depth), so steps are bounded & comparable
                                scale = min(px, args.px_cap) / args.fx
                                t_step = t_dir * scale
                                moving += 1
                            else:
                                rot_only += 1   # static/rotation (low px) or bad pose (high px)
                        else:
                            rot_only += 1
            if not ok:
                held += 1   # no usable pose: carry previous pose unchanged
            # chain world->cam:  x_i = R_rel x_{i-1} + t_step
            cur_t = cur_t + cur_R @ t_step
            cur_R = R_rel @ cur_R
            emit(idx, cur_R, cur_t)

        prev_kp, prev_des = kp, des
        idx += 1
        if idx % 900 == 0:
            print(f"    VO {idx} frames", flush=True)
    dec.wait()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{args.stem}.tum"
    with p.open("w") as f:
        f.write("# monocular VO (up-to-scale, rotation-gated, parallax-scaled) "
                "-- t tx ty tz qx qy qz qw\n")
        for r in rows:
            f.write(" ".join(f"{v:.6f}" for v in r) + "\n")
    pos = np.array([r[1:4] for r in rows])
    path_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) > 1 else 0.0
    print(f"{args.stem}: {idx} frames | moving {moving} | rotation-only {rot_only} | "
          f"no-pose {held} | path-len {path_len:.1f} (up-to-scale) -> {p}")


if __name__ == "__main__":
    main()
