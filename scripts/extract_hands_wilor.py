#!/usr/bin/env python3
"""
GPU hand extractor using WiLoR (via WiLoR-mini) -- the higher-quality replacement
for extract_hands.py (MediaPipe). Same output contract: one .npz per clip with
hand_{right,left}_cam [N,63], hand_{right,left}_cam_rotation [N,84], and present
flags, which build_hand_dataset.py injects via --hands-dir.

It reuses extract_hands.py's detection-view undistortion (896 px / 90 deg on the
SOURCE fisheye -- decoupled from the dataset video) and its bone-direction joint
rotations, so the two extractors are interchangeable and self-consistent; only
the detector/regressor differs.

WiLoR gives a 21-keypoint 3D hand (metric, root-relative) + MANO params per hand,
plus a right/left flag. We store the 21x3 keypoints as the [63] vector and derive
the [84] per-joint quaternions with the shared bone-direction method (not MANO
pose), to match the MediaPipe producer exactly.

REQUIRES (installed on the VM, not locally):
    pip install git+https://github.com/warmshao/WiLoR-mini.git torch torchvision
    MANO_RIGHT.pkl / MANO_LEFT.pkl placed where WiLoR-mini expects (--mano-dir).

USAGE
    python extract_hands_wilor.py --video SOURCE.mp4 --stem <id> --out hands_dir/ \
        --mano-dir /opt/mano [--detect-size 896 --detect-hfov 90]
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv is required")

# reuse the shared helpers from the MediaPipe extractor (same directory)
_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("extract_hands", _here / "extract_hands.py")
_eh = importlib.util.module_from_spec(_spec)
# extract_hands imports mediapipe at module load; guard so WiLoR host need not have it
try:
    _spec.loader.exec_module(_eh)
    FISHEYE_K, FISHEYE_D = _eh.FISHEYE_K, _eh.FISHEYE_D
    detect_K, ffprobe, fps_of = _eh.detect_K, _eh.ffprobe, _eh.fps_of
    joint_rotations = _eh.joint_rotations
except SystemExit:
    # mediapipe missing: inline the pure-numpy bits we need
    import math as _math
    FISHEYE_K = np.array([[707.2808075315367, 0.0, 956.0608949841028],
                          [0.0, 707.6480183518742, 550.7429189294107], [0.0, 0.0, 1.0]])
    FISHEYE_D = np.array([0.011734004280818429, 9.279535348401179e-06,
                          -0.00015706402699172342, -0.0002178771185773861])
    CHILD = {0: 9, 1: 2, 2: 3, 3: 4, 4: 4, 5: 6, 6: 7, 7: 8, 8: 8, 9: 10, 10: 11,
             11: 12, 12: 12, 13: 14, 14: 15, 15: 16, 16: 16, 17: 18, 18: 19, 19: 20, 20: 20}

    def detect_K(size, hfov_deg):
        fp = (size / 2.0) / _math.tan(_math.radians(hfov_deg / 2.0))
        return np.array([[fp, 0, size / 2.0], [0, fp, size / 2.0], [0, 0, 1]])

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

    def _mat_to_quat(R):
        t = R[0, 0] + R[1, 1] + R[2, 2]
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q = [(R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s, 0.25*s]
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0 + R[0,0]-R[1,1]-R[2,2]) * 2
            q = [0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s]
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1]-R[0,0]-R[2,2]) * 2
            q = [(R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s]
        else:
            s = np.sqrt(1.0 + R[2,2]-R[0,0]-R[1,1]) * 2
            q = [(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s]
        return np.array(q, dtype=np.float32)

    def joint_rotations(pts):
        v1, v2 = pts[5] - pts[0], pts[17] - pts[0]
        n = np.cross(v1, v2); nn = np.linalg.norm(n)
        palm_n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
        quats = np.zeros((21, 4), dtype=np.float32)
        for j in range(21):
            bone = pts[CHILD[j]] - pts[j]; bn = np.linalg.norm(bone)
            if bn < 1e-9:
                quats[j] = [0, 0, 0, 1]; continue
            x = bone / bn
            z = palm_n - np.dot(palm_n, x) * x; zn = np.linalg.norm(z)
            z = z / zn if zn > 1e-9 else np.array([0.0, 0.0, 1.0])
            y = np.cross(z, x)
            quats[j] = _mat_to_quat(np.column_stack([x, y, z]))
        return quats.reshape(84)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="SOURCE fisheye clip")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mano-dir", default=None, help="Dir holding MANO_*.pkl")
    ap.add_argument("--detect-size", type=int, default=896)
    ap.add_argument("--detect-hfov", type=float, default=90.0)
    args = ap.parse_args()

    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)

    if args.mano_dir:
        # WiLoR-mini resolves MANO from this env dir if its default is absent
        os.environ.setdefault("MANO_DIR", args.mano_dir)
        os.environ.setdefault("SMPLX_MODELS", args.mano_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    print(f"WiLoR on {device} ({dtype})", flush=True)

    stream = ffprobe(args.video)
    sw, sh = int(stream["width"]), int(stream["height"])
    fps = fps_of(stream)
    size = args.detect_size
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        FISHEYE_K, FISHEYE_D, np.eye(3), detect_K(size, args.detect_hfov),
        (size, size), cv2.CV_16SC2)

    pos = {"right": [], "left": []}
    rot = {"right": [], "left": []}
    present = {"right": [], "left": []}

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(args.video), "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    nbytes = sw * sh * 3
    idx = 0
    while True:
        raw = dec.stdout.read(nbytes)
        if len(raw) < nbytes:
            break
        bgr = np.frombuffer(raw, np.uint8).reshape(sh, sw, 3)
        view = cv2.remap(bgr, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)

        frame = {"right": None, "left": None}
        try:
            outs = pipe.predict(rgb)
        except Exception as e:                      # never let one frame kill the run
            outs = []
            if idx % 1500 == 0:
                print(f"    [warn] frame {idx}: {e}", flush=True)
        for o in outs or []:
            side = "right" if float(o.get("is_right", 1.0)) >= 0.5 else "left"
            kp = np.asarray(o["wilor_preds"]["pred_keypoints_3d"], dtype=np.float32).reshape(-1, 3)
            if kp.shape[0] >= 21:
                frame[side] = kp[:21]

        for s in ("right", "left"):
            if frame[s] is not None:
                pos[s].append(frame[s].reshape(63)); rot[s].append(joint_rotations(frame[s]))
                present[s].append(True)
            else:
                pos[s].append(np.zeros(63, np.float32))
                q = np.zeros((21, 4), np.float32); q[:, 3] = 1.0
                rot[s].append(q.reshape(84)); present[s].append(False)
        idx += 1
        if idx % 900 == 0:
            print(f"    wilor {idx} frames", flush=True)
    dec.wait()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"{args.stem}.npz",
        hand_right_cam=np.array(pos["right"], np.float32),
        hand_right_cam_rotation=np.array(rot["right"], np.float32),
        hand_left_cam=np.array(pos["left"], np.float32),
        hand_left_cam_rotation=np.array(rot["left"], np.float32),
        hand_right_present=np.array(present["right"], bool),
        hand_left_present=np.array(present["left"], bool),
    )
    r, l = int(np.sum(present["right"])), int(np.sum(present["left"]))
    either = int(np.sum(np.array(present["right"]) | np.array(present["left"])))
    print(f"{args.stem}: {idx} frames | right {r} | left {l} | either {either} "
          f"({100*either/max(idx,1):.1f}%) -> {out / (args.stem + '.npz')}")


if __name__ == "__main__":
    main()
