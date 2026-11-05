#!/usr/bin/env python3
"""
Extract per-frame 3D hand keypoints + per-joint rotations for both hands from an
egocentric clip, producing the --hands-dir input that build_hand_dataset.py uses.

Output (one .npz per clip, keyed by clip stem):
    hand_right_cam            [N, 63]   21 MediaPipe world landmarks (x,y,z), metres,
    hand_left_cam             [N, 63]   wrist-relative (world-landmark origin).
    hand_right_cam_rotation   [N, 84]   21 per-joint quaternions (x,y,z,w).
    hand_left_cam_rotation    [N, 84]
    hand_right_present         [N]      bool, hand detected that frame
    hand_left_present          [N]      bool

DETECTION VIEW IS DECOUPLED FROM THE DATASET VIDEO
--------------------------------------------------
MediaPipe `hand_world_landmarks` are a metric, wrist-relative hand model --
independent of the image projection and resolution. So detection runs on its own
undistorted view of the SOURCE fisheye, tuned purely to maximise detection, while
build_hand_dataset.py renders the dataset video at whatever FOV it wants; the
stored keypoints are valid for either.

The defaults (896 px, 90 deg FOV) were chosen empirically on the garment-worker
footage: a tighter, higher-resolution pinhole makes hands larger and less
distorted, which roughly doubled MediaPipe's hit rate versus a wide 640/110 view.
Detection on egocentric factory video is still limited -- hands are often out of
frame -- so expect a minority of frames to carry hands.

CONVENTIONS (reconcile with the reference pipeline if it differs)
-----------------------------------------------------------------
  * Source: MediaPipe Tasks HandLandmarker (float16), num_hands=2, VIDEO mode.
  * Positions: `hand_world_landmarks` (metres, wrist-relative) -- NOT full
    camera-frame metric placement (that needs per-hand depth from camera pose).
  * Handedness: MediaPipe's "Left"/"Right" taken as the *_cam side; MediaPipe
    assumes a mirrored (selfie) view, so pass --swap-handedness if the reference
    uses the opposite assignment.
  * Per-joint rotation: local frame per landmark (x = bone direction, z = palm
    normal, y = z x x) -> quaternion. Undetected frames: zeros / identity.

USAGE
    python extract_hands.py --video SOURCE_fisheye.mp4 --stem MTL-BPT01426-001 \
        --model models/hand_landmarker.task --out hands_dir/ \
        [--detect-size 896] [--detect-hfov 90] [--min-conf 0.2]
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv is required:  pip install opencv-python-headless")
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    sys.exit("mediapipe is required (python3.12): pip install mediapipe")

# EgoCapture-1 fisheye calibration (calib_all_s30-camchain.yaml)
FISHEYE_K = np.array([[707.2808075315367, 0.0, 956.0608949841028],
                      [0.0, 707.6480183518742, 550.7429189294107],
                      [0.0, 0.0, 1.0]])
FISHEYE_D = np.array([0.011734004280818429, 9.279535348401179e-06,
                      -0.00015706402699172342, -0.0002178771185773861])

# MediaPipe 21-landmark parent/child along each finger (for bone directions).
CHILD = {0: 9, 1: 2, 2: 3, 3: 4, 4: 4, 5: 6, 6: 7, 7: 8, 8: 8,
         9: 10, 10: 11, 11: 12, 12: 12, 13: 14, 14: 15, 15: 16, 16: 16,
         17: 18, 18: 19, 19: 20, 20: 20}


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


def detect_K(size, hfov_deg):
    fp = (size / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    return np.array([[fp, 0, size / 2.0], [0, fp, size / 2.0], [0, 0, 1]])


def mat_to_quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0 + R[0,0]-R[1,1]-R[2,2]) * 2
        w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1]-R[0,0]-R[2,2]) * 2
        w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = np.sqrt(1.0 + R[2,2]-R[0,0]-R[1,1]) * 2
        w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    return np.array([x, y, z, w], dtype=np.float32)


def joint_rotations(pts):
    v1, v2 = pts[5] - pts[0], pts[17] - pts[0]
    n = np.cross(v1, v2)
    nn = np.linalg.norm(n)
    palm_n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
    quats = np.zeros((21, 4), dtype=np.float32)
    for j in range(21):
        bone = pts[CHILD[j]] - pts[j]
        bn = np.linalg.norm(bone)
        if bn < 1e-9:
            quats[j] = [0, 0, 0, 1]; continue
        x = bone / bn
        z = palm_n - np.dot(palm_n, x) * x
        zn = np.linalg.norm(z)
        z = z / zn if zn > 1e-9 else np.array([0.0, 0.0, 1.0])
        y = np.cross(z, x)
        quats[j] = mat_to_quat(np.column_stack([x, y, z]))
    return quats.reshape(84)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="SOURCE fisheye clip (not the 640 video)")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--model", default="models/hand_landmarker.task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detect-size", type=int, default=896,
                    help="Square resolution of the detection view (empirical best 896)")
    ap.add_argument("--detect-hfov", type=float, default=90.0,
                    help="FOV of the detection view in degrees (empirical best ~90)")
    ap.add_argument("--swap-handedness", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.2)
    args = ap.parse_args()

    video = Path(args.video)
    stream = ffprobe(video)
    sw, sh = int(stream["width"]), int(stream["height"])
    fps = fps_of(stream)

    size = args.detect_size
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        FISHEYE_K, FISHEYE_D, np.eye(3), detect_K(size, args.detect_hfov),
        (size, size), cv2.CV_16SC2)

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=args.min_conf,
        min_hand_presence_confidence=args.min_conf,
        min_tracking_confidence=args.min_conf)
    landmarker = vision.HandLandmarker.create_from_options(opts)

    sides = {"right": None, "left": None}
    pos = {s: [] for s in sides}; rot = {s: [] for s in sides}; present = {s: [] for s in sides}

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
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
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = landmarker.detect_for_video(image, int(idx * 1000 / fps))

        frame = {s: None for s in sides}
        if res.handedness:
            for k, hd in enumerate(res.handedness):
                cat = hd[0].category_name.lower()
                if args.swap_handedness:
                    cat = "left" if cat == "right" else "right"
                if cat in frame and k < len(res.hand_world_landmarks):
                    wl = res.hand_world_landmarks[k]
                    frame[cat] = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)

        for s in sides:
            if frame[s] is not None:
                pos[s].append(frame[s].reshape(63)); rot[s].append(joint_rotations(frame[s]))
                present[s].append(True)
            else:
                pos[s].append(np.zeros(63, np.float32))
                q = np.zeros((21, 4), np.float32); q[:, 3] = 1.0
                rot[s].append(q.reshape(84)); present[s].append(False)
        idx += 1
        if idx % 900 == 0:
            print(f"    hands {idx} frames", flush=True)
    dec.wait(); landmarker.close()

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
