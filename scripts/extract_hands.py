#!/usr/bin/env python3
"""
Extract per-frame 3D hand keypoints + per-joint rotations for both hands from an
undistorted egocentric video, producing the --hands-dir input that
build_hand_dataset.py consumes.

Output (one .npz per clip, keyed by clip stem):
    hand_right_cam            [N, 63]   21 MediaPipe world landmarks (x,y,z), metres,
    hand_left_cam             [N, 63]   wrist-relative (world-landmark origin).
    hand_right_cam_rotation   [N, 84]   21 per-joint quaternions (x,y,z,w).
    hand_left_cam_rotation    [N, 84]
    hand_right_present         [N]      bool, hand detected that frame
    hand_left_present          [N]      bool

CONVENTIONS (reconcile with the reference pipeline if it differs)
-----------------------------------------------------------------
  * Source: MediaPipe Tasks HandLandmarker (float16), num_hands=2.
  * Positions are `hand_world_landmarks` -- metres, origin at the wrist. This is
    NOT full camera-frame metric placement (that needs per-hand depth/scale from
    the camera pose); it is the standard MediaPipe metric hand shape.
  * Handedness: MediaPipe's category ("Left"/"Right") is taken as the *_cam side
    directly. MediaPipe assumes a mirrored (selfie) view by default; if the
    reference uses the opposite assignment, pass --swap-handedness.
  * Per-joint rotation: a local frame per landmark -- x = bone direction
    (joint->child along the finger; wrist uses wrist->middle-MCP; fingertips
    reuse the incoming bone), z = palm normal, y = z x x -- converted to a
    quaternion (x,y,z,w). Undetected frames: zeros positions, identity rotation.

USAGE
    python extract_hands.py --video undistorted.mp4 --stem MTL-BPT01426-001 \
        --model models/hand_landmarker.task --out hands_dir/
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    sys.exit("mediapipe is required (python3.12): pip install mediapipe")

# MediaPipe 21-landmark parent/child along each finger (for bone directions).
CHILD = {0: 9,  # wrist -> middle MCP
         1: 2, 2: 3, 3: 4, 4: 4,
         5: 6, 6: 7, 7: 8, 8: 8,
         9: 10, 10: 11, 11: 12, 12: 12,
         13: 14, 14: 15, 15: 16, 16: 16,
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


def mat_to_quat(R):
    """3x3 rotation matrix -> quaternion (x,y,z,w)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def joint_rotations(pts):
    """21x3 landmark positions -> 84 (21 quaternions x,y,z,w)."""
    # palm normal from wrist(0), index MCP(5), pinky MCP(17)
    v1 = pts[5] - pts[0]
    v2 = pts[17] - pts[0]
    n = np.cross(v1, v2)
    nn = np.linalg.norm(n)
    palm_n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
    quats = np.zeros((21, 4), dtype=np.float32)
    for j in range(21):
        bone = pts[CHILD[j]] - pts[j]
        bn = np.linalg.norm(bone)
        if bn < 1e-9:
            quats[j] = [0, 0, 0, 1]
            continue
        x = bone / bn
        z = palm_n - np.dot(palm_n, x) * x       # orthogonalise palm normal to x
        zn = np.linalg.norm(z)
        z = z / zn if zn > 1e-9 else np.array([0.0, 0.0, 1.0])
        y = np.cross(z, x)
        quats[j] = mat_to_quat(np.column_stack([x, y, z]))
    return quats.reshape(84)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--stem", required=True, help="Clip id for the output .npz name")
    ap.add_argument("--model", default="models/hand_landmarker.task")
    ap.add_argument("--out", required=True, help="Output --hands-dir")
    ap.add_argument("--swap-handedness", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.3)
    args = ap.parse_args()

    video = Path(args.video)
    stream = ffprobe(video)
    w, h = int(stream["width"]), int(stream["height"])
    fps = fps_of(stream)
    n_frames = int(stream["nb_frames"]) if stream.get("nb_frames") else 0

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=args.min_conf,
        min_hand_presence_confidence=args.min_conf,
        min_tracking_confidence=args.min_conf,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)

    sides = {"right": "hand_right_cam", "left": "hand_left_cam"}
    pos = {s: [] for s in sides}
    rot = {s: [] for s in sides}
    present = {s: [] for s in sides}

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    nbytes = w * h * 3
    idx = 0
    while True:
        raw = dec.stdout.read(nbytes)
        if len(raw) < nbytes:
            break
        rgb = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts_ms = int(idx * 1000 / fps)
        res = landmarker.detect_for_video(image, ts_ms)

        frame = {s: None for s in sides}
        if res.handedness:
            for k, hd in enumerate(res.handedness):
                cat = hd[0].category_name.lower()  # 'left'/'right'
                if args.swap_handedness:
                    cat = "left" if cat == "right" else "right"
                if cat in frame and k < len(res.hand_world_landmarks):
                    wl = res.hand_world_landmarks[k]
                    frame[cat] = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)

        for s in sides:
            if frame[s] is not None:
                pos[s].append(frame[s].reshape(63))
                rot[s].append(joint_rotations(frame[s]))
                present[s].append(True)
            else:
                pos[s].append(np.zeros(63, np.float32))
                q = np.zeros((21, 4), np.float32); q[:, 3] = 1.0
                rot[s].append(q.reshape(84))
                present[s].append(False)
        idx += 1
        if idx % 900 == 0:
            print(f"    hands {idx} frames", flush=True)
    dec.wait()
    landmarker.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"{args.stem}.npz",
        hand_right_cam=np.array(pos["right"], np.float32),
        hand_right_cam_rotation=np.array(rot["right"], np.float32),
        hand_left_cam=np.array(pos["left"], np.float32),
        hand_left_cam_rotation=np.array(rot["left"], np.float32),
        hand_right_present=np.array(present["right"], bool),
        hand_left_present=np.array(present["left"], bool),
    )
    r = int(np.sum(present["right"]))
    l = int(np.sum(present["left"]))
    print(f"{args.stem}: {idx} frames | right hand in {r} | left hand in {l} "
          f"-> {out / (args.stem + '.npz')}")


if __name__ == "__main__":
    main()
