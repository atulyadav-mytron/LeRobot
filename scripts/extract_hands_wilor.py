#!/usr/bin/env python3
"""
GPU hand extractor using WiLoR (WiLoR-mini) -- OPTIMIZED. Same .npz output
contract as the MediaPipe version, consumed by build_hand_dataset.py --hands-dir.

OPTIMIZATIONS vs the naive per-frame version
--------------------------------------------
  * BATCHED DETECTION: the YOLO hand detector runs on a batch of frames per GPU
    call (not one at a time). 3D pose uses WiLoR-mini's own predict_with_bboxes
    (correct crop/flip/pose), so only the detector path is reimplemented.
  * CPU<->GPU OVERLAP: a producer thread decodes (ffmpeg) + undistorts frames
    into a queue while the main thread runs batched inference -- the L4 no longer
    idles waiting on per-frame CPU prep.
  * FRAME-STRIDE: run WiLoR every --frame-stride frames and hold the result for
    the skipped frames (e.g. stride 3 -> ~10 fps hands, ~3x fewer GPU calls).

Detection still runs on a decoupled undistorted view (896/90) of the SOURCE
fisheye; world keypoints are projection-independent. Per-joint rotations are
derived from the keypoints (shared bone-direction method). Output npz keys and
conventions are unchanged, so this is a drop-in replacement.

USAGE
    python extract_hands_wilor.py --video SRC.mp4 --stem <id> --out hands_dir/ \
        [--frame-stride 3] [--batch 32] [--detect-size 896 --detect-hfov 90]
"""

import argparse
import importlib.util
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv is required")

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("extract_hands", _here / "extract_hands.py")
_eh = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_eh)
    FISHEYE_K, FISHEYE_D = _eh.FISHEYE_K, _eh.FISHEYE_D
    detect_K, ffprobe, fps_of, joint_rotations = _eh.detect_K, _eh.ffprobe, _eh.fps_of, _eh.joint_rotations
except SystemExit:
    import math as _math
    FISHEYE_K = np.array([[707.2808075315367, 0.0, 956.0608949841028],
                          [0.0, 707.6480183518742, 550.7429189294107], [0.0, 0.0, 1.0]])
    FISHEYE_D = np.array([0.011734004280818429, 9.279535348401179e-06,
                          -0.00015706402699172342, -0.0002178771185773861])
    _CHILD = {0: 9, 1: 2, 2: 3, 3: 4, 4: 4, 5: 6, 6: 7, 7: 8, 8: 8, 9: 10, 10: 11,
              11: 12, 12: 12, 13: 14, 14: 15, 15: 16, 16: 16, 17: 18, 18: 19, 19: 20, 20: 20}

    def detect_K(size, hfov_deg):
        fp = (size / 2.0) / _math.tan(_math.radians(hfov_deg / 2.0))
        return np.array([[fp, 0, size / 2.0], [0, fp, size / 2.0], [0, 0, 1]])

    def ffprobe(path):
        import json
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)], text=True)
        return json.loads(out)["streams"][0]

    def fps_of(s):
        num, _, den = s.get("r_frame_rate", "30/1").partition("/")
        try:
            return int(num) / int(den)
        except (ValueError, ZeroDivisionError):
            return 30.0

    def _m2q(R):
        t = R[0, 0] + R[1, 1] + R[2, 2]
        if t > 0:
            s = np.sqrt(t + 1.0) * 2; q = [(R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s, 0.25*s]
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; q = [0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s]
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; q = [(R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s]
        else:
            s = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; q = [(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s]
        return np.array(q, dtype=np.float32)

    def joint_rotations(pts):
        v1, v2 = pts[5]-pts[0], pts[17]-pts[0]
        n = np.cross(v1, v2); nn = np.linalg.norm(n)
        palm = n/nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
        qs = np.zeros((21, 4), np.float32)
        for j in range(21):
            bone = pts[_CHILD[j]]-pts[j]; bn = np.linalg.norm(bone)
            if bn < 1e-9:
                qs[j] = [0, 0, 0, 1]; continue
            x = bone/bn; z = palm-np.dot(palm, x)*x; zn = np.linalg.norm(z)
            z = z/zn if zn > 1e-9 else np.array([0.0, 0.0, 1.0]); y = np.cross(z, x)
            qs[j] = _m2q(np.column_stack([x, y, z]))
        return qs.reshape(84)


def decoder(video, sw, sh, map1, map2, stride, q):
    """Producer: decode source, undistort every stride-th frame to RGB, enqueue."""
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    nbytes = sw * sh * 3
    idx = 0
    while True:
        raw = dec.stdout.read(nbytes)
        if len(raw) < nbytes:
            break
        if idx % stride == 0:
            bgr = np.frombuffer(raw, np.uint8).reshape(sh, sw, 3)
            view = cv2.remap(bgr, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            q.put((idx, cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
        idx += 1
    dec.wait()
    q.put((idx, None))  # sentinel carries the total frame count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mano-dir", default=None)      # accepted for compat; MANO auto-downloads
    ap.add_argument("--detect-size", type=int, default=896)
    ap.add_argument("--detect-hfov", type=float, default=90.0)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    print(f"WiLoR on {device} ({dtype}); stride={args.frame_stride} batch={args.batch}", flush=True)

    stream = ffprobe(args.video)
    sw, sh = int(stream["width"]), int(stream["height"])
    size = args.detect_size
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        FISHEYE_K, FISHEYE_D, np.eye(3), detect_K(size, args.detect_hfov), (size, size), cv2.CV_16SC2)

    q = queue.Queue(maxsize=args.batch * 4)
    th = threading.Thread(target=decoder, args=(args.video, sw, sh, map1, map2, args.frame_stride, q))
    th.start()

    # results per PROCESSED frame index
    res = {}  # idx -> {"right": (pos63,rot84)|None, "left": ...}
    n_frames = 0
    done = False

    def run_batch(items):
        imgs = [im for _, im in items]
        dets = pipe.hand_detector(imgs, conf=args.conf, verbose=False)
        for (idx, img), d in zip(items, dets):
            frame = {"right": None, "left": None}
            b = d.boxes
            if b is not None and len(b) > 0:
                data = b.data.cpu().numpy()                 # [n, 6] xyxy conf cls
                bboxes = data[:, :4]
                is_rights = data[:, 5].tolist()
                try:
                    rets = pipe.predict_with_bboxes(img, bboxes, is_rights)
                except Exception:
                    rets = []
                for o in rets:
                    side = "right" if float(o.get("is_right", 1.0)) >= 0.5 else "left"
                    kp = np.asarray(o["wilor_preds"]["pred_keypoints_3d"], np.float32).reshape(-1, 3)
                    if kp.shape[0] >= 21:
                        frame[side] = (kp[:21].reshape(63), joint_rotations(kp[:21]))
            res[idx] = frame

    batch = []
    processed = 0
    while True:
        idx, img = q.get()
        if img is None:
            n_frames = idx
            done = True
        else:
            batch.append((idx, img))
        if len(batch) >= args.batch or (done and batch):
            run_batch(batch)
            processed += len(batch)
            batch = []
            if processed % (args.batch * 8) < args.batch:
                print(f"    wilor processed {processed} frames", flush=True)
        if done:
            break
    th.join()

    # expand processed results to every frame (hold across the stride gap)
    stride = args.frame_stride
    pos = {"right": np.zeros((n_frames, 63), np.float32), "left": np.zeros((n_frames, 63), np.float32)}
    rot = {s: np.tile(np.concatenate([np.zeros((21, 3)), np.ones((21, 1))], 1).reshape(84).astype(np.float32),
                      (n_frames, 1)) for s in ("right", "left")}
    present = {"right": np.zeros(n_frames, bool), "left": np.zeros(n_frames, bool)}
    for i in range(n_frames):
        src = (i // stride) * stride
        r = res.get(src)
        if not r:
            continue
        for s in ("right", "left"):
            if r[s] is not None:
                pos[s][i] = r[s][0]; rot[s][i] = r[s][1]; present[s][i] = True

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.stem}.npz",
             hand_right_cam=pos["right"], hand_right_cam_rotation=rot["right"],
             hand_left_cam=pos["left"], hand_left_cam_rotation=rot["left"],
             hand_right_present=present["right"], hand_left_present=present["left"])
    r, l = int(present["right"].sum()), int(present["left"].sum())
    either = int((present["right"] | present["left"]).sum())
    print(f"{args.stem}: {n_frames} frames (stride {stride}) | right {r} | left {l} | "
          f"either {either} ({100*either/max(n_frames,1):.1f}%) -> {out / (args.stem + '.npz')}")


if __name__ == "__main__":
    main()
