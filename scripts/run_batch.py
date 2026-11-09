#!/usr/bin/env python3
"""
Batch orchestrator: turn many EgoCapture clips from GCS into ONE multi-episode
LeRobot v3.0 hand dataset and upload it. Resumable; **CPU and GPU work overlap**.

The only prerequisite for any producer is the downloaded source clip:
  - WiLoR (GPU) runs on the source directly (it undistorts its own detect view).
  - undistort->640 + VO (CPU) also run on the source.
So they are independent. A single GPU worker thread streams WiLoR over the clips
while a CPU thread-pool does undistort+VO in parallel -- the L4 stays busy instead
of idling through a separate CPU phase. Then quality (needs hands+pose), build,
upload. Per-clip outputs are skipped if present, so a restart resumes cleanly.

Intrinsics are fixed to the nvwa reference clip-001 K for all clips.

USAGE
  python run_batch.py --src gs://.../1102 --src gs://.../1107 \
      --out gs://annotationportal-raw/lerobot_sample_10hr \
      --work /home/work --mano-dir /opt/mano [--limit N] [--hands wilor|mediapipe]
"""

import argparse
import concurrent.futures as cf
import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
INTR = "157.77337646484375,157.78651428222656,320.27496337890625,320.2656555175781"
FX, FY, CX, CY = (float(v) for v in INTR.split(","))

_spec = importlib.util.spec_from_file_location("bhd", HERE / "build_hand_dataset.py")
bhd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bhd)
import numpy as np
K640 = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])

_lock = threading.Lock()
_dl_locks = {}
_failed = set()


def sh(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def gcs_ls_mp4(prefix):
    out = subprocess.run(["gcloud", "storage", "ls", f"{prefix.rstrip('/')}/**.mp4"],
                         capture_output=True, text=True)
    return [l.strip() for l in out.stdout.splitlines() if l.strip().endswith(".mp4")]


class Log:
    def __init__(self, path, out_gcs):
        self.path, self.out_gcs, self.t0 = path, out_gcs, time.time()
    def __call__(self, msg):
        line = f"[{time.time()-self.t0:8.1f}s] {msg}"
        with _lock:
            print(line, flush=True)
            with open(self.path, "a") as f:
                f.write(line + "\n")
    def push(self):
        try:
            sh(["gcloud", "storage", "cp", str(self.path), f"{self.out_gcs}/_logs/run.log"])
        except Exception:
            pass


def _download(uri, clips_dir):
    stem = Path(uri).stem
    base = uri[:-4]
    for ext in (".mp4", ".txt", ".json"):
        dst = clips_dir / f"{stem}{ext}"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        r = subprocess.run(["gcloud", "storage", "cp", base + ext, str(dst)],
                           capture_output=True, text=True)
        if r.returncode != 0 and ext == ".mp4":
            raise RuntimeError(f"download {stem}{ext}: {r.stderr[:200]}")
    return stem


def ensure_download(uri, clips_dir):
    """Thread-safe, idempotent: only one thread downloads a given clip."""
    stem = Path(uri).stem
    with _lock:
        lk = _dl_locks.setdefault(stem, threading.Lock())
    with lk:
        return _download(uri, clips_dir)


def cpu_stage(uri, dirs):
    """ensure download -> undistort(640) -> VO. (runs in the CPU pool)"""
    stem = ensure_download(uri, dirs["clips"])
    src = dirs["clips"] / f"{stem}.mp4"
    vid = dirs["videos"] / f"{stem}.mp4"
    if not (vid.exists() and vid.stat().st_size > 0):
        st = bhd.ffprobe(src)
        bhd.undistort_to_640(src, vid, bhd.fps_of(st), K640)
    tum = dirs["pose"] / f"{stem}.tum"
    if not tum.exists():
        sh([PY, str(HERE / "estimate_camera_pose.py"), "--video", str(vid), "--stem", stem,
            "--fx", str(FX), "--fy", str(FY), "--cx", str(CX), "--cy", str(CY),
            "--out", str(dirs["pose"])])
    return stem


def gpu_stage(stem, dirs, hands_impl, mano_dir):
    npz = dirs["hands"] / f"{stem}.npz"
    if npz.exists():
        return
    src = dirs["clips"] / f"{stem}.mp4"
    if hands_impl == "wilor":
        cmd = [PY, str(HERE / "extract_hands_wilor.py"), "--video", str(src), "--stem", stem,
               "--out", str(dirs["hands"])]
        if mano_dir:
            cmd += ["--mano-dir", mano_dir]
    else:
        cmd = [PY, str(HERE / "extract_hands.py"), "--video", str(src), "--stem", stem,
               "--model", str(HERE.parent / "models/hand_landmarker.task"), "--out", str(dirs["hands"])]
    sh(cmd)


def gpu_worker(uris, dirs, hands_impl, mano_dir, log):
    """Single GPU consumer: streams WiLoR over clips, overlapping the CPU pool."""
    for i, uri in enumerate(uris, 1):
        stem = Path(uri).stem
        try:
            ensure_download(uri, dirs["clips"])   # usually already fetched by CPU side
            gpu_stage(stem, dirs, hands_impl, mano_dir)
        except Exception as e:
            with _lock:
                _failed.add(stem)
            log(f"  [FAIL gpu] {stem}: {e}")
        if i % 10 == 0:
            log(f"  GPU(hands) {i}/{len(uris)}")
            log.push()
    log("GPU(hands) worker done")


def quality_stage(stem, dirs):
    q = dirs["quality"] / f"{stem}.parquet"
    if q.exists():
        return
    sh([PY, str(HERE / "compute_quality.py"), "--hands", str(dirs["hands"] / f"{stem}.npz"),
        "--pose", str(dirs["pose"] / f"{stem}.tum"), "--stem", stem,
        "--out", str(dirs["quality"]), "--fps", "30"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--mano-dir", default=None)
    ap.add_argument("--hands", choices=["wilor", "mediapipe"], default="wilor")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-hands", type=int, default=0,
                    help="Abort (exit 2) if fewer than N hand-present frames are detected "
                         "across all clips -- guards against a silently-broken hand model.")
    args = ap.parse_args()

    work = Path(args.work)
    dirs = {k: work / k for k in ("clips", "videos", "hands", "pose", "quality")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    log = Log(work / "run.log", args.out)

    uris = sorted(u for p in args.src for u in gcs_ls_mp4(p))
    if args.limit:
        uris = uris[:args.limit]
    log(f"clips: {len(uris)}; hands={args.hands}; CPU+GPU overlapped (workers={args.workers})")
    log.push()

    # GPU worker (WiLoR) runs concurrently with the CPU pool (undistort+VO).
    gpu_t = threading.Thread(target=gpu_worker, args=(uris, dirs, args.hands, args.mano_dir, log))
    gpu_t.start()

    done = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cpu_stage, u, dirs): u for u in uris}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            u = futs[fut]
            try:
                done.append(fut.result())
            except Exception as e:
                with _lock:
                    _failed.add(Path(u).stem)
                log(f"  [FAIL cpu] {Path(u).stem}: {e}")
            if i % 10 == 0:
                log(f"  CPU(undistort+VO) {i}/{len(uris)}")
                log.push()
    log("CPU pool done; waiting on GPU worker...")
    log.push()
    gpu_t.join()

    # guard: a silently-broken hand model would produce all-zero hands
    total_hands = 0
    for u in uris:
        f = dirs["hands"] / f"{Path(u).stem}.npz"
        if f.exists():
            z = np.load(f)
            total_hands += int(z["hand_right_present"].sum() + z["hand_left_present"].sum())
    log(f"total hand-present frames across clips: {total_hands}")
    log.push()
    if args.require_hands and total_hands < args.require_hands:
        log(f"ABORT: require-hands={args.require_hands} but only {total_hands} detected "
            "-- the hand model is likely broken (not shipping an empty dataset).")
        sys.exit(2)

    # quality (needs hands + pose), parallel
    usable = [Path(u).stem for u in uris
              if Path(u).stem not in _failed and (dirs["hands"] / f"{Path(u).stem}.npz").exists()
              and (dirs["pose"] / f"{Path(u).stem}.tum").exists()]
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda s: _safe(quality_stage, s, dirs, log), usable))
    log(f"quality done for {len(usable)} clips ({len(_failed)} failed)")
    log.push()

    # anno + build + upload
    sh([PY, str(HERE / "make_anno_template.py"), "--input", str(dirs["clips"]),
        "--out", str(work / "anno.json"), "--subtasks", "1"])
    ds = work / "dataset"
    if ds.exists():
        sh(["rm", "-rf", str(ds)])
    sh([PY, str(HERE / "build_hand_dataset.py"), "--input", str(dirs["clips"]),
        "--video-dir", str(dirs["videos"]), "--hands-dir", str(dirs["hands"]),
        "--pose-dir", str(dirs["pose"]), "--quality-dir", str(dirs["quality"]),
        "--anno", str(work / "anno.json"), "--intrinsics", INTR, "--output", str(ds)])
    log("build complete; uploading...")
    log.push()
    sh(["gcloud", "storage", "cp", "--recursive", str(ds) + "/*", args.out.rstrip("/") + "/"])
    log(f"UPLOAD complete -> {args.out}  ({len(usable)} episodes, {len(_failed)} failed)")
    log.push()
    print("RUN_BATCH_DONE")


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception as e:
        print(f"  [warn] {fn.__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
