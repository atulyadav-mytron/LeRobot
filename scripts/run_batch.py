#!/usr/bin/env python3
"""
Batch orchestrator: turn many EgoCapture clips from GCS into ONE multi-episode
LeRobot v3.0 hand dataset and upload it. Resumable (per-clip artifacts are
skipped if already present), CPU stages parallel, WiLoR serial on the GPU.

Per clip:
  download (mp4/txt/json)  ->  undistort to 640  ->  VO pose        [CPU, parallel]
                           ->  WiLoR hands                           [GPU, serial]
                           ->  quality flags                         [CPU, parallel]
Then: anno template -> build_hand_dataset (reusing the 640 videos) -> upload.

Intrinsics are fixed to the nvwa reference clip-001 K for all clips (one fisheye
calibration; byte-matches the reference's camera_intrinsics).

USAGE
  python run_batch.py --src gs://ots-egocentric-inventory/Garment_Factory_workbench/2026-04-28/1102 \
                      --src gs://.../2026-04-28/1107 \
                      --out gs://narration-raw-delivery/lerobot_sample_10hr \
                      --work /mnt/work --mano-dir /opt/mano [--limit N] [--hands mediapipe|wilor]
"""

import argparse
import concurrent.futures as cf
import importlib.util
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
# nvwa reference clip-001 K (640 view): fx,fy,cx,cy
INTR = "157.77337646484375,157.78651428222656,320.27496337890625,320.2656555175781"
FX, FY, CX, CY = (float(v) for v in INTR.split(","))

_spec = importlib.util.spec_from_file_location("bhd", HERE / "build_hand_dataset.py")
bhd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bhd)
import numpy as np
K640 = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])


def sh(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def gcs_ls_mp4(prefix):
    out = subprocess.run(["gcloud", "storage", "ls", f"{prefix.rstrip('/')}/**.mp4"],
                         capture_output=True, text=True)
    return [l.strip() for l in out.stdout.splitlines() if l.strip().endswith(".mp4")]


class Log:
    def __init__(self, path, out_gcs):
        self.path, self.out_gcs = path, out_gcs
        self.t0 = time.time()
    def __call__(self, msg):
        line = f"[{time.time()-self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")
    def push(self):
        try:
            sh(["gcloud", "storage", "cp", str(self.path),
                f"{self.out_gcs}/_logs/run.log"])
        except Exception:
            pass


def download(uri, clips_dir):
    stem = Path(uri).stem
    base = uri[:-4]  # strip .mp4
    for ext in (".mp4", ".txt", ".json"):
        dst = clips_dir / f"{stem}{ext}"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        r = subprocess.run(["gcloud", "storage", "cp", base + ext, str(dst)],
                           capture_output=True, text=True)
        if r.returncode != 0 and ext != ".txt":  # txt/json may rarely be absent
            raise RuntimeError(f"download {stem}{ext}: {r.stderr[:200]}")
    return stem


def cpu_stage(uri, dirs, log):
    """download -> undistort(640) -> VO. Returns stem or raises."""
    stem = download(uri, dirs["clips"])
    src = dirs["clips"] / f"{stem}.mp4"
    vid = dirs["videos"] / f"{stem}.mp4"
    if not (vid.exists() and vid.stat().st_size > 0):
        st = bhd.ffprobe(src)
        bhd.undistort_to_640(src, vid, bhd.fps_of(st), K640)
    tum = dirs["pose"] / f"{stem}.tum"
    if not tum.exists():
        sh([PY, str(HERE / "estimate_camera_pose.py"), "--video", str(vid),
            "--stem", stem, "--fx", str(FX), "--fy", str(FY), "--cx", str(CX),
            "--cy", str(CY), "--out", str(dirs["pose"])])
    return stem


def gpu_stage(stem, dirs, hands_impl, mano_dir, log):
    npz = dirs["hands"] / f"{stem}.npz"
    if npz.exists():
        return
    src = dirs["clips"] / f"{stem}.mp4"
    if hands_impl == "wilor":
        cmd = [PY, str(HERE / "extract_hands_wilor.py"), "--video", str(src),
               "--stem", stem, "--out", str(dirs["hands"])]
        if mano_dir:
            cmd += ["--mano-dir", mano_dir]
    else:
        cmd = [PY, str(HERE / "extract_hands.py"), "--video", str(src),
               "--stem", stem, "--model", str(HERE.parent / "models/hand_landmarker.task"),
               "--out", str(dirs["hands"])]
    sh(cmd)


def quality_stage(stem, dirs):
    q = dirs["quality"] / f"{stem}.parquet"
    if q.exists():
        return
    sh([PY, str(HERE / "compute_quality.py"), "--hands", str(dirs["hands"] / f"{stem}.npz"),
        "--pose", str(dirs["pose"] / f"{stem}.tum"), "--stem", stem,
        "--out", str(dirs["quality"]), "--fps", "30"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="GCS prefix(es) of clips")
    ap.add_argument("--out", required=True, help="GCS output prefix")
    ap.add_argument("--work", required=True)
    ap.add_argument("--mano-dir", default=None)
    ap.add_argument("--hands", choices=["wilor", "mediapipe"], default="wilor")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    work = Path(args.work)
    dirs = {k: work / k for k in ("clips", "videos", "hands", "pose", "quality")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    log = Log(work / "run.log", args.out)

    uris = []
    for p in args.src:
        uris += gcs_ls_mp4(p)
    uris = sorted(uris)
    if args.limit:
        uris = uris[:args.limit]
    log(f"clips: {len(uris)} from {len(args.src)} source(s); hands={args.hands}")
    log.push()

    # Phase 1: CPU (download + undistort + VO), parallel
    done, failed = [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cpu_stage, u, dirs, log): u for u in uris}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            u = futs[fut]
            try:
                done.append(fut.result())
            except Exception as e:
                failed.append(Path(u).stem)
                log(f"  [FAIL cpu] {Path(u).stem}: {e}")
            if i % 10 == 0:
                log(f"  cpu stage {i}/{len(uris)} done")
                log.push()
    log(f"CPU stage: {len(done)} ok, {len(failed)} failed")
    log.push()

    # Phase 2: GPU (WiLoR), serial
    for i, stem in enumerate(sorted(done), 1):
        try:
            gpu_stage(stem, dirs, args.hands, args.mano_dir, log)
        except Exception as e:
            failed.append(stem)
            log(f"  [FAIL gpu] {stem}: {e}")
        if i % 10 == 0:
            log(f"  gpu stage {i}/{len(done)}")
            log.push()
    log("GPU stage done")
    log.push()

    # Phase 3: quality, parallel
    usable = [s for s in sorted(set(done)) if s not in failed
              and (dirs["hands"] / f"{s}.npz").exists()]
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda s: _safe(quality_stage, s, dirs, log), usable))
    log(f"quality stage done for {len(usable)} clips")

    # Phase 4: anno + build + upload
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
    log(f"UPLOAD complete -> {args.out}  ({len(usable)} episodes, {len(failed)} failed clips)")
    log.push()
    print("RUN_BATCH_DONE")


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception as e:
        print(f"  [warn] {fn.__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
