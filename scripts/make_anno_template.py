#!/usr/bin/env python3
"""
Generate the --anno JSON (task / subtask vocabulary + per-frame subtask
segmentation) that build_hand_dataset.py consumes.

This is the tooling for producer #4. The *labels and boundaries* are inherently
an annotation product -- a human or a VLM has to say what the subtasks are and
where they start/end. This script:
  * pulls the task text + scene from each clip's <stem>.json metadata,
  * writes a ready-to-edit anno entry per clip, and
  * can emit a PLACEHOLDER uniform segmentation (--subtasks N) so the pipeline's
    multi-subtask path (tasks/subtasks parquet + per-subtask train_intervals) is
    exercised end-to-end before real labels exist.

Replace the placeholder segmentation with real labels by either editing the JSON
by hand, or filling it from a VLM pass over the video (see --vlm notes in README).

anno.json schema consumed by build_hand_dataset.py:
    { "<clip-stem>": {
        "task": "<natural-language task>",
        "scene": "<scene id>", "scene_label": "<localized label>",
        "subtasks": ["<subtask 0 text>", "<subtask 1 text>", ...],
        "subtask_per_frame": [0,0,...,1,1,...]    # length N, index into subtasks
    } }

USAGE
    python make_anno_template.py --input <clips_dir_or_clip> --out anno.json \
        [--subtasks 2]        # 0/1 = single subtask; N>1 = uniform placeholder split
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def nb_frames(mp4: Path) -> int:
    if not mp4.exists():
        return 0           # assemble step has no source mp4; seg falls back to None
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of",
            "default=noprint_wrappers=1:nokey=1", str(mp4)], text=True).strip()
        return int(out)
    except (ValueError, subprocess.CalledProcessError):
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subtasks", type=int, default=1,
                    help="N>1 writes a uniform PLACEHOLDER segmentation into N subtasks")
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_file():
        clips = [inp.with_suffix("")]
    else:
        stems = {p.with_suffix("") for p in inp.glob("*.mp4")}
        stems |= {p.with_suffix("") for p in inp.glob("*.json")}  # assemble: json only
        clips = sorted(stems)
    if not clips:
        sys.exit(f"No clips under {inp}")

    anno = {}
    for stem in clips:
        name = stem.name
        meta = {}
        jp = stem.with_suffix(".json")
        if jp.exists():
            try:
                meta = json.loads(jp.read_text())
            except json.JSONDecodeError:
                meta = {}
        task = meta.get("task_description", "activity recording")
        scene = meta.get("environment_l3", meta.get("environment-id", "unknown"))
        n = nb_frames(stem.with_suffix(".mp4"))

        k = max(1, args.subtasks)
        if k == 1:
            subtasks = [task]
            seg = [0] * n if n else None
        else:
            subtasks = [f"{task} [subtask {i+1}/{k} — PLACEHOLDER uniform split]"
                        for i in range(k)]
            seg = [min(i * k // n, k - 1) for i in range(n)] if n else None

        anno[name] = {
            "task": task, "scene": scene, "scene_label": scene,
            "subtasks": subtasks,
            "subtask_per_frame": seg,
            "_note": ("PLACEHOLDER segmentation — replace subtasks + subtask_per_frame "
                      "with real labels (human or VLM)." if k > 1 else
                      "Single subtask — add real subtasks + subtask_per_frame when available."),
        }
        print(f"  {name}: {n} frames, {k} subtask(s)")

    Path(args.out).write_text(json.dumps(anno, indent=2) + "\n")
    print(f"wrote {args.out}  ({len(anno)} clip(s))")


if __name__ == "__main__":
    main()
