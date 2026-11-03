# VDEGO-C2 — LeRobot v3.0 Dataset Conversion

## Objective

Convert the VDEGO-C2 stereo egocentric recording into a **LeRobot v3.0**
dataset.

## Inputs

- Camera calibration (`calibration/calibration.json`)
- VIO trajectory (`trajectory.csv`)
- IMU measurements (`imu.csv`)
- Video frame timestamps (`video_frames_timestamp.csv`)
- Stereo camera recordings (`cam_left.mp4`, `cam_right.mp4` per segment)

## Conversion

`scripts/build_lerobot_dataset.py`:

- Parses calibrated camera parameters
- Aligns the VIO trajectory with video frames
- Converts quaternion orientation to axis-angle rotation vectors
- Associates IMU measurements with video frames (nearest-neighbour)
- Writes per-frame Parquet, per-episode metadata, task/subtask vocabularies,
  frame masks and training intervals
- Encodes camera video in H.264

### State / action / IMU

- `observation.state` — per-frame VIO pose `[tx, ty, tz, rx, ry, rz]`.
  `rx, ry, rz` is an **axis-angle rotation vector in radians** (cv2 `rvec`
  convention, converted from the trajectory quaternion). Units: metres, radians.
- `action` — the next frame's state; the final frame repeats its own.
- `observation.imu` — `[ax, ay, az, gx, gy, gz]`, m/s² and rad/s.

### Cameras

The target format uses a single camera key, `observation.images.main`. The
left camera becomes `main` by default (`--main-camera right` to flip). The
other camera is dropped unless `--stereo` is passed, which adds it as
`observation.images.second`.

### Annotation tables (defaults)

`subtasks.parquet`, `frame_masks/` and `train_intervals/` describe annotations
the raw capture does not carry. They are written with explicit defaults so the
dataset is structurally complete and immediately loadable; replace them with
real annotations later without changing the layout:

- **subtasks.parquet** — one default subtask (`--subtask-label`) covering the
  whole recording.
- **frame_masks** — every frame flagged `valid=True`. Set flags to `False` to
  exclude frames.
- **train_intervals** — the maximal contiguous runs of valid frames (one per
  episode under the default). Re-running after editing `frame_masks` recomputes
  them.

## Dataset structure

```text
dataset_root/
├── data/chunk-000/
│   ├── file-000.parquet                      # episode 0, per-frame rows
│   └── file-001.parquet                      # episode 1, per-frame rows
├── meta/
│   ├── info.json                             # format, fps, path templates
│   ├── episodes/chunk-000/file-000.parquet   # one location record per episode
│   ├── tasks.parquet                         # task vocabulary
│   ├── subtasks.parquet                      # action-stage vocabulary
│   ├── frame_masks/part-00000.parquet        # per-frame validity flags
│   ├── train_intervals/part-00000.parquet    # contiguous valid intervals
│   └── stats.json                            # aggregate feature statistics
└── videos/observation.images.main/chunk-000/
    ├── file-000.mp4
    └── file-001.mp4
```

Path templates (recorded in `meta/info.json`):

- `data_path`  = `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
- `video_path` = `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`
- `meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`

## Usage

```bash
pip install numpy pandas pyarrow          # ffmpeg/ffprobe must be on PATH

python scripts/build_lerobot_dataset.py \
    --input  "/path/to/Stereo_1080p" \
    --output "/path/to/lerobot_dataset" \
    --task-label "factory worker activity recording"
```

Useful flags: `--stereo`, `--main-camera {left,right}`, `--no-imu`,
`--copy-video` (keep HEVC instead of re-encoding), `--subtask-label`.

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `pyarrow`
- `ffmpeg` / `ffprobe` on `PATH`
