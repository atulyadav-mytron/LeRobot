# VDEGO-C2 — LeRobot v3.0 Dataset Conversion

## Objective

Convert the VDEGO-C2 / EgoCapture-1 egocentric recordings into a **LeRobot v3.0**
dataset.

## Two builders (pick by target schema)

- **`scripts/build_hand_dataset.py`** — the **current target**: a hand-manipulation
  dataset (`robot_type: human_hand`) matching the `nvwa_2full_episodes` reference
  exactly. Per-frame 3D hand keypoints + joint rotations (both hands), camera pose,
  per-frame intrinsics, 640×640 video, a 19-flag quality `frame_masks`, and
  `window_contract` / `frame_filter` metadata. **Start here.**
- **`scripts/build_lerobot_dataset.py`** + **`scripts/ingest_egocapture.py`** — an
  earlier camera-pose/IMU variant (generic v3.0 skeleton). Kept for reference.

### `build_hand_dataset.py` — what it fills vs. what needs upstream data

Fills now (no extra inputs): the 640×640 undistorted pinhole video, the real
`camera_intrinsics`, all index/timestamp columns, and the complete meta tree
(info.json, episodes, tasks, subtasks, 20-column frame_masks, train_intervals).
Task/subtask/scene text is read from each clip's `<stem>.json`.

Typed placeholders until the matching source is supplied:

| Field | Placeholder | Supply via |
|---|---|---|
| `hand_{right,left}_cam` [63], `..._rotation` [84] | zeros / identity quats | `--hands-dir` (3D hand-pose pipeline: MediaPipe / HaMeR / WiLoR) |
| `camera_position` [3], `camera_rotation` [4] | zeros / identity | `--pose-dir` (SLAM trajectory, TUM/CSV) |
| `frame_masks` quality flags | all-valid, `train_keep=True` | `--quality-dir` (hand-kinematics + VLM validation) |
| `subtask_index` + subtask vocab | single subtask | `--anno` (segmentation JSON) |

```bash
pip install numpy pyarrow opencv-python-headless    # + ffmpeg on PATH
python scripts/build_hand_dataset.py --input /path/to/clips --output /path/to/dataset
```

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

## Ingesting raw EgoCapture-1 clips

For the `ots-egocentric-inventory` bucket layout (one clip = `<stem>.mp4` fisheye
video + `<stem>.txt` JSON-Lines IMU + `<stem>.json` metadata + `<stem>.aac`
audio), use `scripts/ingest_egocapture.py`. It:

- undistorts the 185° fisheye to a **pinhole** view using the Kalibr calibration
  (same K/D/FOV as the calibration deliverable's `undistort_to_pinhole.py`),
- converts the `.txt` IMU to `imu.csv` (SI units kept: m/s², rad/s),
- writes per-frame timestamps from the IMU microsecond clock (IMU is 1:1 with
  frames at 30 Hz),
- optionally injects per-clip SLAM poses (`--trajectory-dir`), then
- runs `build_lerobot_dataset.py` to emit the v3.0 dataset.

```bash
pip install numpy pandas pyarrow opencv-python-headless   # + ffmpeg on PATH

python scripts/ingest_egocapture.py \
    --input   /path/to/clips \
    --staging /path/to/staging \
    --output  /path/to/lerobot_dataset \
    --trajectory-dir /path/to/poses      # optional; omit = zero-filled state/action
```

### Remaining external input: camera poses

The bucket does **not** contain camera trajectories, and they cannot be derived
from this data alone: the IMU is only 30 Hz (one sample per frame), too low for
reliable visual-inertial fusion or metric scale. Poses must come from running
**monocular (or monocular-inertial) SLAM** — e.g. ORB-SLAM3 / COLMAP — on the
undistorted pinhole video, producing one trajectory file per clip (TUM or CSV)
for `--trajectory-dir`. Expect **up-to-scale** translation (orientation is still
good); record metric-vs-up-to-scale in `info.json`. Until poses are supplied,
`observation.state` / `action` are zero-filled and every other field is complete.

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `pyarrow`
- `opencv-python-headless` (for `ingest_egocapture.py` undistortion)
- `ffmpeg` / `ffprobe` on `PATH`
