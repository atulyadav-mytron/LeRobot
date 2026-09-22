\# VDEGO-C2 Task 2 — LeRobot Dataset Conversion



\## Objective



Convert the VDEGO-C2 stereo egocentric recording into a LeRobot v2.1-compatible dataset.



\## Inputs



\- Camera calibration

\- VIO trajectory

\- IMU measurements

\- Video frame timestamps

\- Stereo camera recordings



\## Conversion



The `build\_lerobot\_dataset.py` script:

\- Parses calibrated camera parameters

\- Aligns VIO trajectory with video frames

\- Converts quaternion orientation to axis-angle rotation vectors

\- Associates IMU measurements with video frames

\- Generates LeRobot metadata

\- Generates Parquet episode data

\- Encodes camera videos in H.264



\## Dataset Structure



```text

lerobot\_dataset/

├── meta/

│   ├── info.json

│   ├── camera.json

│   ├── episodes.jsonl

│   └── tasks.jsonl

├── data/

│   └── chunk-000/

│       └── episode\_000000.parquet

└── videos/

&#x20;   └── chunk-000/

