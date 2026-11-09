#!/bin/bash
# Produce VM: process one shard of clips with WiLoR and upload per-clip artifacts
# to the shared staging area. Reads the shard ("i/N") from instance metadata.
# Self-deletes on any exit (no idle). One assemble VM later builds the dataset.
set -uo pipefail
export HOME=/root DEBIAN_FRONTEND=noninteractive
export PATH=/opt/venv/bin:/opt/conda/bin:/snap/bin:/usr/local/bin:/usr/bin:/bin:$PATH

OUT="gs://annotationportal-raw/lerobot_sample_10hr"
STAGING="$OUT/_staging"
SRC1="gs://ots-egocentric-inventory/Garment_Factory_workbench/2026-04-28/1102"
SRC2="gs://ots-egocentric-inventory/Garment_Factory_workbench/2026-04-28/1107"
WORK=/home/work ; PYBIN=/opt/venv/bin/python ; LOG=/var/log/lerobot_run.log
mkdir -p "$WORK" ; exec > >(tee -a "$LOG") 2>&1

NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/name)
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
SHARD=$(curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/attributes/shard)
TAG="shard$(echo "$SHARD" | tr '/' '_')"

marker() { echo "$2 $(date -u)" >/tmp/_m; gcloud storage cp /tmp/_m "$OUT/_logs/${TAG}_$1" 2>/dev/null || true; }
cleanup() { local rc=$?; echo "cleanup rc=$rc"; marker "EXIT_$rc" "exit $rc";
  gcloud storage cp "$LOG" "$OUT/_logs/${TAG}.log" 2>/dev/null || true
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || shutdown -h now; }
trap cleanup EXIT
( sleep 43200; shutdown -h now ) &   # 12h safety

echo "=== STARTED $TAG $(date -u) ==="; marker STARTED started; nvidia-smi || true

set -e
apt-get update -y
apt-get install -y python3-pip python3-venv git build-essential ffmpeg libgl1 libglib2.0-0 libsm6 libxext6
python3 -m venv /opt/venv
"$PYBIN" -m pip install --upgrade pip setuptools wheel Cython numpy dill
"$PYBIN" -m pip install --no-build-isolation "git+https://github.com/warmshao/WiLoR-mini"
"$PYBIN" -m pip install pyarrow
"$PYBIN" -c "import wilor_mini, cv2, pyarrow, torch; print('deps OK; cuda', torch.cuda.is_available())"
set +e

cd /home && rm -rf repo && git clone --depth 1 https://github.com/atulyadav-mytron/LeRobot.git repo
echo "=== PRODUCE shard $SHARD ==="
if "$PYBIN" /home/repo/scripts/run_batch.py --src "$SRC1" --src "$SRC2" \
     --mode produce --shard "$SHARD" --staging "$STAGING" --work "$WORK" \
     --hands wilor --frame-stride 3 --require-hands 1; then
  echo "=== shard $SHARD DONE ==="; marker DONE done
else
  echo "=== shard $SHARD FAILED ==="; marker FAILED failed
fi
