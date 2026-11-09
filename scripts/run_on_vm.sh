#!/bin/bash
# Autonomous VM driver for the 10-hr sample. Runs as the instance startup-script.
# NO-IDLE GUARANTEE: on ANY exit (success, failure, crash) the VM deletes itself;
# a 16h safety timer stops it even if this script hangs. The VM is never idle-billing.
set -uo pipefail
export PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export DEBIAN_FRONTEND=noninteractive

OUT="gs://narration-raw-delivery/lerobot_sample_10hr"
SRC1="gs://ots-egocentric-inventory/Garment_Factory_workbench/2026-04-28/1102"
SRC2="gs://ots-egocentric-inventory/Garment_Factory_workbench/2026-04-28/1107"
WORK=/home/work
LOG=/var/log/lerobot_run.log
mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/name)
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')

marker() { echo "$2 $(date -u)" >/tmp/_m; gcloud storage cp /tmp/_m "$OUT/_logs/$1" 2>/dev/null || true; }
pushlog() { gcloud storage cp "$LOG" "$OUT/_logs/startup.log" 2>/dev/null || true; }

cleanup() {
  local rc=$?
  echo "=== cleanup (exit $rc); deleting VM $NAME in $ZONE ==="
  marker "EXIT_$rc" "exit $rc"
  pushlog
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet \
    || sudo shutdown -h now
}
trap cleanup EXIT

# Safety: hard stop after 16h regardless of state.
( sleep 57600; echo "SAFETY TIMEOUT"; sudo shutdown -h now ) &

echo "=== STARTED $(date -u) on $NAME/$ZONE ==="
marker STARTED started
nvidia-smi || echo "WARN: no GPU visible yet"

echo "=== install deps ==="
sudo apt-get update -y && sudo apt-get install -y ffmpeg git
python -m pip install --upgrade pip
# torch for CUDA 12.4 (matches the common-cu124 image); skip if already present
python -c "import torch" 2>/dev/null || \
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install opencv-python-headless numpy pyarrow pandas
python -m pip install "git+https://github.com/warmshao/WiLoR-mini.git"

echo "=== clone repo ==="
cd /home
rm -rf repo
git clone --depth 1 https://github.com/atulyadav-mytron/LeRobot.git repo

echo "=== fetch MANO ==="
mkdir -p /opt/mano
gcloud storage cp "gs://narration-raw-delivery/mano/MANO_RIGHT.pkl" /opt/mano/ || true
gcloud storage cp "gs://narration-raw-delivery/mano/MANO_LEFT.pkl"  /opt/mano/ || true
# place MANO where WiLoR-mini commonly looks, too
mkdir -p "$HOME/.cache/wilor_mini/mano_data" /opt/conda/lib/mano
cp /opt/mano/MANO_*.pkl "$HOME/.cache/wilor_mini/mano_data/" 2>/dev/null || true
cp /opt/mano/MANO_*.pkl /opt/conda/lib/mano/ 2>/dev/null || true

RB=/home/repo/scripts/run_batch.py
echo "=== VALIDATE on 2 clips ==="
if ! python "$RB" --src "$SRC1" --src "$SRC2" --out "$OUT/_validate" \
     --work "$WORK/val" --mano-dir /opt/mano --hands wilor --limit 2; then
  echo "=== VALIDATION FAILED -- aborting before full run ==="
  marker VALIDATION_FAILED failed
  exit 1
fi
marker VALIDATED ok

echo "=== FULL RUN (all clips) ==="
python "$RB" --src "$SRC1" --src "$SRC2" --out "$OUT" \
  --work "$WORK/full" --mano-dir /opt/mano --hands wilor
echo "=== DONE $(date -u) ==="
marker DONE done
