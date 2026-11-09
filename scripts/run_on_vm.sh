#!/bin/bash
# Autonomous VM driver for the 10-hr sample. Runs as the instance startup-script.
# NO-IDLE GUARANTEE: on ANY exit (success, failure, crash) the VM deletes itself;
# a 16h safety timer stops it even if this script hangs. The VM is never idle-billing.
set -uo pipefail
export HOME=/root
export DEBIAN_FRONTEND=noninteractive
export PATH=/opt/venv/bin:/opt/conda/bin:/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

OUT="gs://annotationportal-raw/lerobot_sample_10hr"
SRC1="gs://ots-egocentric-inventory/Garment_Factory_workbench/2026-04-28/1102"
SRC2="gs://ots-egocentric-inventory/Garment_Factory_workbench/2026-04-28/1107"
WORK=/home/work
PYBIN=/opt/venv/bin/python
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
  marker "EXIT_$rc" "exit $rc"; pushlog
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || shutdown -h now
}
trap cleanup EXIT
( sleep 57600; echo "SAFETY TIMEOUT"; shutdown -h now ) &

echo "=== STARTED $(date -u) on $NAME/$ZONE ==="
marker STARTED started
nvidia-smi || echo "WARN: nvidia-smi failed"

echo "=== system deps ==="
apt-get update -y && apt-get install -y python3-pip python3-venv ffmpeg git

echo "=== python venv (self-contained; independent of the image layout) ==="
python3 -m venv /opt/venv
"$PYBIN" -m pip install --upgrade pip wheel
"$PYBIN" -m pip install torch torchvision          # Linux x86_64 wheels bundle CUDA
"$PYBIN" -m pip install opencv-python-headless numpy pyarrow pandas
"$PYBIN" -m pip install "git+https://github.com/warmshao/WiLoR-mini.git"
"$PYBIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== clone repo ==="
cd /home && rm -rf repo && git clone --depth 1 https://github.com/atulyadav-mytron/LeRobot.git repo

echo "=== fetch MANO ==="
mkdir -p /opt/mano "$HOME/.cache/wilor_mini/mano_data"
gcloud storage cp "$OUT/../mano/MANO_RIGHT.pkl" /opt/mano/ 2>/dev/null \
  || gcloud storage cp "gs://annotationportal-raw/mano/MANO_RIGHT.pkl" /opt/mano/ || true
gcloud storage cp "gs://annotationportal-raw/mano/MANO_LEFT.pkl" /opt/mano/ || true
cp /opt/mano/MANO_*.pkl "$HOME/.cache/wilor_mini/mano_data/" 2>/dev/null || true

RB=/home/repo/scripts/run_batch.py
echo "=== VALIDATE on 2 clips ==="
if ! "$PYBIN" "$RB" --src "$SRC1" --src "$SRC2" --out "$OUT/_validate" \
     --work "$WORK/val" --mano-dir /opt/mano --hands wilor --limit 2; then
  echo "=== VALIDATION FAILED -- aborting before full run ==="; marker VALIDATION_FAILED failed; exit 1
fi
marker VALIDATED ok

echo "=== FULL RUN (all clips) ==="
"$PYBIN" "$RB" --src "$SRC1" --src "$SRC2" --out "$OUT" \
  --work "$WORK/full" --mano-dir /opt/mano --hands wilor
echo "=== DONE $(date -u) ==="
marker DONE done
