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

echo "=== system deps (libgl* for opencv; ffmpeg for video) ==="
set -e   # from here, any failed install HALTS the script (-> self-delete, no silent-broken run)
apt-get update -y
apt-get install -y python3-pip python3-venv git build-essential ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6

echo "=== python venv + validated WiLoR recipe ==="
python3 -m venv /opt/venv
"$PYBIN" -m pip install --upgrade pip setuptools wheel Cython numpy dill
# chumpy (a WiLoR-mini dep) only builds with --no-build-isolation; this also pulls
# torch (CUDA), opencv, ultralytics, timm, smplx, etc. MANO auto-downloads from HF.
"$PYBIN" -m pip install --no-build-isolation "git+https://github.com/warmshao/WiLoR-mini"
"$PYBIN" -m pip install pyarrow        # extra needed by our pipeline
"$PYBIN" -c "import wilor_mini, cv2, pyarrow, torch; print('deps OK; cuda', torch.cuda.is_available())"
set +e  # back to tolerant mode for the run itself

echo "=== clone repo ==="
cd /home && rm -rf repo && git clone --depth 1 https://github.com/atulyadav-mytron/LeRobot.git repo

RB=/home/repo/scripts/run_batch.py
echo "=== VALIDATE on 2 clips (require >=1 real hand detection) ==="
if ! "$PYBIN" "$RB" --src "$SRC1" --src "$SRC2" --out "$OUT/_validate" \
     --work "$WORK/val" --hands wilor --limit 2 --require-hands 1; then
  echo "=== VALIDATION FAILED -- aborting before full run ==="; marker VALIDATION_FAILED failed; exit 1
fi
marker VALIDATED ok

echo "=== FULL RUN (all clips) ==="
"$PYBIN" "$RB" --src "$SRC1" --src "$SRC2" --out "$OUT" \
  --work "$WORK/full" --hands wilor
echo "=== DONE $(date -u) ==="
marker DONE done
