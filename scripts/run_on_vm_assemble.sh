#!/bin/bash
# Assemble VM (cheap CPU, no GPU/WiLoR): pull every shard's per-clip artifacts
# from staging, build the single v3.0 dataset, upload. Self-deletes on any exit.
set -uo pipefail
export HOME=/root DEBIAN_FRONTEND=noninteractive
export PATH=/opt/venv/bin:/snap/bin:/usr/local/bin:/usr/bin:/bin:$PATH

BASE="gs://annotationportal-raw/lerobot_sample_10hr"
md() { curl -s -f -H "Metadata-Flavor: Google" "http://metadata/computeMetadata/v1/instance/attributes/$1" 2>/dev/null; }
STAGING="$(md staging)"; [ -z "$STAGING" ] && STAGING="$BASE/_staging"
OUT="$(md outdir)"; [ -z "$OUT" ] && OUT="$BASE"
WORK=/home/work ; PYBIN=/opt/venv/bin/python ; LOG=/var/log/lerobot_assemble.log
mkdir -p "$WORK" ; exec > >(tee -a "$LOG") 2>&1

NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/name)
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
marker() { echo "$2 $(date -u)" >/tmp/_m; gcloud storage cp /tmp/_m "$BASE/_logs/assemble_$1" 2>/dev/null || true; }
cleanup() { local rc=$?; marker "EXIT_$rc" "exit $rc";
  gcloud storage cp "$LOG" "$BASE/_logs/assemble.log" 2>/dev/null || true
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || shutdown -h now; }
trap cleanup EXIT
( sleep 25200; shutdown -h now ) &   # 7h safety (waits for shards + assemble)

echo "=== ASSEMBLE STARTED $(date -u) ==="; marker STARTED started
set -e
apt-get update -y
apt-get install -y python3-pip python3-venv git ffmpeg libgl1 libglib2.0-0
python3 -m venv /opt/venv
"$PYBIN" -m pip install --upgrade pip wheel
"$PYBIN" -m pip install numpy pandas pyarrow opencv-python-headless
set +e
cd /home && rm -rf repo && git clone --depth 1 https://github.com/atulyadav-mytron/LeRobot.git repo

# Coordinator: wait for all 4 produce shards to finish (self-coordinating, no
# external session needed). Tolerates a FAILED shard and a hard deadline so the
# dataset is still assembled from whatever completed.
echo "=== WAIT FOR SHARDS ==="
DEADLINE=$(( $(date +%s) + 18000 ))   # 5h hard cap
while true; do
  mk=$(gcloud storage ls "$BASE/_logs/" 2>/dev/null | sed 's#.*/##')
  dn=$(echo "$mk" | grep -cE "shard[0-3]_4_DONE")
  en=$(echo "$mk" | grep -cE "shard[0-3]_4_(DONE|FAILED)")
  echo "  shards done=$dn ended=$en $(date -u)"
  [ "$dn" -ge 4 ] && { echo "all 4 shards DONE"; break; }
  [ "$en" -ge 4 ] && { echo "all shards ended (some FAILED); assembling the DONE ones"; break; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "deadline reached; assembling whatever is available"; break; }
  sleep 60
done
marker SHARDS_READY ready
echo "=== BUILD + UPLOAD ==="
if "$PYBIN" /home/repo/scripts/run_batch.py --src "$STAGING" --mode assemble \
     --staging "$STAGING" --out "$OUT" --work "$WORK"; then
  echo "=== ASSEMBLE DONE ==="; marker DONE done
else
  echo "=== ASSEMBLE FAILED ==="; marker FAILED failed
fi
