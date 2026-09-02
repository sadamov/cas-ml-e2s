#!/usr/bin/env bash
# Run a command inside an existing .sqsh and re-export it, without a full podman
# rebuild (which would recompile onnxruntime from source, ~70 min). Compute node
# only. Any network use inside CMD needs SSL_CERT_FILE (set below).
#
#   sbatch --account=ab016 --partition=normal -N1 -n1 --cpus-per-task=32 \
#          --mem=200G --time=00:40:00 --output=logs/patch_%j.log --wrap \
#     "bash .instructor/containers/patch_sqsh.sh IN.sqsh OUT.sqsh 'pip uninstall -y --break-system-packages transformer-engine transformer-engine-torch'"
set -euo pipefail

IN="${1:?usage: patch_sqsh.sh <in.sqsh> <out.sqsh> <bash -lc command>}"
OUT="${2:?}"
CMD="${3:?}"

export ENROOT_DATA_PATH="/dev/shm/${USER}/enroot-data"
export ENROOT_TEMP_PATH="/iopsstor/scratch/cscs/sadamov/tmp"
export ENROOT_CACHE_PATH="/iopsstor/scratch/cscs/sadamov/tmp"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"

name="casmlpatch_${SLURM_JOB_ID:-$$}"
trap 'enroot remove -f "$name" 2>/dev/null || true' EXIT

echo "==> enroot create $name from $IN"
enroot create --name "$name" "$IN"

echo "==> running inside container: $CMD"
enroot start --root --rw --mount /iopsstor --mount /capstor "$name" bash -lc "
    export SSL_CERT_FILE=\$(python -c 'import certifi; print(certifi.where())')
    set -x
    $CMD
"

echo "==> enroot export -> $OUT"
rm -f "$OUT"
enroot export --output "$OUT" "$name"
ls -lh "$OUT"
