#!/usr/bin/env bash
# Build the cas-ml-e2s GH200 (aarch64) container and import it to a .sqsh.
#
#   podman build  - linux/arm64 image from .instructor/containers/cas-ml.dockerfile
#   enroot import - convert to .sqsh for pyxis / --environment
#
# Runs on a compute node (see submit_build.sh); podman + enroot are not to be
# used on the login nodes. Env vars:
#   OUTPUT        target .sqsh  (default: $CAS_ML_STORE/cas-ml.sqsh)
#   CAS_ML_STORE  output dir    (default: /capstor/store/cscs/mch/s83/sadamov/cas-ml-e2s)
#   ORT_VERSION   onnxruntime git tag passed as --build-arg (default: dockerfile's)
#   PLATFORM      (default: linux/arm64)
set -uo pipefail
IFS=$'\n\t'

cd "$(dirname "$0")/.."

# Do NOT default from $STORE: CSCS predefines it as the group store root
# (/capstor/store/cscs/mch/s83) and Slurm --export=ALL leaks it into the job.
CAS_ML_STORE="${CAS_ML_STORE:-/capstor/store/cscs/mch/s83/sadamov/cas-ml-e2s}"
OUTPUT="${OUTPUT:-${CAS_ML_STORE}/cas-ml.sqsh}"
PLATFORM="${PLATFORM:-linux/arm64}"
IMAGE_TAG="cas-ml-e2s:latest"
DOCKERFILE="containers/cas-ml.dockerfile"

# podman/buildah RUN steps build their rootfs under TMPDIR; it must be on a
# filesystem rootless user namespaces can mkdir into. Lustre (/iopsstor,
# /capstor) rejects that -> keep it on the same tmpfs as podman's graphroot.
# enroot import (~25 GB staging) is the only step that needs real scratch.
export TMPDIR="/dev/shm/${USER}/build-tmp"
SCRATCH="${SCRATCH:-/iopsstor/scratch/cscs/sadamov/tmp}"
mkdir -p "$TMPDIR" "$SCRATCH" "$CAS_ML_STORE"

command -v podman >/dev/null || { echo "podman not found" >&2; exit 1; }
command -v enroot >/dev/null || { echo "enroot not found" >&2; exit 1; }

BUILD_ARGS=()
[[ -n "${ORT_VERSION:-}" ]] && BUILD_ARGS+=(--build-arg "ORT_VERSION=${ORT_VERSION}")

echo "==> podman build (${PLATFORM}) ${DOCKERFILE} -> ${IMAGE_TAG}"
set -e
podman build --platform "${PLATFORM}" "${BUILD_ARGS[@]}" \
    -t "${IMAGE_TAG}" -f "${DOCKERFILE}" .
set +e

echo "==> enroot import -> ${OUTPUT}"
rm -f "${OUTPUT}"
# mksquashfs exits 1 on Lustre purely from "Unrecognised xattr prefix lustre.lov"
# warnings even when the image is written fine, so don't let that abort the job.
TMPDIR="${SCRATCH}" ENROOT_TEMP_PATH="${SCRATCH}" \
    enroot import -o "${OUTPUT}" "podman://localhost/${IMAGE_TAG}"
rc=$?

if unsquashfs -s "${OUTPUT}" >/dev/null 2>&1; then
    echo "==> done: ${OUTPUT}"
    ls -lh "${OUTPUT}"
    exit 0
fi
echo "enroot import failed (rc=$rc) and ${OUTPUT} is not a valid squashfs" >&2
exit 1
