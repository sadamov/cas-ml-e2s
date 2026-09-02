#!/usr/bin/env bash
# Submit the container build to a GH200 compute node.
#   bash .instructor/containers/submit_build.sh                 # normal partition, 4h
#   ORT_VERSION=v1.23.2 bash .instructor/containers/submit_build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

EXPORTS="ALL"
[[ -n "${ORT_VERSION:-}" ]]   && EXPORTS="${EXPORTS},ORT_VERSION=${ORT_VERSION}"
[[ -n "${CAS_ML_STORE:-}" ]]  && EXPORTS="${EXPORTS},CAS_ML_STORE=${CAS_ML_STORE}"

jid=$(sbatch --parsable --export="${EXPORTS}" containers/build.sbatch)
echo "submitted job ${jid}"
echo "log: logs/build_cas-ml_${jid}.log"
echo "watch: tail -f logs/build_cas-ml_${jid}.log"
