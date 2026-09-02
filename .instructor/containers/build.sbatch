#!/usr/bin/env bash
#SBATCH --job-name=build_cas-ml
#SBATCH --account=ab016
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_cas-ml_%j.log
#SBATCH --error=logs/build_cas-ml_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "host $(hostname)  start $(date -Is)"
bash containers/build.sh
echo "end $(date -Is)"
