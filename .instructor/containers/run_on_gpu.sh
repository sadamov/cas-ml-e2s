#!/usr/bin/env bash
# Execute the notebook inside the container on one GH200: runs Pangu + CorrDiff
# for real, writes cached_outputs/*.nc, and re-renders every cell. Commit the
# notebook + cached_outputs/ afterwards so no-GPU users can replay it.
#   bash .instructor/containers/run_on_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

srun --account=ab016 --partition=debug --nodes=1 --ntasks=1 \
     --gpus-per-task=1 --cpus-per-task=32 --mem=200G --time=01:30:00 \
     --environment="$(pwd)/containers/cas-ml.toml" \
     bash -lc '
set -e
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export FORCE_MODELS=1 EARTH2STUDIO_PACKAGE_TIMEOUT=1800
python - <<PY
import torch, onnxruntime, natten
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
# exercise a real NATTEN CUDA kernel: catches any torch<->natten ABI mismatch
q = torch.randn(1, 4, 8, 8, 16, device="cuda")
o = natten.functional.na2d(q, q, q, kernel_size=3)
print("natten", natten.__version__, "na2d ok", tuple(o.shape))
PY
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 .instructor/storm_boris_ai_forecast_solutions.ipynb
echo "=== cached_outputs/ ==="
ls -lh cached_outputs/
'
