# GH200 (aarch64) container for storm_boris_ai_forecast.ipynb: Pangu24 + CorrDiff.
#
# Base: NGC PyTorch 26.01 (torch 2.10.0a0, Python 3.12, CUDA 13.1, cuDNN 9.17,
# TensorRT 10.14). Same base as ai-models-ensembles/containers/atlas.dockerfile,
# which is where the natten-for-SM90 build pattern below comes from.
#
# The notebook's section-0 install list targets Colab (x86_64, cp313): its
# onnxruntime-gpu==1.26.0 wheel and the whl.natten.org wheel are x86_64-only, so
# both are built from source here against the container's CUDA stack instead.
#
# Build:  bash .instructor/containers/submit_build.sh          (sbatch -> cas-ml.sqsh)
# Run:    srun --environment=./.instructor/containers/cas-ml.toml ...
FROM nvcr.io/nvidia/pytorch:26.01-py3

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_CONSTRAINT= \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PACKAGES=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    CUDA_HOME=/usr/local/cuda \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EARTH2STUDIO_CACHE=/workspace/.cache/earth2studio \
    EARTH2STUDIO_PACKAGE_TIMEOUT=1800

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl cmake ninja-build libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 1. natten for GH200 / SM 9.0 (Hopper), compiled from source. Verbatim from
#    atlas.dockerfile: whl.natten.org publishes x86_64 wheels only.
# ---------------------------------------------------------------------------
ENV NATTEN_CUDA_ARCH="9.0" \
    NATTEN_WITH_CUDA=1 \
    FORCE_CUDA_EXTENSION=1
RUN uv pip install --system --break-system-packages \
        hatchling ninja Cython wheel_stub py-cpuinfo setuptools-scm
RUN pip install --no-build-isolation --break-system-packages natten==0.21.5

# ---------------------------------------------------------------------------
# 2. earth2studio + the rest of the notebook's section-0 list. physicsnemo is
#    pinned to the exact revision earth2studio's [tool.uv.sources] uses for the
#    cosmo extra (predates the DiT RoPE-table refactor that breaks
#    CorrDiffCosmoEra5.set_domain on cropped domains). onnxruntime-gpu / natten
#    / torch are handled elsewhere, so they are left out of this line.
#    Plain pip (not uv): it accepts the container's prerelease torch 2.10.0a0 as
#    satisfying earth2studio's `torch>=` and leaves the NGC build untouched;
#    uv's resolver would treat the prerelease as unsatisfying and pull a wheel.
# ---------------------------------------------------------------------------
RUN pip install --no-build-isolation --break-system-packages \
        "earth2studio==0.17.0" \
        "nvidia-physicsnemo @ git+https://github.com/NVIDIA/physicsnemo.git@ced75d93d014f70bb691372788eee2d201171c12" \
        "einops>=0.8.1" nvtx onnx==1.21.0 \
        cartopy matplotlib \
        ipykernel jupyterlab \
 && python -c "import torch; print('torch', torch.__version__)"

# earth2studio's deps (tensordict/timm/physicsnemo) pull a fresh torch from PyPI
# (2.13.0+cu130), replacing the NGC build. That leaves NGC's prebuilt
# transformer_engine linked against the old libtorch ABI
# (undefined symbol: c10::impl::cow::materialize_cow_storage), and physicsnemo
# imports it, which breaks earth2studio.models.dx.corrdiff. CorrDiff inference
# does not need transformer_engine, and rebuilding it against torch 2.13 drags in
# transformer_engine_cu13 + a newer cudnn-frontend that breaks torch itself, so
# the fix is simply to drop it.
RUN pip uninstall -y --break-system-packages \
        transformer-engine transformer-engine-torch \
        transformer_engine transformer_engine_torch || true \
 && python -c "from earth2studio.models.dx import CorrDiffCosmoEra5; from earth2studio.models.px import Pangu24; print('earth2studio model imports OK')"

# ---------------------------------------------------------------------------
# 3. onnxruntime-gpu from source (no aarch64 wheel exists on PyPI, any version).
#    Built against the container's CUDA 13.1 / cuDNN 9 for sm_90 only. Kept as
#    the last layer: it is the long pole (~60-90 min) and the most likely to
#    need iteration, so a rebuild does not disturb layers 1-2.
# ---------------------------------------------------------------------------
ARG ORT_VERSION=v1.26.0
RUN git clone --branch ${ORT_VERSION} --recursive \
        https://github.com/microsoft/onnxruntime.git /opt/onnxruntime \
 && cd /opt/onnxruntime \
 && python tools/ci_build/build.py \
        --build_dir build/Linux --config Release \
        --parallel 32 \
        --skip_tests --skip_submodule_sync \
        --build_wheel --allow_running_as_root \
        --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr \
        --cmake_extra_defines \
            CMAKE_CUDA_ARCHITECTURES=90 \
            onnxruntime_BUILD_UNIT_TESTS=OFF \
 && pip install --break-system-packages "$(find build -name 'onnxruntime_gpu-*.whl' | head -1)" \
 && cd / && rm -rf /opt/onnxruntime ~/.cache

RUN mkdir -p ${EARTH2STUDIO_CACHE}
WORKDIR /workspace
CMD ["python", "-c", "import torch, natten, onnxruntime, earth2studio; from earth2studio.models.px import Pangu24; from earth2studio.models.dx import CorrDiffCosmoEra5; print('cuda', torch.cuda.is_available(), '| ort', onnxruntime.__version__, onnxruntime.get_available_providers())"]
