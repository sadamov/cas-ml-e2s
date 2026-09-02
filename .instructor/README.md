# Instructor / HPC materials

Not needed to run the worksheet. Course participants use Google Colab (T4) and the
section-0 install cell; this directory is the CSCS Alps (GH200 / aarch64) path used
to regenerate `../cached_outputs/*.nc` and re-render the notebook before class.

- `storm_boris_ai_forecast_solutions.ipynb` - the fully worked, pre-rendered notebook.
  The root `../storm_boris_ai_forecast.ipynb` is byte-identical to this file except
  for six cells blanked to a `# TODO` plus a hint: the `run.deterministic` call,
  `wrmse` + `persistence`, `set_domain`, `map_coords`, `number_of_samples`, `spread`.
  Edit either notebook, then port the change to the other so only those six cells differ.
- `containers/` - GH200 container that runs Pangu24 + CorrDiff for real.
  - `cas-ml.dockerfile` - NGC PyTorch 26.01 base + natten (SM 9.0, from source),
    onnxruntime-gpu 1.26.0 (from source, no aarch64 wheel exists), earth2studio
    0.17.0, physicsnemo git pin; transformer-engine removed.
  - `submit_build.sh` -> `build.sbatch` -> `build.sh` - podman build + enroot import
    to `/capstor/store/cscs/mch/s83/sadamov/cas-ml-e2s/cas-ml.sqsh`.
  - `cas-ml.toml` - enroot `--environment` file.
  - `run_on_gpu.sh` - srun the notebook with `FORCE_MODELS=1` (regenerates the cache
    and every rendered cell).
  - `patch_sqsh.sh` - apply a pip change to an existing `.sqsh` without a full rebuild.
- `preflight.py` - standalone copy of the notebook's "Instructor preflight" section
  (VRAM coexistence + diffusion cost probe). Keep `BBOX` in sync with the config cell in section 1.
- `requirements-local.txt` - `uv pip freeze` of the CSCS login-node `.venv` (the CPU
  / data path used for cache-replay renders).

Run from the repo root, e.g. `bash .instructor/containers/submit_build.sh`.
