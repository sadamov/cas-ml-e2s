# Instructor / HPC materials

Not needed to run the worksheet. Course participants use Google Colab (T4) and the
section-0 install cell; this directory is the CSCS Alps (GH200 / aarch64) path used
to regenerate `../cached_outputs/*.nc` and re-render the notebook before class.

- `storm_boris_ai_forecast_solutions.ipynb` - the fully worked, pre-rendered notebook.
  The root `../storm_boris_ai_forecast.ipynb` is byte-identical to this file except
  for the section-0 intro cell and six cells blanked to the `# TODO` hint that this
  file keeps above the worked answer: the `run.deterministic` call, `wrmse` +
  `persistence`, `set_domain`, `map_coords`, `number_of_samples`, `spread`.
  Edit either notebook, then port the change to the other so only those cells differ.
- `containers/` - GH200 container that runs Pangu24 + CorrDiff for real.
  - `cas-ml.dockerfile` - NGC PyTorch 26.01 base + natten (SM 9.0, from source),
    onnxruntime-gpu 1.26.0 (from source, no aarch64 wheel exists), earth2studio
    0.17.0, physicsnemo git pin; transformer-engine removed.
  - `submit_build.sh` -> `build.sbatch` -> `build.sh` - podman build + enroot import
    to `/capstor/store/cscs/mch/s83/sadamov/cas-ml-e2s/cas-ml.sqsh`.
  - `cas-ml.toml` - enroot `--environment` file.
  - `run_on_gpu.sh` - srun `storm_boris_ai_forecast_solutions.ipynb`
    with `FORCE_MODELS=1` (regenerates the cache and every rendered cell). `debug`
    queue, 1.5 h; port the six TODO cells back to the worksheet afterwards.
  - `start_jupyter.sh` - srun JupyterLab in the container on one GH200 (`debug`
    queue, 1.5 h) to run cells by hand. Note the node it prints, then in VSCode:
    Select Kernel > Existing Jupyter Server > the URL it prints (fresh token each run).
  - `patch_sqsh.sh` - apply a pip change to an existing `.sqsh` without a full rebuild.
- `preflight.py` - standalone copy of the notebook's "Instructor preflight" section
  (VRAM coexistence + diffusion cost probe). Keep `BBOX` in sync with the config cell in section 1.
- `requirements-local.txt` - `uv pip freeze` of the CSCS login-node `.venv` (the CPU
  / data path used for cache-replay renders).

Run from the repo root, e.g. `bash .instructor/containers/submit_build.sh`.

## What `cached_outputs/` holds

Three tiers, all written by a `FORCE_MODELS=1` render and all committed. The notebook
has three helpers in the "Model cells" setup cell, one per tier:

| Tier | Files | Helper | GPU run | CPU run |
| ---- | ----- | ------ | ------- | ------- |
| ERA5 / climatology slices | `era5_overview`, `era5_truth_*`, `era5_truth_check_*`, `wb2_clim_*` | `cached()` | cache | cache |
| CorrDiff model inputs | `corrdiff_in_*` | `cached_model_input()` | cache | unused |
| Inference outputs | `pangu_forecast`, `corrdiff_mean`, `corrdiff_hourly`, `corrdiff_diffusion` | `load_output()` / `save_output()` | rewritten | cache |

So: the shared ERA5 slices always come off disk regardless of `RUN_MODELS`; a GPU run
reruns the models from cached inputs and rewrites the outputs; a CPU run skips the
models and replays the outputs.

Two fetches are deliberately still live on the GPU path: the Pangu initial condition
(inside `run.deterministic`, not interceptable without restructuring the call) and the
single-hour CorrDiff input in the mean-vs-ERA5 comparison cell (routing it through
`cached_model_input` would swallow the `map_coords` TODO). Both are one time step, and
`corrdiff_hourly` covers the same hour, so ARCO's own on-disk cache absorbs the repeat.

Why the input cache exists at all: ARCO-ERA5 chunks are `(1, 721, 1440)` for surface
fields and `(1, 37, 721, 1440)` for pressure levels, i.e. **no spatial chunking**. A
29x45 window over the Alps costs the same download as the whole globe -- 4.2 MB per
surface variable per hour, 153.7 MB per pressure-level one. Cropped and cached that is
under a megabyte, so it belongs in git. There is no lat/lon slice that avoids this;
`interp_to` on `fetch_data` crops after the transfer.

Cache keys carry the parameters they depend on, so changing one misses rather than
silently serving the wrong fields: `RUN_TAG` (from `INIT` + `LEAD_HOURS`) on the ERA5
slices, `DOWNSCALE_TAG` (from `DOWNSCALE_RES` + the four `DOWNSCALE_BBOX` edges) plus
`N_HOURS` on the inputs. Exercise 1 moves `INIT`, and refetching there is correct.
