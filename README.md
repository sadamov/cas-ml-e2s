# CAS ML - AI weather forecasting session

[storm_boris_ai_forecast.ipynb](storm_boris_ai_forecast.ipynb) is a one-hour Google Colab
session built for a free-tier **T4 GPU**. It runs a **Pangu-Weather** daily forecast of Storm
Boris (September 2024) from ERA5, verifies it, and downscales ERA5 to **2.2 km, hourly** over
the Alps with CorrDiff -- including a small diffusion-sampled ensemble.

The notebook at the repo root is the **worksheet**: six code cells are blanked to a `# TODO`
plus a hint for participants to complete (the rollout call, latitude-weighted RMSE, the
persistence baseline, `set_domain`, `map_coords`, the ensemble size, the spread map). The
fully worked, pre-rendered version is
[`.instructor/storm_boris_ai_forecast_solutions.ipynb`](.instructor/storm_boris_ai_forecast_solutions.ipynb).

Slimmed down from [ai-models-ensembles](ai-models-ensembles/), reusing the same
earth2studio patterns (registry -> data source -> rollout -> verification) with the HPC,
Slurm, container and perturbation machinery stripped out.

Course-facing files are the worksheet notebook, this README, [GLOSSARY.md](GLOSSARY.md) (every
abbreviation used anywhere in the repo) and `cached_outputs/` (the netCDF fallback for
participants without a GPU). The solutions notebook and everything used to regenerate the cache
on CSCS Alps (GH200 container, Slurm build, preflight probe, local pins) live in
[`.instructor/`](.instructor/) and are not needed to run the worksheet.

## Session structure

| Section | Content | Budget |
|---|---|---|
| 0 | Runtime check, `earth2studio` install | 10 min |
| 1 | ERA5 from ARCO, synoptic overview of the storm | 5 min |
| 2 | Pangu-Weather (`Pangu24`), 6-step daily rollout | 15 min |
| 3 | RMSE / ACC against persistence and climatology | 10 min |
| 4 | CorrDiff ERA5 -> COSMO-REA2, hourly, mean + diffusion ensemble | 20 min |



## Model history (why Pangu24, not something else)

The notebook went through several global models before landing on `Pangu24`. Worth knowing so a
future "why not just use X" doesn't re-open a closed question:

- **FourCastNet 3** -- ruled out, *measured*: peaked at 13.22 GB against 14.56 GB usable on a T4,
  OOMing inside `torch_harmonics`'s DISCO convolution (an `einsum` over the full-globe latent,
  independent of whether the optimised CUDA kernels are built). No domain-cropping knob exists
  upstream of it, unlike CorrDiff -- FCN3's operators are spherical harmonic transforms, which are
  inherently global.
- **Atlas** -- ruled out on weights alone: the HF repo is 31.25 GB, and the lightest usable
  regression+autoencoder pair is still ~17.8 GB before a single activation.
- **AIFS v2** -- ruled out twice over: `flash-attn` (a hard dependency of the `aifs2` extra) has no
  Turing support upstream (`flash-attn` supports Ampere/Ada/Hopper only), and separately,
  earth2studio's CDS lexicon still lacks the wave-period-band heights (`h1012`-`h2530`) AIFS v2
  needs -- the same gap already recorded in
  [`e2s_models.py`](ai-models-ensembles/ai_models_ensembles/e2s_models.py) for this repo's own
  pipeline.
- **FengWu** -- *measured*: OOMs inside `onnxruntime`'s BFC arena on the same T4.
- **SFNO** -- *measured to fit* (before Pangu24 was tried), but its checkpoint is 6.4 GB against
  Pangu24's 1.1 GB. With ~30 students each pulling that over one lecture-room network, checkpoint
  size became a real classroom constraint, not just a VRAM one -- that tipped the choice to Pangu.
- **Pangu24** -- *measured to fit*: 12.37 GB peak (2-step probe), loads in under 2 minutes, ~8 s
  per 24 h step. Deterministic (ONNX, no RNG to re-seed), which is why the ensemble in this
  notebook lives entirely in section 4's CorrDiff diffusion sampler rather than in section 2.

## torch-harmonics and makani are gone

Earlier drafts installed `torch-harmonics` (compiled from source, since it ships no cp313 wheel
anywhere: cp310-cp312 only on PyPI, no sdist there or on `pypi.nvidia.com`) and `makani`, both
needed only by FCN3/SFNO. Since nothing in the current notebook calls either model, both
dependencies were dropped.

This is safe by construction, not by luck: `earth2studio.models.px.sfno`/`fcn3` wrap their
`makani`/`torch-harmonics` imports in `try: ... except ImportError: OptionalDependencyFailure(...)`,
so `from earth2studio import run` (which eagerly imports every model in `models.px`) succeeds
without either package installed -- the failure would only surface if `SFNO.load_model()` or
`FCN3.load_model()` were actually called, which they no longer are. Confirmed by reading the
source at the pinned `earth2studio==0.17.0`, and verified end-to-end on a GH200: the shorter
install list imports cleanly and the full notebook runs. The one place this could still bite is
Colab-specific drift in the section-0 install (see below).

## Run this once before the class

The full pipeline (Pangu24 rollout -> verification -> CorrDiff mean + hourly + 8-member diffusion
ensemble) has been run end to end on a CSCS GH200, and `cached_outputs/` is the result of that
run, so a no-GPU participant can replay every figure. What has *not* been re-checked is a free
Colab **T4** specifically, so do a dry run there the day before. Things that behave differently on
a T4 than on the GH200 the cache came from:

1. **Do Pangu24 and CorrDiff (both modes) coexist in one kernel on 16 GB?** On the GH200 (96 GB)
   this is a non-issue; on a T4 it is tight. Confirmed pieces:
   - Pangu24 peaks at 12.37 GB (2-step probe) and **`onnxruntime` hands its memory back cleanly**
     on `del` + `gc.collect()` + `torch.cuda.empty_cache()` -- measured: 12.37 GB -> 0.13 GB -- so
     loading CorrDiff afterwards in the same kernel is not blocked by ORT's arena.
   - The `nvidia-physicsnemo` git pin (`ced75d93...`, cell 5) fixes the CorrDiff `set_domain`
     RoPE-table crash (`Expected rope_cos/rope_sin of shape (H, W, 64)`): the PyPI 2.2.0 release
     prebuilds the RoPE tables from the per-call `attn_kwargs`, but `set_domain`'s rebind mutates
     `attn_kwargs_forward`, which the refactored code no longer reads. The pinned revision predates
     that refactor. Verified on the GH200 run; `.instructor/preflight.py` reproduces the check.
   - Diffusion-mode cost: 18 sampler steps x `N_SAMPLES` (8 by default). On the GH200, 8 samples
     over the cropped `DOWNSCALE_BBOX` took ~7 min. On a T4 this is the most likely thing to OOM or
     overrun the budget -- drop to `resolution="rea6"` (6 km, smaller latent) and/or a lower
     `N_SAMPLES`, or just replay the committed cache.
2. **Is there a `natten` wheel for Colab's current torch?** Section 0 resolves the wheel URL from
   <https://whl.natten.org/> at install time (matching torch, CUDA and Python ABI) rather than
   pinning a version, so it self-adjusts to most Colab drift -- but not all of it. NATTEN
   publishes wheels only for the **two most recent** official torch builds and requires
   **torch >= 2.8** since 0.21.5; as of the last check it has cp313 wheels for torch 2.8-2.13 on
   cu126/cu128/cu129/cu130/cu132, but not cu121 or cu124. The kernels themselves are fine on a T4
   -- NATTEN's CUTLASS FNA/FMHA backend explicitly targets SM50/SM70/**SM75 (Turing)**/SM80, so
   this is a supported device, not a fallback path. If the wheel resolution genuinely fails (no
   match for the installed torch/CUDA), the fallback is `pip install natten` with no local version
   tag, which gives the pure-PyTorch Flex Attention backend (also sm_75-capable, slower).

Also worth timing on the day: Pangu24's checkpoint is 1.1 GB from HuggingFace; CorrDiff's `rea2`
package is roughly 1 GB combined (regression ~373 MB + diffusion ~663 MB, both observed in an
earlier run's download log), plus small invariant/grid files. On a slow room network, downloads
still dominate the section-0/4 budget more than compute does.

## Design decisions worth knowing

**Why no second global model / no WeatherBench2 baselines.** The precomputed GraphCast / Pangu /
Aurora / IFS-ENS forecasts in WeatherBench 2 **stop at 2022**, and Boris is 2024.
(`gs://weatherbench2/datasets/ifs_ens/2016-2024-*` looks like an exception but is a broken store
with no root group metadata.) So the baselines are persistence and the WB2 1990-2019 climatology,
indexed by day-of-year and hour and therefore valid for any date. Moving the case to 2022 (e.g.
Storm Eunice, 16-18 February) would unlock the full WB2 lineup for multi-model intercomparison, at
the cost of the Alpine-flood framing.

**Why physicsnemo must be the pinned git revision, not the PyPI release of the same version
number.** See "Run this once before the class" above -- a released-version-number match does not
imply code identity, and this cost real debugging time to discover. Treat any earth2studio
`[tool.uv.sources]` pin as authoritative over what looks like an equivalent PyPI release.

**Why CorrDiff still downscales the ERA5 analysis, not Pangu's own forecast.** `CorrDiffCosmoEra5`
needs 47 ERA5 inputs including surface pressure `sp` and the 100 m wind components; Pangu's 69
outputs (winds/temperature/geopotential on 13 levels, plus `msl`/`u10m`/`v10m`/`t2m`) include none
of the three. Chaining them needs `earth2studio.models.dx.DerivedSurfacePressure` for `sp`, and the
100 m winds would need to be dropped from CorrDiff's input list entirely (no clean substitute).
Left as exercise 4 rather than hidden in the main path, because the variable bookkeeping is
the actual lesson -- and it was exactly this kind of gap (FCN3 lacked `sp` too) that first showed
why "just chain the models" is never a one-liner in practice.

**Why section 4 runs hourly, both modes, at the finest resolution.** ARCO-ERA5 is native hourly, so
downscaling six consecutive hours costs one extra forward call (mean mode is cheap) and shows real
temporal texture rather than a single frozen instant. Diffusion mode is reserved for a single hour
(the same `DOWNSCALE_TIME` peak used elsewhere) specifically because it is expensive -- 18 sampler
steps per sample -- so combining "every hour" with "every sample" was cut on time-budget grounds,
not technical ones; that combination is exercise 3.

**Why the domain sits over the eastern Alps and Bohemia rather than Switzerland alone.** Boris's
extreme rainfall fell mostly over Czechia, Austria and Poland, not Switzerland. The current
`DOWNSCALE_BBOX` (45.9-49.1 N, 12.2-19.6 E) frames the northern-Alps rim and that rainfall
band; the northern 40% of an earlier, taller box was cropped because the `rea2` diffusion
ensemble cost scales with the cell count (a full `rea2` run over the taller box was ~19 min on a
GH200, roughly hours on a T4). Narrowing it to Switzerland is a one-line change and a fine live
demo, but shows a less dramatic slice of the actual event.

**Why the `Cache` wrapper.** Not about ensemble members any more (Pangu is a single deterministic
run) -- it exists so the same ERA5 slice (e.g. the forecast initial condition) is not re-fetched
from GCS every time later cells touch the same time/variable combination.

## Knobs

All in the configuration cell in section 1:

- `QUICK = True` drops to a 72 h forecast (3 daily steps) and a 4-member diffusion ensemble
  (`N_SAMPLES, N_SHOW = 4, 2`), for a first test run.
- `INIT`, `LEAD_HOURS` - the Pangu forecast experiment (`NSTEPS = LEAD_HOURS // 24`).
- `LAT_RANGE`, `LON_RANGE` - the verification and plotting window.
- `DOWNSCALE_RES` - `"rea2"` (2.2 km) or `"rea6"` (6 km, cheaper -- useful if diffusion mode is
  too slow/memory-heavy on the day).
- `DOWNSCALE_TIME` - the anchor hour (peak of the event); `DOWNSCALE_TIMES` (the six-hour sequence
  for the temporal-resolution demo) and the diffusion-ensemble hour are both derived from it, not
  set independently.
- `DOWNSCALE_BBOX` - the downscaling crop, over the eastern Alpine rim and Bohemia. Its southern
  and eastern edges sit in COSMO-REA2's extended margin, so `set_domain` prints a one-time
  out-of-distribution warning (not fatal). Pushing much further south or east eventually leaves the
  `rea2` footprint and errors; shrinking it is always safe and makes the diffusion ensemble
  cheaper.
- `N_SAMPLES` - CorrDiff diffusion-ensemble size (section 4). Cost scales linearly with it (each
  sample is a full 18-step sampler run).
