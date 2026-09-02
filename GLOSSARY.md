# Glossary

Abbreviations used in the notebook and README, grouped by theme. Variable shorthands
(`z500`, `tp`, ...) are near the end.

## Course and institutions

- **DWD** - Deutscher Wetterdienst, the German national weather service; produces COSMO-REA.
- **ECMWF** - European Centre for Medium-Range Weather Forecasts; source of ERA5, IFS and AIFS.
- **NVIDIA** - vendor of the GPUs and of earth2studio, CorrDiff, FourCastNet and NATTEN.

## Weather data and physics-based models

- **NWP** - Numerical Weather Prediction; forecasting by integrating the discretised equations of atmospheric physics.
- **ERA5** - ECMWF Reanalysis v5; a global hourly reconstruction of the atmosphere from 1940 to present at ~28 km, used here as both initial condition and verification truth.
- **Reanalysis** - a physically consistent past atmospheric state produced by assimilating historical observations into a fixed model.
- **ARCO** (ARCO-ERA5) - Analysis-Ready, Cloud-Optimized ERA5; Google's public Zarr copy of ERA5 on GCS, native hourly.
- **IC** - Initial Condition; the atmospheric state a forecast starts from.
- **IFS** - Integrated Forecasting System; ECMWF's operational physics-based model.
- **IFS-ENS** / **ENS** - the 50-member IFS ensemble.
- **COSMO** - Consortium for Small-scale Modelling; the limited-area NWP model behind COSMO-REA.
- **COSMO-REA** (`rea2`, `rea6`) - COSMO Regional Reanalysis over Europe; the high-resolution target CorrDiff maps ERA5 onto, at 2.2 km (`rea2`) or 6 km (`rea6`).
- **WB2** / **WeatherBench 2** - a benchmark dataset and scoring suite for data-driven weather forecasting.
- **CDS** - Copernicus Climate Data Store; ECMWF's data portal, and the name of its variable lexicon in earth2studio.
- **Cut-off low** - an upper-level cold low pinched off from the jet stream and left with no steering flow, so it stalls; what made Boris sit over the region for four days.

## AI weather models

- **Pangu-Weather** (`Pangu24`, `Pangu6`, `Pangu3`) - Huawei's transformer-based global AI model (Bi et al., 2023); the number is the autoregressive step in hours. This session uses `Pangu24`.
- **CorrDiff** - Correction (or corrector) Diffusion; NVIDIA's generative downscaler: a regression network for the conditional mean plus a diffusion network for the fine-scale residual.

## Architecture and inference

- **ONNX** - Open Neural Network Exchange; the framework-agnostic graph format Pangu ships as.
- **ORT** - ONNX Runtime; the engine that executes that graph (`onnxruntime-gpu` here).
- **BFC** - Best-Fit-with-Coalescing; ORT's GPU memory arena, which sits outside PyTorch's allocator (so `torch.cuda.empty_cache()` will not free it).
- **DiT** - Diffusion Transformer; CorrDiff's backbone.
- **RoPE** - Rotary Position Embedding; the position encoding that lets the DiT run on an arbitrary crop without retraining, which is what `set_domain` exploits.
- **EDM** - Elucidated Diffusion Model (Karras et al., 2022); the sampler family CorrDiff's diffusion mode uses, ~18 steps per sample.
- **Heun** - Heun's method; the second-order ODE step inside that sampler.
- **NATTEN** - Neighborhood Attention Extension; CUDA kernels for local-window attention that keep CorrDiff affordable at 2 km.
- **FNA** / **FMHA** - Fused Neighborhood Attention / Fused Multi-Head Attention; NATTEN's CUTLASS-based kernel backends.
- **CUTLASS** - CUDA Templates for Linear Algebra Subroutines; NVIDIA's kernel-building library.
- **ABI** - Application Binary Interface; the compiled-code contract that must match across torch, CUDA, Python and NATTEN's prebuilt wheels.

## Verification metrics

- **RMSE** - Root Mean Square Error; here latitude-weighted, in each variable's own units.
- **ACC** - Anomaly Correlation Coefficient; correlation of forecast and observed departures from climatology. ACC = 0.6 is the rough "useful synoptic forecast" threshold.
- **CRPS** - Continuous Ranked Probability Score; a proper score for probabilistic or ensemble forecasts.
- **FSS** - Fractions Skill Score; a neighbourhood score for precipitation that does not punish small position errors the way point-wise RMSE does.
- **Persistence** - the trivial baseline "nothing changes from the initial condition".
- **Climatology** - the long-term (1990-2019) average for the date and hour; the score every forecast converges to at long lead times.
- **Spread-skill ratio** - ensemble spread divided by ensemble-mean error; near 1 means the ensemble is neither over- nor under-confident.

## Meteorological variable shorthands (as written in the code)

- **msl** - mean sea level pressure [Pa].
- **t2m** - 2 m air temperature [K].
- **u10m**, **v10m** - 10 m eastward / northward wind components [m/s].
- **z500** - geopotential at 500 hPa [m^2/s^2]; divide by g = 9.81 for 500 hPa geopotential height in metres.
- **q700** - specific humidity at 700 hPa [kg/kg].
- **tcwv** - total column water vapour [kg/m^2]; the moisture plume in section 1.
- **tp** / **TOT_PRECIP** - total precipitation; CorrDiff returns it in metres, converted to mm in the notebook.
- **sp** - surface pressure; a CorrDiff input that Pangu does not produce (exercise 4).
