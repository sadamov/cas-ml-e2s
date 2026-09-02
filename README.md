# CAS ML - AI Weather Forecasting and Downscaling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sadamov/cas-ml-e2s/blob/main/storm_boris_ai_forecast.ipynb)

[storm_boris_ai_forecast.ipynb](storm_boris_ai_forecast.ipynb) is a one-hour Google Colab
session built for a free-tier **T4 GPU**. It runs a **Pangu-Weather** daily forecast of Storm
Boris (September 2024) from ERA5, verifies it, and downscales ERA5 to **2.2 km, hourly** over
the Alps with CorrDiff -- including a small diffusion-sampled ensemble.

The notebook at the repo root is the **worksheet**: six code cells are blanked to a `# TODO`
plus a hint for participants to complete (the rollout call, latitude-weighted RMSE, the
persistence baseline, `set_domain`, `map_coords`, the ensemble size, the spread map). The
fully worked, pre-rendered version is
[`.instructor/storm_boris_ai_forecast_solutions.ipynb`](.instructor/storm_boris_ai_forecast_solutions.ipynb).

## Session structure

| Section | Content                                                        | Budget |
| ------- | -------------------------------------------------------------- | ------ |
| 0       | Runtime check,`earth2studio` install                         | 10 min |
| 1       | ERA5 from ARCO, synoptic overview of the storm                 | 5 min  |
| 2       | Pangu-Weather (`Pangu24`), 6-step daily rollout              | 15 min |
| 3       | RMSE / ACC against persistence and climatology                 | 10 min |
| 4       | CorrDiff ERA5 -> COSMO-REA2, hourly animation, mean + diffusion ensemble | 20 min |
| 5       | Exercises (take-home)                                          | 30 min |
