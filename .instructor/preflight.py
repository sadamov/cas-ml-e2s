"""Probe 3: can Pangu24 and CorrDiff diffusion coexist on one T4?

Standalone copy of the notebook's "Instructor preflight" section, kept here so it
can be run without opening the notebook. Keep BBOX / N_SAMPLES in sync with cell 9.

Pangu24 FITS (measured: 15.30/15.6 GB peak, 8 s/step, 24 h step, 1.10 GB ONNX).
FengWu OOMed in the ORT arena; FCN3 and Atlas ruled out earlier.

Two open questions, both of which decide the session design:
  A. ONNX Runtime allocates outside torch's allocator, so torch.cuda.empty_cache()
     cannot free it. If deleting the Pangu session does not return the ~15 GB,
     CorrDiff cannot load in the same kernel and the notebook needs a restart
     between sections -- which is fine, but I have to write it that way.
  B. What diffusion mode costs: 18 EDM/Heun steps x N samples on the downscale domain.

Run after cells 0-5 in a FRESH kernel.
"""

import gc
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch

from earth2studio import run
from earth2studio.data import ARCO, fetch_data
from earth2studio.io import XarrayBackend
from earth2studio.utils.coords import map_coords

REPORT = []
BBOX = dict(lat_min=45.9, lat_max=49.1, lon_min=12.2, lon_max=19.6)  # keep in sync with cell 9
N_SAMPLES = 4


def log(k, v):
    REPORT.append(f"{k:<30} {v}")
    print(f"{k:<30} {v}")


def vram(tag):
    free, total = torch.cuda.mem_get_info()
    log(tag, f"{(total - free) / 1e9:.2f} / {total / 1e9:.1f} GB in use")
    return (total - free) / 1e9


# ---- A. does ORT hand the memory back? ---------------------------------------
vram("baseline VRAM")
try:
    from earth2studio.models.px import Pangu24

    model = Pangu24.load_model(Pangu24.load_default_package())
    lat = np.linspace(90.0, -90.0, 721)
    oc = OrderedDict({
        "variable": np.array(["msl", "t2m", "u10m", "v10m"]),
        "lat": lat[(lat >= 32.0) & (lat <= 62.0)],
    })
    t0 = time.perf_counter()
    run.deterministic([np.datetime64("2024-09-09T00:00:00")], 2, model, ARCO(),
                      XarrayBackend(), output_coords=oc, device="cuda", verbose=False)
    log("Pangu24 2 steps", f"{time.perf_counter() - t0:.0f} s")
    vram("VRAM with Pangu resident")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    after = vram("VRAM after deleting Pangu")
    log("ORT memory released", "YES" if after < 2.0 else "NO -- needs kernel restart")
except Exception as e:
    log("Pangu24 stage", f"FAILED {type(e).__name__}: {str(e)[:140]}")
    traceback.print_exc()


# ---- B. CorrDiff diffusion cost ----------------------------------------------
for mode in ["mean", "diffusion"]:
    try:
        from earth2studio.models.dx import CorrDiffCosmoEra5

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        dx = CorrDiffCosmoEra5.load_model(
            CorrDiffCosmoEra5.load_default_package(), device="cuda",
            mode=mode, resolution="rea2",
        )
        alps = dx.set_domain(**BBOX).to("cuda")
        if mode == "diffusion":
            alps.number_of_samples = N_SAMPLES
        log(f"{mode}: load", f"{time.perf_counter() - t0:.0f} s")

        ic = alps.input_coords()
        log(f"{mode}: input grid", f"{len(ic['lat'])} x {len(ic['lon'])}")
        log(f"{mode}: output grid", f"{np.asarray(alps.output_coords(ic)['lat']).shape}")

        x, coords = fetch_data(
            source=ARCO(), time=np.array(["2024-09-15T00:00:00"], dtype="datetime64[ns]"),
            variable=ic["variable"], device="cuda",
        )
        x = x[:, 0]
        coords = OrderedDict((k, v) for k, v in coords.items() if k != "lead_time")
        x, coords = map_coords(x, coords, ic)

        t0 = time.perf_counter()
        with torch.inference_mode():
            hires, hcoords = alps(x, coords)
        log(f"{mode}: forward", f"{time.perf_counter() - t0:.1f} s -> {tuple(hires.shape)}")
        log(f"{mode}: peak torch alloc", f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        names = [str(v) for v in hcoords["variable"]]
        if mode == "mean":
            log("output variables", ",".join(names))
        f = hires.float().cpu().numpy()  # (sample, time, variable, H, W); no batch axis to drop
        if "tp" in names:
            tp = f[:, 0, names.index("tp")] * 1e3
            log(f"{mode}: precip max [mm]", f"{tp.max():.2f}")
            if mode == "diffusion":
                log("spread across samples", f"{tp.max(axis=0).mean() - tp.min(axis=0).mean():.3f} mm mean range")
        del dx, alps, hires
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        log(f"{mode}: VERDICT", f"FAILED {type(e).__name__}: {str(e)[:140]}")
        traceback.print_exc()

print("\n\n" + "=" * 60)
print("\n".join(REPORT))
