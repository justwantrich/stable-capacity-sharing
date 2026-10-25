"""Loading and quality control of the public VEA industrial load-profile dataset (Section 4.1.1).

Input: 5,359 German industrial and commercial grid connections, 15-minute load in kW for 2016 (35,136 steps), with
master data. Dataset: Zenodo, DOI 10.5281/zenodo.13910298, licence CC BY 4.0 (download separately from Zenodo).

Expected raw files (tab-separated, extracted from the Zenodo archive) in ``VEA_RAW_DIR``:
  master_data_tabsep.csv     one row per site; columns are lower-cased on loading. Required columns:
                             id, zip_code, sector_group_id, grid_level,
                             capacity_price_over_2500h_eur_per_kw, energy_price_over_2500h_eur_per_kwh,
                             capacity_price_under_2500h_eur_per_kw, energy_price_under_2500h_eur_per_kwh
  load_profiles_tabsep.csv   one row per site (same order as the master data): id, time0 ... time35135 (kW)

Paths are controlled by the environment variables VEA_RAW_DIR and VEA_PROCESSED_DIR (defaults: ./data/vea_raw and
./data/vea_processed relative to the working directory).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(os.environ.get("VEA_RAW_DIR", "data/vea_raw"))
PROC_DIR = Path(os.environ.get("VEA_PROCESSED_DIR", "data/vea_processed"))
T = 35136
DT_H = 0.25


def convert_raw():
    """Tab-separated CSV -> float32 .npy (memory-mappable) and a master-data parquet file."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(RAW_DIR / "master_data_tabsep.csv", sep="\t")
    master.columns = [c.lower() for c in master.columns]
    loads = np.empty((len(master), T), dtype=np.float32)
    ids = []
    for k, chunk in enumerate(pd.read_csv(RAW_DIR / "load_profiles_tabsep.csv", sep="\t",
                                          chunksize=250, dtype=np.float32)):
        ids.append(chunk["id"].to_numpy().astype(int))
        loads[k * 250:k * 250 + len(chunk)] = chunk[[f"time{j}" for j in range(T)]].to_numpy(np.float32)
    ids = np.concatenate(ids)
    assert (ids == master["id"].to_numpy()).all()
    np.save(PROC_DIR / "loads_raw.npy", loads)
    master.to_parquet(PROC_DIR / "master_raw.parquet")


def quality_filter(loads, master):
    """Site-level quality control. Returns the boolean keep-mask and a per-site QC table.

    A site is retained if its profile has no missing or negative values, fewer than 50% non-positive values, an annual
    peak of at least 20 kW and a ratio of peak to the 99.9th percentile below 3, and if the capacity price for annual
    utilisation above 2,500 h is available.
    """
    nan_frac = np.isnan(loads).mean(1)
    l = np.nan_to_num(loads)
    peak = l.max(1)
    energy = l.sum(1) * DT_H
    neg_frac = (l < 0).mean(1)
    zero_frac = (l <= 0).mean(1)
    p999 = np.quantile(l, 0.999, axis=1)
    util_h = np.where(peak > 0, energy / np.maximum(peak, 1e-9), 0)
    qc = pd.DataFrame(dict(id=master["id"], nan_frac=nan_frac, neg_frac=neg_frac, zero_frac=zero_frac,
                           peak_kw=peak, energy_mwh=energy / 1e3, util_h=util_h,
                           spike_ratio=peak / np.maximum(p999, 1e-9)))
    keep = (
        (nan_frac == 0) & (neg_frac == 0)
        & (zero_frac < 0.5)
        & (peak >= 20)
        & (qc.spike_ratio < 3.0)
        & master["capacity_price_over_2500h_eur_per_kw"].notna().to_numpy()
    )
    qc["keep"] = keep
    return keep, qc


def build_processed():
    """Apply the quality filter and write the retained loads and master data (unknown labels become explicit)."""
    loads = np.load(PROC_DIR / "loads_raw.npy", mmap_mode="r")
    master = pd.read_parquet(PROC_DIR / "master_raw.parquet")
    keep, qc = quality_filter(np.asarray(loads), master)
    qc.to_csv(PROC_DIR / "qc_table.csv", index=False)
    m = master[keep].reset_index(drop=True)
    m["sector_group_id"] = m["sector_group_id"].fillna("U")  # unknown sector
    m["grid_level"] = m["grid_level"].fillna("unknown")
    L = np.asarray(loads)[keep].astype(np.float32)
    m["peak_kw"] = L.max(1)
    m["energy_mwh"] = L.sum(1) * DT_H / 1e3
    m["util_h"] = m.energy_mwh * 1e3 / m.peak_kw
    m["row"] = np.arange(len(m))
    np.save(PROC_DIR / "loads.npy", L)
    m.to_parquet(PROC_DIR / "master.parquet")
    return m, qc


def load_processed(mmap=True):
    """Return (loads (n_sites, 35136) in kW, master data frame) of the retained sites."""
    L = np.load(PROC_DIR / "loads.npy", mmap_mode="r" if mmap else None)
    m = pd.read_parquet(PROC_DIR / "master.parquet")
    return L, m


if __name__ == "__main__":
    import sys
    if "--convert" in sys.argv:
        convert_raw()
    m, qc = build_processed()
    print(json.dumps(dict(n_raw=len(qc), n_kept=int(qc.keep.sum()),
                          grid_levels=m.grid_level.value_counts().to_dict()), indent=1, default=int))
