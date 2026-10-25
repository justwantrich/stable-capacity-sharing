"""Does added demand, with or without a connection-category update, affect core feasibility? (Sections 4.1.3, 4.2.4)

Design (500 matched parks, no storage). Added demand is a winter-weighted component alpha x mean load x
(1 + 0.5 cos(2 pi t / T)) with alpha in {0, 0.1, 0.3, 0.5} (Eq. (30)).
  connections "frozen"  : capacity categories and physical levels stay at their pre-electrification values
  connections "upgrade" : the category and the >2 MW level proxy follow the electrified peak
Tariff parameters are the baseline ones in every cell; revenue-matched scales are NOT re-matched after electrification.
  Netherlands: original = NLGTA,    neutral = NLNEUTRAL_rev (revenue-matched scale)
  Germany:     original = DE_legal, neutral = DENEUTRAL     (reference line, scale s = 1)

    python -m scripts.run_electrification_sensitivity --out outputs/electrification.parquet [--workers 4]

With --workers > 1 parks are distributed over worker processes (requires the POSIX "fork" start method).
"""
import argparse
import time
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from nzip.data import load_processed
from nzip.designs import calibrate_scales, make_game
from nzip.evaluation import clean_flags, evaluate
from nzip.kernels import DT
from nzip.parks import electrify, park_inputs, study_parks
from nzip.tariffs import capacity_category, physical_level

ALPHAS = [0.0, 0.1, 0.3, 0.5]
CONNECTIONS = ["frozen", "upgrade"]
DESIGNS = {("NL", "Original"): "NLGTA", ("NL", "Neutral"): "NLNEUTRAL_rev",
           ("DE", "Original"): "DE_legal", ("DE", "Neutral"): "DENEUTRAL"}
_STATE = {}


def _init(scales):
    L, m = load_processed()
    _STATE.update(L=L, m=m, scales=scales, parks=study_parks(m))


def run_park(k):
    L, m, scales = _STATE["L"], _STATE["m"], _STATE["scales"]
    pk = _STATE["parks"][k]
    X0, cats0, levels0 = park_inputs(L, m, pk["members"])
    host = m.loc[pk["host"]]
    grid_levels = m.loc[pk["members"], "grid_level"].tolist()
    rows = []
    for alpha in ALPHAS:
        X = electrify(X0, alpha) if alpha > 0 else X0
        for conn in CONNECTIONS:
            if conn == "upgrade":
                cats, levels = capacity_category(X.max(1)), physical_level(grid_levels, X.max(1))
            else:
                cats, levels = cats0, levels0
            for (country, design), variant in DESIGNS.items():
                g = make_game(variant, X, host, cats, levels, 0.0, scales)
                r = dict(park=k, n=len(X0), members=pk["members"], country=country, design=design, variant=variant,
                         alpha=alpha, connections=conn, energy_base_kwh=float(X0.sum() * DT),
                         energy_kwh=float(X.sum() * DT), n_cat_changed=int((cats != cats0).sum()),
                         n_lvl_changed=int((levels != levels0).sum()))
                r.update(evaluate(g))
                r["phi_sum_gap_eur"] = float(sum(r["phi"]) - r["vN"])          # budget-balance checks
                r["lc_sum_gap_eur"] = float(sum(r["lc"]) - r["vN"])
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_parks", type=int, default=500)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    L, m = load_processed()
    scales = calibrate_scales(L, m, study_parks(m))
    print(f"revenue-matched scales: Netherlands {scales['nl']:.6f}, Germany {scales['de']:.6f}", flush=True)
    rows, t0 = [], time.time()
    if a.workers > 1:
        with get_context("fork").Pool(a.workers, initializer=_init, initargs=(scales,)) as pool:
            results = pool.imap_unordered(run_park, range(a.n_parks))
            for j, rr in enumerate(results):
                rows += rr
                if j % 25 == 0:
                    print(j, f"{time.time() - t0:.0f}s", flush=True)
                    clean_flags(pd.DataFrame(rows)).to_parquet(a.out)
    else:
        _init(scales)
        for k in range(a.n_parks):
            rows += run_park(k)
            if k % 25 == 0:
                print(k, f"{time.time() - t0:.0f}s", flush=True)
                clean_flags(pd.DataFrame(rows)).to_parquet(a.out)
    df = clean_flags(pd.DataFrame(rows)).sort_values(["country", "design", "connections", "alpha", "park"])
    df.to_parquet(a.out)
    pd.set_option("display.width", 200)
    print(df.groupby(["country", "design", "connections", "alpha"]).agg(
        rows=("vN", "size"), core_empty=("core_empty", "mean"), shap_in_core=("shap_in_core", "mean"),
        owen_in_core=("owen_in_core", "mean")).round(4))


if __name__ == "__main__":
    main()
