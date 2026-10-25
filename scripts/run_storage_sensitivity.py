"""Sensitivity of coalition stability to shared-storage size and duration (Sections 3.1, 4.1.3, 4.2.4).

The storage operator is an additional player with a fixed size in every coalition. Power is a share of the park's raw
aggregate peak; round-trip efficiency is 90% and identical storage is used across designs. Revenue-matched scales are
those calibrated in the baseline (not re-matched). The German original is evaluated with the legal 2,500 h rule and
peak-shaving dispatch only (see ``kernels.de_dispatch``).

    python -m scripts.run_storage_sensitivity --country nl --out outputs/storage_nl.parquet
    python -m scripts.run_storage_sensitivity --country de --out outputs/storage_de.parquet
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from nzip.data import load_processed
from nzip.designs import calibrate_scales, make_game
from nzip.evaluation import clean_flags, evaluate
from nzip.parks import park_inputs, study_parks

CONFIGS = {"bat_20_2h": (0.2, 2.0), "bat_10_2h": (0.1, 2.0), "bat_30_2h": (0.3, 2.0),
           "bat_20_1h": (0.2, 1.0), "bat_20_4h": (0.2, 4.0)}      # (power share of raw park peak, duration in hours)
DESIGNS = {"nl": ["NLGTA", "NLNEUTRAL", "NLNEUTRAL_rev"], "de": ["DE_legal", "DENEUTRAL", "DENEUTRAL_rev"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True, choices=["nl", "de"])
    ap.add_argument("--n_parks", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    L, m = load_processed()
    parks = study_parks(m)
    scales = calibrate_scales(L, m, parks)
    rows, t0 = [], time.time()
    for k, pk in enumerate(parks[:a.n_parks]):
        X, cats, levels = park_inputs(L, m, pk["members"])
        host = m.loc[pk["host"]]
        for cfg, (share, hours) in CONFIGS.items():
            P_abs = share * X.sum(0).max()
            for name in DESIGNS[a.country]:
                g = make_game(name, X, host, cats, levels, P_abs, scales, duration_h=hours)
                r = dict(park=k, n=len(X), variant=name, config=cfg, p_ratio=share, duration_h=hours, battery=True,
                         s_nl=scales["nl"], s_de=scales["de"])
                r.update(evaluate(g))
                rows.append(r)
        if k % 10 == 0:
            print(k, f"{time.time() - t0:.0f}s", flush=True)
            clean_flags(pd.DataFrame(rows)).to_parquet(a.out)
    df = clean_flags(pd.DataFrame(rows))
    df.to_parquet(a.out)
    pd.set_option("display.width", 200)
    print(df.groupby(["config", "variant"]).agg(rows=("vN", "size"), core_empty=("core_empty", "mean"),
                                                shap_in_core=("shap_in_core", "mean")).round(4))


if __name__ == "__main__":
    main()
