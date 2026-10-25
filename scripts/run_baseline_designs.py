"""Baseline comparison of tariff designs on the 500 matched parks (Sections 4.2.1-4.2.3).

No storage and no added electrification. For every park the exact game (all 2^p coalitions) is evaluated under the
original designs, the two local remedies and the composition-neutral designs (reference-rate and revenue-matched),
and core feasibility, allocation diagnostics, savings and operator revenue are recorded.

The revenue-matched scales of Eq. (20) are calibrated on all 500 parks first (``--n_parks`` only limits the number
of parks evaluated, e.g. for a quick check).

    python -m scripts.run_baseline_designs --out outputs/baseline_designs.parquet
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from nzip.data import load_processed
from nzip.designs import calibrate_scales, make_game
from nzip.evaluation import clean_flags, evaluate
from nzip.parks import park_inputs, study_parks

VARIANTS = ["NLGTA", "NL_no_uplift", "NL_only_catrule", "NLNEUTRAL", "NLNEUTRAL_rev",
            "DE_legal", "DENEUTRAL", "DENEUTRAL_rev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_parks", type=int, default=500)
    ap.add_argument("--out", default="outputs/baseline_designs.parquet")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    L, m = load_processed()
    parks = study_parks(m)
    scales = calibrate_scales(L, m, parks)
    print(f"revenue-matched scales: Netherlands {scales['nl']:.4f}, Germany {scales['de']:.4f}", flush=True)
    rows, t0 = [], time.time()
    for k, pk in enumerate(parks[:a.n_parks]):
        X, cats, levels = park_inputs(L, m, pk["members"])
        host = m.loc[pk["host"]]
        for name in VARIANTS:
            g = make_game(name, X, host, cats, levels, 0.0, scales)
            r = dict(park=k, n=len(X), variant=name, battery=False, s_nl=scales["nl"], s_de=scales["de"],
                     n_lvl_msls=int((levels == 1).sum()), n_lvl_tsms=int((levels == 3).sum()))
            r.update(evaluate(g))
            rows.append(r)
        if k % 25 == 0:
            print(k, f"{time.time() - t0:.0f}s", flush=True)
            clean_flags(pd.DataFrame(rows)).to_parquet(a.out)
    df = clean_flags(pd.DataFrame(rows))
    df.to_parquet(a.out)
    pd.set_option("display.width", 200)
    print(df.groupby("variant").agg(rows=("vN", "size"), core_empty=("core_empty", "mean"),
                                    shap_in_core=("shap_in_core", "mean"), save=("save_rel", "median"),
                                    revenue=("revenue", "sum")).round(4))


if __name__ == "__main__":
    main()
