"""Capacity release and electrification hosting under fixed entitlements on the 500 matched parks (Section 3.6, 4.2.5).

Entitlement margin: 10% above each enterprise's baseline peak (``--margins`` accepts several values).

    python -m scripts.run_hosting --out outputs/hosting.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nzip.data import load_processed
from nzip.hosting import assess_park
from nzip.parks import study_parks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/hosting.parquet")
    ap.add_argument("--n_parks", type=int, default=500)
    ap.add_argument("--margins", type=float, nargs="+", default=[0.10])
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    L, m = load_processed()
    rows = []
    for k, pk in enumerate(study_parks(m)[:a.n_parks]):
        X = np.asarray(L[pk["members"]], dtype=np.float64)
        coincidence = X.sum(0).max() / X.max(1).sum()
        for r in assess_park(X, tuple(a.margins)):
            rows.append(dict(r, park=k, n=len(X), composition=pk["composition"], coincidence=coincidence))
        if k % 25 == 0:
            print(k, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    pd.set_option("display.width", 200)
    g = df.groupby(["margin", "profile"])
    print(g[["release_rel", "release_bat_rel", "alpha_ind_uniform", "alpha_gta", "alpha_gta_bat"]].median().round(3))
    print((g.host_gta_mwh.sum() / g.host_ind_mwh.sum()).round(2).rename("pooled / individual hosted energy"))
    print((g.host_gta_bat_mwh.sum() / g.host_ind_mwh.sum()).round(2).rename("pooled + storage / individual hosted energy"))


if __name__ == "__main__":
    main()
