"""Economic accounting of the revenue-matched designs from the baseline table (Section 3.5, 4.2.3).

No game is re-solved: the script post-processes the table written by ``scripts.run_baseline_designs`` (all 500 parks).

    python -m scripts.run_economic_accounting --baseline outputs/baseline_designs.parquet --out outputs/accounting
"""
import argparse
from pathlib import Path

import pandas as pd

from nzip.accounting import economic_accounting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="outputs/baseline_designs.parquet")
    ap.add_argument("--out", default="outputs/accounting")
    ap.add_argument("--n_parks", type=int, default=500, help="number of parks present in the baseline table")
    ap.add_argument("--n_split", type=int, default=2000, help="random split-half repetitions")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = economic_accounting(pd.read_parquet(a.baseline), a.n_parks, a.n_split, a.seed)
    for name, frame in res.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(res["summary"].T)
    print(res["supplement"].T)


if __name__ == "__main__":
    main()
