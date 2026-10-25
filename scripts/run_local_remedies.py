"""Do local tariff remedies keep the core non-empty under changing conditions? (Section 4.2.2)

Compares the original Dutch design, the two local remedies and the composition-neutral design (reference rates) on
matched scenario inputs, recording core feasibility only (no storage, no added load):

  enriched  composition-enriched parks: at least one TS/MS-level site, at least two low-utilisation MS-category sites
            and random fill (sites with annual peak >= 50 kW)
  insert    single-member entry: each of the 500 matched parks plus one extra TS/MS site (10 draws) or one random
            control site (5 draws); each trial restarts from the baseline membership
  rates     tariff-rate perturbation: multipliers on the (A, B, E) coefficients of every category drawn from
            U[0.7, 1.3]; membership and demand are retained, stand-alone benchmarks and member adjustments are
            recalculated, fixed charges are not perturbed and the uplift follows the perturbed capacity coefficients

    python -m scripts.run_local_remedies --exp enriched --out outputs/remedies_enriched.parquet
    python -m scripts.run_local_remedies --exp insert   --out outputs/remedies_insert.parquet
    python -m scripts.run_local_remedies --exp rates    --out outputs/remedies_rates.parquet
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nzip.data import load_processed
from nzip.designs import make_game
from nzip.evaluation import evaluate
from nzip.kernels import DT
from nzip.parks import park_inputs, study_parks

VARIANTS = ["NLGTA", "NL_no_uplift", "NL_only_catrule", "NLNEUTRAL"]
LOW_UTIL_H = 2300.0     # utilisation (h) below which an MS-category site is classed as low-utilisation
DEFAULT_PARKS = {"enriched": 300, "insert": 500, "rates": 200}


def _checkpoint(rows, out):
    pd.DataFrame(rows).to_parquet(out)


def run_enriched(L, m, n_parks, seed, out):
    rng = np.random.default_rng(seed)
    pk_kw = m.peak_kw
    ok = m.index[pk_kw >= 50]
    util = {i: float(np.asarray(L[[i]]).sum() * DT / pk_kw[i]) for i in ok}
    tsms = [i for i in ok if pk_kw[i] > 2000]
    low_ms = [i for i in ok if 136 < pk_kw[i] <= 2000 and util[i] < LOW_UTIL_H]
    print(f"pools: TS/MS {len(tsms)}, low-utilisation MS {len(low_ms)}", flush=True)
    tsms_order = rng.permutation(tsms)
    rows, t0 = [], time.time()
    for k in range(n_parks):
        n = int(rng.integers(4, 11))
        mem = [int(tsms_order[k % len(tsms_order)])]
        mem += [int(x) for x in rng.choice([i for i in low_ms if i not in mem], 2, replace=False)]
        rest = rng.choice([i for i in ok if i not in mem], n - 3, replace=False)
        mem += [int(x) for x in rest]
        X, cats, levels = park_inputs(L, m, mem)
        for name in VARIANTS:
            g = make_game(name, X, None, cats, levels)
            r = dict(park=k, n=n, variant=name, members=mem, n_lvl_tsms=int((levels == 3).sum()),
                     n_lvl_msls=int((levels == 1).sum()))
            r.update(evaluate(g, full=False))
            rows.append(r)
        if k % 25 == 0:
            print(k, f"{time.time() - t0:.0f}s", flush=True)
            _checkpoint(rows, out)
    return rows


def run_insert(L, m, parks, seed, k_tsms, k_rand, out):
    rng = np.random.default_rng(seed)
    pk_kw = m.peak_kw
    ok = m.index[pk_kw >= 50].to_numpy()
    tsms = m.index[pk_kw > 2000].to_numpy()
    rows, t0 = [], time.time()
    for k, pk in enumerate(parks):
        cand_t = rng.choice(np.setdiff1d(tsms, pk["members"]), k_tsms, replace=False)
        cand_r = rng.choice(np.setdiff1d(ok, pk["members"]), k_rand, replace=False)
        for kind, cands in (("tsms", cand_t), ("random", cand_r)):
            for j in cands:
                X, cats, levels = park_inputs(L, m, pk["members"] + [int(j)])
                for name in VARIANTS:
                    g = make_game(name, X, None, cats, levels)
                    r = dict(park=k, insert=kind, site=int(j), variant=name, n=len(X))
                    r.update(evaluate(g, full=False))
                    rows.append(r)
        if k % 10 == 0:
            print(k, f"{time.time() - t0:.0f}s", flush=True)
            _checkpoint(rows, out)
    return rows


def run_rates(L, m, parks, seed, n_draws, out):
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    for k, pk in enumerate(parks):
        X, cats, levels = park_inputs(L, m, pk["members"])
        for d in range(n_draws):
            mult = rng.uniform(0.7, 1.3, (4, 3))
            for name in VARIANTS:
                g = make_game(name, X, None, cats, levels, 0.0, rate_mult=mult)
                r = dict(park=k, draw=d, variant=name, n=len(X), uplift_eur=g.uplift)
                r.update(evaluate(g, full=False))
                rows.append(r)
        if k % 10 == 0:
            print(k, f"{time.time() - t0:.0f}s", flush=True)
            _checkpoint(rows, out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["enriched", "insert", "rates"])
    ap.add_argument("--n_parks", type=int, default=None, help="default: 300 (enriched), 500 (insert), 200 (rates)")
    ap.add_argument("--n_draws", type=int, default=20, help="rates: perturbation draws per park")
    ap.add_argument("--k_tsms", type=int, default=10, help="insert: extra TS/MS sites per park")
    ap.add_argument("--k_rand", type=int, default=5, help="insert: random control sites per park")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n_parks = a.n_parks or DEFAULT_PARKS[a.exp]
    L, m = load_processed()
    if a.exp == "enriched":
        rows = run_enriched(L, m, n_parks, a.seed, a.out)
    elif a.exp == "insert":
        rows = run_insert(L, m, study_parks(m)[:n_parks], a.seed, a.k_tsms, a.k_rand, a.out)
    else:
        rows = run_rates(L, m, study_parks(m)[:n_parks], a.seed, a.n_draws, a.out)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    keys = ["insert", "variant"] if a.exp == "insert" else ["variant"]
    print(df.groupby(keys).agg(rows=("vN", "size"), core_empty=("core_empty", "mean")).round(4))
    if a.exp == "rates":     # share of parks with an empty core in at least one of their draws
        print(df.groupby(["variant", "park"]).core_empty.max().groupby("variant").mean().round(4))


if __name__ == "__main__":
    main()
