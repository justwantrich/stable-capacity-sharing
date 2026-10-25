"""Validate the bill kernels against independently formulated linear programmes (Section 3.6.2).

1. Dutch designs (NLGTA, NLNEUTRAL) with and without storage: kernel bill vs the HiGHS LP of the coalition's variable
   bill at the rates of its category, plus energy, uplift / adjustment and fixed-charge terms computed independently.
   German original without storage: kernel bill vs direct evaluation of the legal 2,500 h rule, Eq. (6).
2. German neutral design with storage: kernel bill vs the HiGHS LP (``dual.nl_lp`` with a = capacity price, b = 0 and
   charging priced at e (1 - eta^2), plus e x load energy).

Exit status 1 if any absolute relative gap exceeds the tolerance (1e-5). Random coalitions are drawn from the
quality-controlled sites with fixed seeds.

    python -m scripts.validate_kernels
"""
import sys

import numpy as np

from nzip.data import load_processed
from nzip.dual import nl_lp
from nzip.game import GTAGame
from nzip.kernels import DT
from nzip.parks import make_tariff, park_inputs, study_parks
from nzip.tariffs import (CAT_A, CAT_B, CAT_E, CAT_FIX, HIGH_MASK, LS_E_HIGH, LS_E_LOW, REF, UPLIFT, GTATariff,
                          physical_level)

TOL = 1e-5


def reference_bill(g, S, kind, tf, X, cat):
    """Independent evaluation of the bill of coalition S (boolean player vector) under design ``kind``.

    ``cat`` is the tariff category of the coalition under NLGTA; the neutral design always uses the reference (MS)
    schedule at scale 1.
    """
    firm = S[:g.n]
    agg = X[firm].sum(0)
    if kind == "DE":
        C, En = agg.max(), agg.sum() * DT
        return tf.P_over * C + tf.e_over * En if En >= 2500 * C else tf.P_under * C + tf.e_under * En
    Pb = g.P_bat if S[-1] and g.has_bat else 0.0
    single = firm.sum() + int(S[-1] and g.has_bat) == 1
    rate = REF if kind == "NLNEUTRAL" else cat
    if rate == 0:         # LS group (never with storage): contracted capacity only + time-of-use energy
        e_cost = LS_E_HIGH * X[firm][:, HIGH_MASK].sum() * DT + LS_E_LOW * X[firm][:, ~HIGH_MASK].sum() * DT
        fun = CAT_A[0] * agg.max()
    else:
        fun, _, _ = nl_lp(agg, Pb, 2 * Pb, g.eta, CAT_A[rate], CAT_B[rate], CAT_E[rate])
        e_cost = CAT_E[rate] * X[firm].sum() * DT
    ref = fun + e_cost
    if kind == "NLGTA":
        if cat >= 2 and not single:
            ref += UPLIFT * g.ind_peaks[(g.levels == 1) & firm].sum()
        ref += CAT_FIX[g.cats[firm]].sum() if single else CAT_FIX[g.levels[firm]].sum()
    else:
        ref += g.adj[firm].sum() + CAT_FIX[g.cats[firm]].sum()
    return ref


def validate_original_and_neutral(L, m, n_sets=30):
    big = m.index[m.peak_kw >= 50].to_numpy()
    rng = np.random.default_rng(42)
    gaps = []
    for k in range(n_sets):
        n = int(rng.integers(2, 6))
        mem = rng.choice(big, n, replace=False)
        X = np.asarray(L[mem], float)
        levels = physical_level(m.loc[mem, "grid_level"].tolist(), X.max(1))
        if k % 5 == 4:
            levels = np.zeros(n, int)                   # force LS-level groups to exercise the LS branch
        host = m.loc[int(mem[0])]
        kind = ["NLGTA", "NLNEUTRAL", "DE"][k % 3]
        if kind == "DE":
            tf = make_tariff("DE", host)
            g = GTAGame(X, tf, levels=levels)
        else:
            tf = GTATariff(kind=kind)
            g = GTAGame(X, tf, p_ratio=0.2, levels=levels)
        for _ in range(2):
            S = np.zeros(g.n_players, bool)
            S[rng.choice(g.n, int(rng.integers(1, g.n + 1)), replace=False)] = True
            if g.has_bat:
                S[-1] = rng.random() < 0.7
            cost = g.cost_masks(S[None])[0]
            cat = -1
            if kind == "NLGTA":
                bat = np.array([bool(g.has_bat and S[-1])])
                cat = int(g.coalition_category(S[None, :g.n], bat)[0][0])
            ref = reference_bill(g, S, kind, tf, X, cat)
            gaps.append(((cost - ref) / ref, kind))
    return gaps


def validate_german_neutral_storage(L, m):
    parks = study_parks(m)
    rng = np.random.default_rng(3)
    gaps = []
    for k in rng.choice(len(parks), 6, replace=False):
        pk = parks[int(k)]
        X, _, _ = park_inputs(L, m, pk["members"])
        host = m.loc[pk["host"]]
        for share, hours in ((0.2, 2.0), (0.1, 2.0), (0.3, 2.0), (0.2, 1.0), (0.2, 4.0)):
            tf = make_tariff("DE", host)
            tf.kind = "DENEUTRAL"
            g = GTAGame(X, tf, P_abs=share * X.sum(0).max(), duration_h=hours)
            for _ in range(2):                          # random coalitions including the storage operator
                mask = np.zeros(g.n_players, bool)
                mask[rng.choice(g.n, rng.integers(1, g.n + 1), replace=False)] = True
                mask[-1] = True
                cost = g.cost_masks(mask[None])[0] - g.adj[mask[:g.n]].sum()
                agg = X[mask[:g.n]].sum(0)
                fun, _, _ = nl_lp(agg, g.P_bat, g.E_bat, g.eta, tf.P_over, 0.0, tf.e_over * (1 - g.eta ** 2))
                lp = fun + tf.e_over * agg.sum() * DT
                gaps.append(((cost - lp) / lp, "DENEUTRAL+storage"))
    return gaps


def main():
    L, m = load_processed()
    ok = True
    for title, gaps in (("original and neutral designs", validate_original_and_neutral(L, m)),
                        ("German neutral design with storage", validate_german_neutral_storage(L, m))):
        worst = {}
        for gap, kind in gaps:
            worst[kind] = max(worst.get(kind, 0.0), abs(gap))
        print(f"{title}: {len(gaps)} coalitions; maximum absolute relative gap per design:", worst)
        ok = ok and max(worst.values()) <= TOL
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
