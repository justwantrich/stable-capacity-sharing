"""Stability evaluation of a park game: core feasibility and allocation diagnostics (Sections 3.3 and 4.1.3).

One numerical tolerance is used throughout: tol = 1e-6 x the sum of the enterprises' stand-alone bills (EUR).
"""
import numpy as np
import pandas as pd

from .cooperative import coalition_excess, exact_shapley, least_core, proportional
from .dual import owen_allocation


def analyse(g, vtab, phi):
    """Core feasibility and allocation diagnostics for the game ``g`` with savings table ``vtab`` and Shapley ``phi``.

    Allocations compared: Shapley ("shap"), least-core allocation closest to Shapley ("lc", Eq. (11)), allocation
    proportional to stand-alone bills ("prop_standalone"), allocation proportional to leave-one-out marginal
    contributions ("loo") and, for the composition-neutral designs, the LP-dual allocation ("owen").
    For each allocation: L1 distance to Shapley relative to grand-coalition savings ("_dist"), maximum positive
    coalition excess in EUR ("_max_excess_eur"), number of blocking coalitions ("_n_blocking"), individual-rationality
    violations in EUR ("_ir_viol_eur") and membership of the core ("_in_core": maximum excess and budget imbalance
    both within the tolerance). Returns (metrics dict, allocations dict).
    """
    p, n = g.n_players, g.n
    vN = vtab[-1]
    sc = max(abs(vN), 1e-9)
    tol = 1e-6 * g.standalone[:n].sum()
    out = {"tol_eur": tol}
    eps, lc = least_core(vtab, p, anchor=phi)
    out["eps_lc_eur"] = eps
    out["eps_lc_rel"] = eps / sc
    out["core_empty"] = eps > tol
    allocs = {"shap": phi, "lc": lc,
              "prop_standalone": proportional(np.r_[g.standalone[:n], np.zeros(p - n)], vN)}
    M = np.ones((p, p), bool)
    np.fill_diagonal(M, False)
    allocs["loo"] = proportional(vN - g.savings_masks(M), vN)
    if g.tariff.lp_representable():
        o = owen_allocation(g)
        allocs["owen"] = g.standalone - o["alloc"]
        out["owen_eff_gap_eur"] = float(allocs["owen"].sum() - vN)     # efficiency vs the enumerated game
    for k, x in allocs.items():
        exc = coalition_excess(x, vtab, p)
        out[f"{k}_dist"] = float(np.abs(x - phi).sum() / sc)
        out[f"{k}_max_excess_eur"] = float(max(exc.max(), 0.0))
        out[f"{k}_n_blocking"] = int((exc > tol).sum())
        out[f"{k}_ir_viol_eur"] = float(np.maximum(0, -x).sum())
        out[f"{k}_in_core"] = bool(exc.max() <= tol and abs(x.sum() - vN) <= tol)
    return out, allocs


def evaluate(g, full=True):
    """Evaluate one park game by exact enumeration of all 2^p coalitions.

    Returns a flat record: grand-coalition savings ``vN``, summed stand-alone bills ``scale``, relative savings
    ``save_rel`` and operator revenue ``revenue`` = summed stand-alone bills - vN (Eq. (19)). With ``full=False`` only
    the least-core value and the core classification are added; with ``full=True`` the diagnostics of ``analyse`` and
    the per-enterprise Shapley, least-core and stand-alone vectors are added.
    """
    vtab = g.savings_all()
    n = g.n
    r = dict(vN=vtab[-1], scale=g.standalone[:n].sum(), save_rel=vtab[-1] / g.standalone[:n].sum(),
             revenue=g.standalone.sum() - vtab[-1])
    if not full:
        eps, _ = least_core(vtab, g.n_players)
        r["eps_lc_eur"] = eps
        r["core_empty"] = eps > 1e-6 * r["scale"]
        return r
    phi = exact_shapley(vtab, g.n_players)
    metrics, allocs = analyse(g, vtab, phi)
    r.update(metrics)
    r.update(phi=phi[:n].tolist(), lc=allocs["lc"][:n].tolist(), standalone=g.standalone[:n].tolist())
    return r


def clean_flags(df):
    """Convert boolean flags to float (NaN where undefined, e.g. ``owen_*`` for designs without a dual allocation)."""
    for c in df.columns:
        if c.endswith("_in_core") or c == "core_empty":
            df[c] = pd.to_numeric(df[c].map(lambda v: np.nan if v is None else float(v)), errors="coerce")
    return df
