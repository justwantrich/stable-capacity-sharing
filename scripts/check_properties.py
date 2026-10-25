"""Property checks of the cost game (Sections 3.2 and 3.4.2).

1. Composition-neutral Dutch design: the LP-dual allocation is budget balanced and lies in the core (exact check over all
   coalitions), with and without storage, and the LP bill equals the kernel bill.
2. Legislated Dutch design: an analytical example with an empty core (two MS/LS-level enterprises with mutually
   exclusive 100 kW loads and one MS-level enterprise with a single 137 kW interval).
3. German original: a two-enterprise example with negative grand-coalition savings under the legal 2,500 h rule.
4. LS-level-only group: billed at LS rates (contracted capacity and time-of-use energy), fixed charges by level.

    python -m scripts.check_properties
"""
import numpy as np

from nzip.cooperative import least_core
from nzip.data import load_processed
from nzip.dual import owen_allocation
from nzip.game import GTAGame
from nzip.kernels import DT, T
from nzip.tariffs import CAT_A, CAT_FIX, HIGH_MASK, LS_E_HIGH, LS_E_LOW, GTATariff, physical_level


def check_neutral_dual_allocation():
    L, m = load_processed()
    big = m.index[m.peak_kw >= 50].to_numpy()
    rng = np.random.default_rng(0)
    worst = 0.0
    for k in range(6):
        n = int(rng.integers(3, 7))
        mem = rng.choice(big, n, replace=False)
        X = np.asarray(L[mem], float)
        levels = physical_level(m.loc[mem, "grid_level"].tolist(), X.max(1))
        g = GTAGame(X, GTATariff("NLNEUTRAL"), p_ratio=[0.0, 0.2][k % 2], levels=levels)
        p = g.n_players
        idx = np.arange(1, 2 ** p)
        M = ((idx[:, None] >> np.arange(p)) & 1).astype(bool)
        cS = g.cost_masks(M)
        o = owen_allocation(g)
        viol = (M @ o["alloc"] - cS).max() / cS[-1]                 # core violation relative to c(N)
        gap = (cS[-1] - o["cost_lp"]) / cS[-1]                      # kernel vs LP bill
        eff = abs(o["alloc"].sum() - cS[-1]) / cS[-1]               # efficiency against the enumerated game
        worst = max(worst, abs(gap), viol, eff, abs(o["dual_gap"]) / cS[-1])
        print(f"NLNEUTRAL n={n} storage={g.has_bat}: kernel-LP relative gap={gap:.1e}, budget gap={o['dual_gap']:.1e}, "
              f"maximum core violation={viol:.1e}")
    assert worst < 1e-5, worst


def check_nlgta_empty_core_example():
    t = np.arange(T)
    f1 = np.where((t // 4) % 2 == 0, 100.0, 0.0)
    f2 = 100.0 - f1
    f3 = np.ones(T)
    f3[1000] = 137.0
    g = GTAGame(np.vstack([f1, f2, f3]), GTATariff("NLGTA"), cats=np.array([1, 1, 2]), levels=np.array([1, 1, 2]))
    vt = g.savings_all()
    eps, _ = least_core(vt, 3)
    print(f"NLGTA example: v({{1,2}})={vt[3]:.1f}  v(N)={vt[7]:.1f}  least-core eps={eps:.1f}")
    assert vt[3] > vt[7] > 0 and eps > 0


def check_german_rule_example():
    """Flat enterprise (8,784 h utilisation) + short-peak enterprise under a schedule whose lines cross at 2,500 h."""
    t = np.arange(T)
    flat = np.ones(T)
    peaky = np.where(t < 96, 1.0, 0.0)
    tf = GTATariff("DE", P_over=100.0, e_over=0.01, P_under=25.0, e_under=0.04)   # P_o - P_u = (e_u - e_o) x 2500
    g = GTAGame(np.vstack([flat, peaky]), tf)
    vt = g.savings_all()
    print(f"DE example: stand-alone {g.standalone.round(1)}, v(N)={vt[-1]:.1f}")
    assert vt[-1] < 0


def check_ls_group():
    rng = np.random.default_rng(3)
    X = 60 + 40 * rng.random((2, T))
    g = GTAGame(X, GTATariff("NLGTA"), levels=np.array([0, 0]))
    c = g.cost_masks(np.array([[True, True]]))[0]
    agg = X.sum(0)
    ref = CAT_A[0] * agg.max() + LS_E_HIGH * agg[HIGH_MASK].sum() * DT + LS_E_LOW * agg[~HIGH_MASK].sum() * DT \
        + 2 * CAT_FIX[0]
    print(f"LS group: kernel={c:.2f} reference={ref:.2f}")
    assert abs(c - ref) < 1e-6 * ref


if __name__ == "__main__":
    check_ls_group()
    check_neutral_dual_allocation()
    check_nlgta_empty_core_example()
    check_german_rule_example()
    print("ok")
