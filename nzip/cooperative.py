"""Cooperative-game tools for savings games with p players (Section 3.3).

Coalitions are encoded as bitmasks (player i = bit i); ``vtab`` is the vector of savings v(S) of all 2^p coalitions,
with v(empty set) = 0 and v({i}) = 0 for every player.
"""
from math import factorial

import numpy as np
from scipy.optimize import linprog


def coalition_matrix(p):
    """Membership matrix of all proper non-empty coalitions (rows 1 ... 2^p - 2) and their bitmask indices."""
    idx = np.arange(1, 2 ** p - 1)
    return ((idx[:, None] >> np.arange(p)) & 1).astype(bool), idx


def exact_shapley(vtab, p):
    """Exact Shapley value, Eq. (9)."""
    idx = np.arange(2 ** p)
    size = np.array([bin(i).count("1") for i in range(2 ** p)])
    w = np.array([factorial(s) * factorial(p - s - 1) / factorial(p) for s in range(p)])
    phi = np.zeros(p)
    for i in range(p):
        without = idx[(idx >> i) & 1 == 0]
        phi[i] = np.sum(w[size[without]] * (vtab[without | (1 << i)] - vtab[without]))
    return phi


def proportional(weights, vN):
    """Allocation of the grand-coalition savings proportional to non-negative weights."""
    w = np.maximum(np.asarray(weights, float), 0)
    return vN * w / w.sum() if w.sum() > 0 else np.full(len(w), vN / len(w))


def coalition_excess(x, vtab, p):
    """Excess e(S) = v(S) - x(S) of all proper non-empty coalitions; e(S) > 0 means S gains by leaving, Eq. (12)."""
    masks, idx = coalition_matrix(p)
    return vtab[idx] - masks @ x


def least_core(vtab, p, anchor=None):
    """Least-core value and, optionally, the least-core allocation closest to ``anchor`` (Eqs. (10)-(11)).

    Stage 1 solves Eq. (10): min eps s.t. x(S) >= v(S) - eps for all proper non-empty S, sum x = v(N); eps is free.
    Returns (eps, x) when ``anchor`` is None, where x is the stage-1 solution returned by the solver.
    Stage 2 solves Eq. (11) for a given anchor (the Shapley value): the allocation minimising the L1 distance to the
    anchor subject to budget balance and x(S) >= v(S) - eps_bar, with
    eps_bar = max(eps, 0) + 1e-9 |v(N)|. All 2^p - 2 constraints are used (exact enumeration). When the L1 optimum is
    not unique, the solution returned by the LP solver (HiGHS) is used. Returns (eps, x).
    """
    masks, idx = coalition_matrix(p)
    masks = masks.astype(float)
    vN = vtab[-1]
    c = np.zeros(p + 1); c[-1] = 1
    A = np.hstack([-masks, -np.ones((len(idx), 1))])
    r = linprog(c, A_ub=A, b_ub=-vtab[idx], A_eq=np.r_[np.ones(p), 0][None], b_eq=[vN],
                bounds=[(None, None)] * (p + 1), method="highs")
    eps = r.x[-1]
    if anchor is None:
        return eps, r.x[:p]
    epsb = max(eps, 0) + 1e-9 * abs(vN)
    nv = 2 * p  # variables: x and d, with |x - anchor| <= d
    c = np.r_[np.zeros(p), np.ones(p)]
    A1 = np.hstack([-masks, np.zeros((len(idx), p))])
    b1 = -(vtab[idx] - epsb)
    A2 = np.vstack([np.hstack([np.eye(p), -np.eye(p)]), np.hstack([-np.eye(p), -np.eye(p)])])
    b2 = np.r_[anchor, -anchor]
    r2 = linprog(c, A_ub=np.vstack([A1, A2]), b_ub=np.r_[b1, b2], A_eq=np.r_[np.ones(p), np.zeros(p)][None],
                 b_eq=[vN], bounds=[(None, None)] * nv, method="highs")
    return eps, r2.x[:p]
