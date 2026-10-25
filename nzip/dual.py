"""Reference linear programmes (HiGHS) for the coalition bills and the LP-dual (Owen) allocation, Section 3.4.2.

Dutch coalition LP (rates a, b, e of the coalition's category; storage with its own connection, monthly-cyclic):
  min  a C + b sum_m P_m + e * DT * sum_t ch_t                      [+ e * member energy, a constant]
  s.t. ch_t - dis_t - P_m(t)          <= -L(t)        (load rows; right-hand side additive in the enterprises)
       P_m - C                        <= 0
       -soc_end(m)                    <= -E           (storage endowment)
       soc_t <= E, ch_t <= P, dis_t <= P              (storage endowment)
       soc_t - soc_{t-1} - eta DT ch_t + DT/eta dis_t = 0,  soc at each month start = E
The same LP, with a = capacity price, b = 0 and charging priced at e (1 - eta^2), reproduces the German neutral design
behind one meter (over each cyclic month the grid-side storage losses equal (1 - eta^2) DT sum(charge)).

Owen allocation: for the composition-neutral designs the cost c(S) has the form of Eq. (16); the dual solution of the
grand-coalition LP, y_i = D_i + e E_i + F_i + lambda*' r_i, lies in the cost core (strong and weak duality), Eq. (17).
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .kernels import DT, MONTH_START
from .tariffs import NREF


def _month_index(Tn):
    m = np.zeros(Tn, int)
    for k in range(12):
        m[MONTH_START[k]:MONTH_START[k + 1]] = k
    return m


def nl_lp(L, P, E, eta, a, b, e):
    """Solve the coalition LP for the aggregated load L. Returns (optimal value excluding the member-energy term,
    scipy result, layout)."""
    Tn = len(L)
    mon = _month_index(Tn)
    bat = P > 0
    iC, iP = 0, 1
    nv = 13 + (3 * Tn if bat else 0)
    ich, idis, isoc = 13, 13 + Tn, 13 + 2 * Tn
    c = np.zeros(nv)
    c[iC] = a
    c[iP:iP + 12] = b
    if bat:
        c[ich:ich + Tn] = e * DT
    t = np.arange(Tn)
    rows, cols, vals, rhs = [], [], [], []
    rows += list(t); cols += list(iP + mon); vals += [-1.0] * Tn
    if bat:
        rows += list(t); cols += list(ich + t); vals += [1.0] * Tn
        rows += list(t); cols += list(idis + t); vals += [-1.0] * Tn
    rhs += list(-L)
    r = Tn
    for k in range(12):
        rows += [r, r]; cols += [iP + k, iC]; vals += [1.0, -1.0]; rhs.append(0.0); r += 1
    if bat:
        for k in range(12):
            rows.append(r); cols.append(isoc + MONTH_START[k + 1] - 1); vals.append(-1.0); rhs.append(-E); r += 1
        rows += list(r + t); cols += list(isoc + t); vals += [1.0] * Tn; rhs += [E] * Tn; r += Tn
        rows += list(r + t); cols += list(ich + t); vals += [1.0] * Tn; rhs += [P] * Tn; r += Tn
        rows += list(r + t); cols += list(idis + t); vals += [1.0] * Tn; rhs += [P] * Tn; r += Tn
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    A_eq = b_eq = None
    start = np.isin(t, MONTH_START[:12])
    if bat:
        prev = t[~start]
        er = list(t) + list(prev) + list(t) + list(t)
        ec = list(isoc + t) + list(isoc + prev - 1) + list(ich + t) + list(idis + t)
        ev = [1.0] * Tn + [-1.0] * len(prev) + [-eta * DT] * Tn + [DT / eta] * Tn
        A_eq = sp.csr_matrix((ev, (er, ec)), shape=(Tn, nv))
        b_eq = np.where(start, E, 0.0)
    res = linprog(c, A_ub=A_ub, b_ub=np.array(rhs), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * nv, method="highs")
    assert res.status == 0, res.message
    return res.fun, res, dict(Tn=Tn, bat=bat, start=start)


def owen_allocation(game):
    """LP-dual cost allocation of a composition-neutral game (Eq. (17)).

    Returns dict(alloc, cost_lp, dual_gap): ``alloc`` are the cost shares of all players, ``cost_lp`` the LP optimum of
    the grand coalition including constant terms and ``dual_gap`` = sum(alloc) - cost_lp (budget balance check).
    NLNEUTRAL: reference rates (a, b, e); members' own energy and storage charging energy billed at e; own fixed charges.
    DENEUTRAL: one line (s * P_over, e_over) behind one meter; no fixed charges.
    """
    assert game.tariff.lp_representable(), "dual allocation is defined for the composition-neutral designs only"
    tf = game.tariff
    if tf.kind == "NLNEUTRAL":
        a, b, e_ch, e_mem = game.A[NREF], game.B[NREF], game.E[NREF], game.E[NREF]
        fixed = game.fix_own
    else:
        a, b = tf.ref_scale * tf.P_over, 0.0
        e_mem, e_ch = tf.e_over, tf.e_over * (1 - game.eta ** 2)
        fixed = np.zeros(game.n)
    L = game.loads
    n, Tn = L.shape
    fun, res, lay = nl_lp(L.sum(0), game.P_bat, game.E_bat, game.eta, a, b, e_ch)
    mu = res.ineqlin.marginals
    y = np.zeros(game.n_players)
    y[:n] = -(L @ mu[:Tn]) + e_mem * game.ind_energy + game.adj + fixed      # + additive constants
    if lay["bat"]:
        rest = mu[Tn + 12:]
        lam = res.eqlin.marginals
        y[n] = (rest[:12] * (-game.E_bat)).sum() + rest[12:12 + Tn].sum() * game.E_bat \
            + rest[12 + Tn:].sum() * game.P_bat + (lam[lay["start"]] * game.E_bat).sum()
    total = fun + e_mem * game.ind_energy.sum() + game.adj.sum() + fixed.sum()
    return dict(alloc=y, cost_lp=total, dual_gap=float(y.sum() - total))
