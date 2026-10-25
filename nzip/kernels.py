"""Exact battery-dispatch and bill kernels (numba), manuscript Sections 3.2.1-3.2.3.

Time grid: 15-minute intervals over a leap year (T = 35,136), months decoupled by a monthly-cyclic battery (full at
every month start and end), see Eq. (3).

Dispatch logic
--------------
* ``dispatch_flows`` is the greedy cyclic dispatch at an import cap C: it discharges only when the load exceeds the cap
  and recharges as early as the cap allows. For a given cap it yields the minimum possible charging energy.
* ``cap_is_feasible`` / ``min_feasible_cap`` give the exact feasibility oracle and the lowest feasible cap of one month.
* Dutch designs (``nl_dispatch``): the monthly cost f_m(x) = b * x + e * charging(x) is convex in the monthly cap x. The
  annual contract C couples months through C >= x_m, so the optimum is x_m = min(C, argmin f_m) with a convex outer
  problem in C. Both levels are solved by golden-section search on top of the exact feasibility oracle.
* German original tariff (``de_dispatch``): peak-shaving dispatch under the legal 2,500 h rule; the cap is searched
  within the region of each utilisation line and the cheaper evaluated bill is returned (Section 3.2.3).
The kernels are validated against independently formulated linear programmes in ``scripts/validate_kernels.py``.
"""
import numba as nb
import numpy as np

DT = 0.25                      # hours per time step
T = 35136                      # time steps per (leap) year
_MONTH_LEN = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 96
MONTH_START = np.r_[0, np.cumsum(_MONTH_LEN)]
GR = 0.6180339887498949        # golden ratio conjugate


@nb.njit(cache=True)
def cap_is_feasible(load, a, b, C, P, E, eta_c, eta_d):
    """True if the cyclic battery (power P, energy E) can keep imports <= C on steps [a, b) and end the month full."""
    soc = E
    for t in range(a, b):
        x = load[t]
        if x > C:
            need = x - C
            if need > P + 1e-9:
                return False
            e = need * DT / eta_d
            if e > soc + 1e-9:
                return False
            soc -= e
        elif soc < E:
            room = min(C - x, P)
            add = min(room * DT * eta_c, E - soc)
            soc += add
    return soc >= E - 1e-6 * max(E, 1.0)


@nb.njit(cache=True)
def dispatch_flows(load, a, b, C, P, E, eta_c, eta_d):
    """Greedy cyclic dispatch at cap C on [a, b): returns (net import kWh, grid-side charging kWh)."""
    soc = E
    net = 0.0
    chg = 0.0
    for t in range(a, b):
        x = load[t]
        if x > C:
            need = min(x - C, P)
            e = min(need * DT / eta_d, soc)
            soc -= e
            net += x * DT - e * eta_d
        else:
            room = min(C - x, P)
            add = min(room * DT * eta_c, E - soc)
            soc += add
            net += x * DT + add / eta_c
            chg += add / eta_c
    return net, chg


@nb.njit(cache=True)
def min_feasible_cap(load, a, b, P, E, eta_c, eta_d, rtol):
    """Lowest feasible import cap on [a, b) by bisection; returns (cap, raw maximum of the load)."""
    mx = 0.0
    s = 0.0
    for t in range(a, b):
        mx = max(mx, load[t])
        s += load[t]
    lo = max(mx - P, s / (b - a))          # cyclic: average import >= average load
    hi = mx
    if cap_is_feasible(load, a, b, lo, P, E, eta_c, eta_d):
        return lo, mx
    while hi - lo > rtol * mx:
        mid = 0.5 * (lo + hi)
        if cap_is_feasible(load, a, b, mid, P, E, eta_c, eta_d):
            hi = mid
        else:
            lo = mid
    return hi, mx


@nb.njit(cache=True)
def monthly_cost(load, a, b, x, P, E, eta_c, eta_d, bcoef, ecoef):
    """Monthly cost b * x + e * charging(x) at monthly cap x."""
    return bcoef * x + ecoef * dispatch_flows(load, a, b, x, P, E, eta_c, eta_d)[1]


@nb.njit(cache=True)
def optimal_monthly_cap(lo0, hi0, load, a, b, P, E, eta_c, eta_d, bcoef, ecoef, iters):
    """argmin over x in [lo0, hi0] of bcoef * x + ecoef * charging(x) for one month (convex)."""
    lo, hi = lo0, hi0
    x1 = hi - GR * (hi - lo)
    x2 = lo + GR * (hi - lo)
    f1 = monthly_cost(load, a, b, x1, P, E, eta_c, eta_d, bcoef, ecoef)
    f2 = monthly_cost(load, a, b, x2, P, E, eta_c, eta_d, bcoef, ecoef)
    for _ in range(iters):
        if f1 < f2:
            hi = x2; x2 = x1; f2 = f1
            x1 = hi - GR * (hi - lo)
            f1 = monthly_cost(load, a, b, x1, P, E, eta_c, eta_d, bcoef, ecoef)
        else:
            lo = x1; x1 = x2; f1 = f2
            x2 = lo + GR * (hi - lo)
            f2 = monthly_cost(load, a, b, x2, P, E, eta_c, eta_d, bcoef, ecoef)
    best_x = 0.5 * (x1 + x2)
    best_f = monthly_cost(load, a, b, best_x, P, E, eta_c, eta_d, bcoef, ecoef)
    for x in (lo0, hi0):
        f = monthly_cost(load, a, b, x, P, E, eta_c, eta_d, bcoef, ecoef)
        if f < best_f:
            best_x = x
            best_f = f
    return best_x


@nb.njit(cache=True)
def contract_cost(C, agg, xs, P, E, eta_c, eta_d, acoef, bcoef, ecoef, mstart):
    """Annual cost as a function of the contracted capacity C (monthly caps are min(C, xs[m]))."""
    tot = acoef * C
    for m in range(12):
        x = min(C, xs[m])
        tot += monthly_cost(agg, mstart[m], mstart[m + 1], x, P, E, eta_c, eta_d, bcoef, ecoef)
    return tot


@nb.njit(cache=True)
def nl_dispatch(agg, P, E, eta_c, eta_d, acoef, bcoef, ecoef, mstart, rtol, iters):
    """Bill-optimal Dutch dispatch for one coalition load. Returns (monthly maxima (12,), charging kWh)."""
    lo = np.zeros(12)
    hi = np.zeros(12)
    xs = np.zeros(12)
    for m in range(12):
        lo[m], hi[m] = min_feasible_cap(agg, mstart[m], mstart[m + 1], P, E, eta_c, eta_d, rtol)
        if ecoef <= 0.0:
            xs[m] = lo[m]
        else:
            xs[m] = optimal_monthly_cap(lo[m], hi[m], agg, mstart[m], mstart[m + 1], P, E, eta_c, eta_d,
                                        bcoef, ecoef, iters)
    Cl = lo.max()
    Cu = max(xs.max(), Cl)
    C = Cu
    if Cu - Cl > 1e-9 and acoef > 0:
        a_, b_ = Cl, Cu
        x1 = b_ - GR * (b_ - a_)
        x2 = a_ + GR * (b_ - a_)
        f1 = contract_cost(x1, agg, xs, P, E, eta_c, eta_d, acoef, bcoef, ecoef, mstart)
        f2 = contract_cost(x2, agg, xs, P, E, eta_c, eta_d, acoef, bcoef, ecoef, mstart)
        for _ in range(iters):
            if f1 < f2:
                b_ = x2; x2 = x1; f2 = f1; x1 = b_ - GR * (b_ - a_)
                f1 = contract_cost(x1, agg, xs, P, E, eta_c, eta_d, acoef, bcoef, ecoef, mstart)
            else:
                a_ = x1; x1 = x2; f1 = f2; x2 = a_ + GR * (b_ - a_)
                f2 = contract_cost(x2, agg, xs, P, E, eta_c, eta_d, acoef, bcoef, ecoef, mstart)
        C = 0.5 * (x1 + x2)
        fC = contract_cost(C, agg, xs, P, E, eta_c, eta_d, acoef, bcoef, ecoef, mstart)
        for cand in (Cl, Cu):
            fc = contract_cost(cand, agg, xs, P, E, eta_c, eta_d, acoef, bcoef, ecoef, mstart)
            if fc < fC:
                C = cand
                fC = fc
    peaks = np.zeros(12)
    chg = 0.0
    for m in range(12):
        x = min(C, xs[m])
        peaks[m] = x
        chg += dispatch_flows(agg, mstart[m], mstart[m + 1], x, P, E, eta_c, eta_d)[1]
    return peaks, chg


@nb.njit(cache=True)
def annual_net_energy(agg, C, P, E, eta_c, eta_d, mstart):
    """Annual net grid import (kWh) under the greedy cyclic dispatch at cap C."""
    tot = 0.0
    for m in range(12):
        tot += dispatch_flows(agg, mstart[m], mstart[m + 1], C, P, E, eta_c, eta_d)[0]
    return tot


@nb.njit(cache=True)
def de_bill(agg, C, P, E, eta_c, eta_d, Po, eo, Pu, eu, mstart):
    """German two-part bill at cap C with the legal utilisation rule: upper line iff net energy >= 2,500 h x C."""
    En = annual_net_energy(agg, C, P, E, eta_c, eta_d, mstart)
    if En >= 2500.0 * C:
        return Po * C + eo * En
    return Pu * C + eu * En


@nb.njit(cache=True)
def de_dispatch(agg, P, E, eta_c, eta_d, Po, eo, Pu, eu, mstart, rtol, iters):
    """German single-meter bill with the legal 2,500 h rule; peak-shaving cap searched within each line's region.

    Candidate caps lie between the lowest feasible cap and the raw load peak; deliberate cycling or peak raising to
    change the utilisation line is not modelled (Section 3.2.3).
    """
    Cl = 0.0
    for m in range(12):
        Cl = max(Cl, min_feasible_cap(agg, mstart[m], mstart[m + 1], P, E, eta_c, eta_d, rtol)[0])
    Craw = agg.max()
    # h(C) = En(C) / C is decreasing in C: switch point Cs with En(Cs) = 2500 Cs
    lo, hi = Cl, Craw
    if annual_net_energy(agg, lo, P, E, eta_c, eta_d, mstart) < 2500.0 * lo:
        Cs = lo
    elif annual_net_energy(agg, hi, P, E, eta_c, eta_d, mstart) >= 2500.0 * hi:
        Cs = hi
    else:
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if annual_net_energy(agg, mid, P, E, eta_c, eta_d, mstart) >= 2500.0 * mid:
                lo = mid
            else:
                hi = mid
        Cs = lo
    r0s = np.array([Cl, Cs])
    r1s = np.array([Cs, Craw])
    Pks = np.array([Po, Pu])
    eks = np.array([eo, eu])
    best = 1e300
    for j in range(2):
        r0, r1, Pk, ek = r0s[j], r1s[j], Pks[j], eks[j]
        if r1 < r0:
            continue
        a_, b_ = r0, r1
        x1 = b_ - GR * (b_ - a_)
        x2 = a_ + GR * (b_ - a_)
        f1 = Pk * x1 + ek * annual_net_energy(agg, x1, P, E, eta_c, eta_d, mstart)
        f2 = Pk * x2 + ek * annual_net_energy(agg, x2, P, E, eta_c, eta_d, mstart)
        for _ in range(iters):
            if f1 < f2:
                b_ = x2; x2 = x1; f2 = f1; x1 = b_ - GR * (b_ - a_)
                f1 = Pk * x1 + ek * annual_net_energy(agg, x1, P, E, eta_c, eta_d, mstart)
            else:
                a_ = x1; x1 = x2; f1 = f2; x2 = a_ + GR * (b_ - a_)
                f2 = Pk * x2 + ek * annual_net_energy(agg, x2, P, E, eta_c, eta_d, mstart)
        for c in (r0, r1, x1, x2):
            best = min(best, de_bill(agg, c, P, E, eta_c, eta_d, Po, eo, Pu, eu, mstart))
    return best


@nb.njit(parallel=True, cache=True)
def batch_costs(masks, use_bat, loads, P, E, eta_c, eta_d, mode, acoef, bcoef, ecoef,
                Po, eo, Pu, eu, mstart, rtol, iters):
    """Evaluate a batch of coalitions in parallel.

    mode 0 (Dutch): per-coalition rates acoef / bcoef / ecoef -> (monthly maxima (B, 12), charging kWh (B,));
    mode 1 (German): -> (monthly maxima of the raw load (B, 12), bill (B,)).
    """
    B, n = masks.shape
    Tn = loads.shape[1]
    peaks = np.zeros((B, 12))
    out = np.zeros(B)
    for bb in nb.prange(B):
        agg = np.zeros(Tn)
        anym = False
        for i in range(n):
            if masks[bb, i]:
                anym = True
                for t in range(Tn):
                    agg[t] += loads[i, t]
        if not anym:
            continue
        withbat = use_bat[bb] and P > 0
        if mode == 0:
            if withbat:
                pk, ch = nl_dispatch(agg, P, E, eta_c, eta_d, acoef[bb], bcoef[bb], ecoef[bb], mstart, rtol, iters)
                for m in range(12):
                    peaks[bb, m] = pk[m]
                out[bb] = ch
            else:
                for m in range(12):
                    mx = 0.0
                    for t in range(mstart[m], mstart[m + 1]):
                        mx = max(mx, agg[t])
                    peaks[bb, m] = mx
        else:
            if withbat:
                out[bb] = de_dispatch(agg, P, E, eta_c, eta_d, Po, eo, Pu, eu, mstart, rtol, iters)
            else:
                C = agg.max()
                En = agg.sum() * DT
                out[bb] = Po * C + eo * En if En >= 2500.0 * C else Pu * C + eu * En
            for m in range(12):
                mx = 0.0
                for t in range(mstart[m], mstart[m + 1]):
                    mx = max(mx, agg[t])
                peaks[bb, m] = mx
    return peaks, out
