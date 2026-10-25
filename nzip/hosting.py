"""Electrification hosting assessment under fixed capacity entitlements (Section 3.6, Eqs. (26)-(29)).

Entitlements: C0_i = own baseline peak x (1 + margin) [assumed contractual margin].
Added load of enterprise i: alpha * mean(L_i) * h(t) with a temporal profile h of unit annual mean [assumed profiles]:
  "heat"     winter-weighted, 1 + 0.5 cos(2 pi t / T), Eq. (27)
  "flat"     constant
  "process"  proportional to the enterprise's own baseline load
Hosting is the largest proportional addition alpha whose load stays within the entitlements:
  individual operation : every enterprise alone within its own C0_i (closed form, Eq. (28) for a singleton set)
  pooled operation     : the park's joint 15-minute load within sum_i C0_i
  pooled with storage  : the same with shared storage (monthly-cyclic), found by bisection
Contractual headroom = sum_i C0_i - joint peak (with storage: lowest feasible joint peak).
"""
import numpy as np

from .kernels import DT, MONTH_START, cap_is_feasible, min_feasible_cap

ETA = np.sqrt(0.9)                                   # one-way efficiency for 90% round trip
STORAGE_POWER_SHARE, STORAGE_HOURS = 0.2, 2.0        # shared storage: 20% of the raw park peak, 2 h
PROFILES = ("heat", "flat", "process")


def profile_shape(kind, X):
    """Temporal profile h(t) of unit mean for every enterprise (rows of X)."""
    T = X.shape[1]
    if kind == "heat":
        h = 1 + 0.5 * np.cos(2 * np.pi * np.arange(T) / T)
        return np.repeat(h[None], len(X), 0)
    if kind == "flat":
        return np.ones_like(X)
    return X / X.mean(1, keepdims=True)


def alpha_max_individual(X, H, C0):
    """Per enterprise: maximum alpha with X_i + alpha * H_i <= C0_i at all times (closed form)."""
    out = np.empty(len(X))
    for i in range(len(X)):
        pos = H[i] > 1e-12
        out[i] = max(0.0, np.min((C0[i] - X[i, pos]) / H[i, pos]))
    return out


def alpha_max_group(agg, Hagg, cap):
    """Maximum alpha with agg + alpha * Hagg <= cap at all times (Eq. (28))."""
    pos = Hagg > 1e-12
    return max(0.0, np.min((cap - agg[pos]) / Hagg[pos]))


def feasible_with_battery(load, cap, P, E):
    """True if shared storage can hold the joint import below ``cap`` in every month."""
    for mth in range(12):
        if not cap_is_feasible(load, MONTH_START[mth], MONTH_START[mth + 1], cap, P, E, ETA, ETA):
            return False
    return True


def alpha_max_group_battery(agg, Hagg, cap, P, E, hi):
    """Maximum common alpha with shared storage under the fixed import cap, by bisection."""
    lo = 0.0
    if not feasible_with_battery(agg, cap, P, E):
        return 0.0
    hi = max(hi, 1e-6)
    while feasible_with_battery(agg + hi * Hagg, cap, P, E) and hi < 50:
        hi *= 2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible_with_battery(agg + mid * Hagg, cap, P, E):
            lo = mid
        else:
            hi = mid
    return lo


def assess_park(X, margins=(0.10,)):
    """Hosting records (one per margin and profile) for a park with baseline loads X (n, T) in kW."""
    agg = X.sum(0)
    P = STORAGE_POWER_SHARE * agg.max()
    E = STORAGE_HOURS * P
    own = X.max(1)
    pk_bat = max(min_feasible_cap(agg, MONTH_START[k], MONTH_START[k + 1], P, E, ETA, ETA, 1e-7)[0]
                 for k in range(12))
    rows = []
    for margin in margins:
        C0 = own * (1 + margin)
        cap = C0.sum()
        base = dict(margin=margin, sumC0_kw=cap, group_peak_kw=agg.max(), group_peak_bat_kw=pk_bat,
                    release_kw=cap - agg.max(), release_rel=1 - agg.max() / cap,
                    release_bat_kw=cap - pk_bat, release_bat_rel=1 - pk_bat / cap,
                    energy_mwh=X.sum() * DT / 1e3)
        for kind in PROFILES:
            H = profile_shape(kind, X) * X.mean(1, keepdims=True)    # alpha = share of the enterprise's mean load
            a_i = alpha_max_individual(X, H, C0)
            e_unit = H.sum(1) * DT / 1e3                             # MWh per unit alpha, per enterprise
            a_g = alpha_max_group(agg, H.sum(0), cap)
            a_gb = alpha_max_group_battery(agg, H.sum(0), cap, P, E, hi=max(a_g, 0.05) * 2)
            rows.append(dict(base, profile=kind,
                             alpha_ind_uniform=a_i.min(),                    # smallest enterprise-specific coefficient
                             host_ind_mwh=float((a_i * e_unit).sum()),       # each enterprise up to its own limit
                             alpha_gta=a_g, host_gta_mwh=float(a_g * e_unit.sum()),
                             alpha_gta_bat=a_gb, host_gta_bat_mwh=float(a_gb * e_unit.sum())))
    return rows
