"""Coalition cost game of an industrial park under the evaluated transport-tariff designs (Sections 3.2-3.4).

Players are the enterprises (measured 15-minute loads) and, optionally, a shared-storage operator with its own
connection (fixed power P and energy E in every coalition, physical level MS). The cooperation savings of a coalition
S are v(S) = sum of stand-alone bills - c(S) (Eq. (7)).

Designs
-------
NLGTA       Dutch group transport agreement as legislated: joint annual contracted capacity and joint monthly maxima on
            the group's net profile; group category = highest physical connection level; transformer uplift on the
            own monthly maxima of MS/LS-level members when the group level is higher; group fixed charges by physical
            level. A single enterprise is billed in its own capacity category.
NLNEUTRAL   Composition-neutral Dutch design: every coalition is billed on the common reference (MS) schedule, plus
            fixed member adjustments D_i = own-category stand-alone bill - reference stand-alone bill (Eqs. (13)-(14)).
DE          German original: whole group behind one meter, host two-part schedule, legal 2,500 h rule.
DENEUTRAL   Composition-neutral German design: common upper-utilisation line for every coalition plus D_i.
Battery dispatch is the bill-minimising dispatch of the respective design (``kernels.py``).
"""
from dataclasses import dataclass, field

import numpy as np

from .kernels import DT, MONTH_START, batch_costs
from .tariffs import (BATTERY_LEVEL, CAT_A, CAT_B, CAT_E, CAT_FIX, HIGH_MASK, LS_E_HIGH, LS_E_LOW, NREF, REF,
                      GTATariff, capacity_category)


@dataclass
class GTAGame:
    loads: np.ndarray                     # (n, T) kW, one row per enterprise
    tariff: GTATariff = field(default_factory=GTATariff)
    p_ratio: float = 0.0                  # battery power as a share of the raw grand-coalition peak
    duration_h: float = 2.0               # battery duration (energy = power x duration)
    eta_rt: float = 0.9                   # round-trip efficiency
    cats: np.ndarray = None               # individual capacity categories (default: from own peak)
    levels: np.ndarray = None             # physical connection levels (default: equal to the categories)
    P_abs: float = None                   # absolute battery power (kW); overrides p_ratio when given
    rtol: float = 1e-7                    # relative tolerance of the cap searches
    iters: int = 40                       # golden-section iterations

    def __post_init__(self):
        self.loads = np.ascontiguousarray(self.loads, dtype=np.float64)
        self.n = self.loads.shape[0]
        own = self.loads.max(1)
        self.cats = capacity_category(own) if self.cats is None else np.asarray(self.cats, int)
        self.levels = self.cats.copy() if self.levels is None else np.asarray(self.levels, int)
        self.P_bat = float(self.P_abs) if self.P_abs is not None else self.p_ratio * self.loads.sum(0).max()
        self.E_bat = self.P_bat * self.duration_h
        self.has_bat = self.P_bat > 0
        if self.tariff.kind == "DE" and self.has_bat and not self.tariff.de_battery:
            raise ValueError("DE with storage requires de_battery=True (peak-shaving dispatch only)")
        self.n_players = self.n + int(self.has_bat)
        self.eta = np.sqrt(self.eta_rt)
        self.ind_peaks = np.stack([[self.loads[i, MONTH_START[k]:MONTH_START[k + 1]].max() for k in range(12)]
                                   for i in range(self.n)])
        self.ind_energy = self.loads.sum(1) * DT
        self.ind_e_hi = self.loads[:, HIGH_MASK].sum(1) * DT
        self.ind_e_lo = self.ind_energy - self.ind_e_hi
        # fixed charges: stand-alone by own capacity category; inside a group by physical level; the storage
        # connection's own fixed charge is paid regardless and cancels from the incremental game (omitted)
        self.fix_own = CAT_FIX[self.cats]
        self.fix_level = CAT_FIX[self.levels]
        # rate table: rows 0-3 = categories (optionally perturbed), row NREF = neutral reference (REF, scaled capacity)
        mult = np.ones((4, 3)) if self.tariff.rate_mult is None else np.asarray(self.tariff.rate_mult, float)
        self.A = np.r_[CAT_A * mult[:, 0], CAT_A[REF] * mult[REF, 0] * self.tariff.ref_scale]
        self.B = np.r_[CAT_B * mult[:, 1], CAT_B[REF] * mult[REF, 1] * self.tariff.ref_scale]
        self.E = np.r_[CAT_E * mult[:, 2], CAT_E[REF] * mult[REF, 2]]
        self.ls_e = (LS_E_HIGH * mult[0, 2], LS_E_LOW * mult[0, 2])
        self.uplift = max(self.A[1] - self.A[2], 0.0) / 12      # EUR/kW/month (Art. 3.15(2)); follows perturbed A
        if self.tariff.kind == "NLNEUTRAL":
            eye = np.eye(self.n, dtype=bool)
            z = np.zeros(self.n)
            own_b = self._nl_bill(self.ind_peaks, z, eye, self.cats)
            ref_b = self._nl_bill(self.ind_peaks, z, eye, np.full(self.n, NREF))
            self.adj = own_b - ref_b          # member adjustments D_i of Eq. (13), excluding fixed charges
        if self.tariff.kind == "DENEUTRAL":
            # German analogue: every coalition on the host's upper-utilisation line (capacity price scaled by
            # ref_scale), plus D_i = legal stand-alone bill - reference-line stand-alone bill
            tf = self.tariff
            C, En = self.loads.max(1), self.ind_energy
            legal = np.where(En >= 2500.0 * C, tf.P_over * C + tf.e_over * En, tf.P_under * C + tf.e_under * En)
            self.adj = legal - (tf.ref_scale * tf.P_over * C + tf.e_over * En)
        self.standalone = self.cost_masks(np.eye(self.n_players, dtype=bool))

    def _nl_bill(self, peaks, chg, firm, cat):
        """Variable Dutch bill of coalitions (rows of ``firm``) at category ``cat`` (one per row)."""
        e_mem = firm @ self.ind_energy
        e_ls = self.ls_e[0] * (firm @ self.ind_e_hi) + self.ls_e[1] * (firm @ self.ind_e_lo)   # LS groups hold no storage
        energy_cost = np.where(cat == 0, e_ls, self.E[cat] * (e_mem + chg))
        return self.A[cat] * peaks.max(1) + self.B[cat] * peaks.sum(1) + energy_cost

    def coalition_category(self, firm, bat):
        """NLGTA: group category = highest physical level among its connections (storage connection at MS);
        a single enterprise without other connections keeps its own capacity category."""
        nconn = firm.sum(1) + bat.astype(int)
        lvl = np.where(firm, self.levels[None, :], -1).max(1)
        lvl = np.where(bat, np.maximum(lvl, BATTERY_LEVEL), lvl)
        single = (nconn == 1) & firm.any(1)
        own = np.where(firm, self.cats[None, :], -1).max(1)
        return np.where(single, own, lvl), single

    def cost_masks(self, masks):
        """Annual transport bill (variable + fixed charges) of each coalition; masks has shape (B, n_players)."""
        masks = np.asarray(masks, bool)
        firm = np.ascontiguousarray(masks[:, :self.n])
        bat = masks[:, self.n].copy() if self.has_bat else np.zeros(len(masks), bool)
        k = self.tariff.kind
        if k in ("NLGTA", "NLNEUTRAL"):
            if k == "NLGTA":
                cat, single = self.coalition_category(firm, bat)
            else:
                cat = np.full(len(masks), NREF)
                single = np.zeros(len(masks), bool)
            peaks, chg = batch_costs(firm, bat, self.loads, self.P_bat, self.E_bat, self.eta, self.eta, 0,
                                     self.A[cat], self.B[cat], self.E[cat], 0.0, 0.0, 0.0, 0.0, MONTH_START,
                                     self.rtol, self.iters)
            variable = self._nl_bill(peaks, chg, firm, cat)
            if k == "NLGTA":
                low = self.levels == 1                                  # MS/LS-level members (uplift class)
                up = self.uplift * (firm[:, low].astype(float) @ self.ind_peaks[low].sum(1))
                if self.tariff.uplift:
                    variable = variable + np.where((cat >= 2) & ~single, up, 0.0)
                fixed = np.where(single | (not self.tariff.fixed_by_level), firm @ self.fix_own, firm @ self.fix_level)
            else:
                variable = variable + firm @ self.adj
                fixed = firm @ self.fix_own        # composition-neutral: own-category fixed charges always
            cost = variable + fixed
        elif k == "DENEUTRAL":
            # one reference line for every coalition: both lines of the German kernel are set equal, so the cap search
            # is a plain convex search on [lowest feasible cap, raw peak]
            tf = self.tariff
            Pr = tf.ref_scale * tf.P_over
            _, bill = batch_costs(firm, bat, self.loads, self.P_bat, self.E_bat, self.eta, self.eta, 1,
                                  np.zeros(len(masks)), np.zeros(len(masks)), np.zeros(len(masks)),
                                  Pr, tf.e_over, Pr, tf.e_over, MONTH_START, self.rtol, self.iters)
            cost = bill + firm @ self.adj
        elif k == "DE":
            tf = self.tariff
            _, bill = batch_costs(firm, bat, self.loads, self.P_bat, self.E_bat, self.eta, self.eta, 1,
                                  np.zeros(len(masks)), np.zeros(len(masks)), np.zeros(len(masks)),
                                  tf.P_over, tf.e_over, tf.P_under, tf.e_under, MONTH_START, self.rtol, self.iters)
            cost = bill
        else:
            raise ValueError(k)
        return np.where(firm.any(1), cost, 0.0)

    def savings_masks(self, masks):
        """Cooperation savings v(S) of the coalitions in ``masks`` (Eq. (7))."""
        masks = np.asarray(masks, bool)
        return masks @ self.standalone - self.cost_masks(masks)

    def savings_all(self, step=2048):
        """Savings of all 2^p coalitions, indexed by bitmask (player i = bit i)."""
        p = self.n_players
        idx = np.arange(2 ** p, dtype=np.int64)
        out = np.empty(2 ** p)
        for s in range(0, 2 ** p, step):
            m = ((idx[s:s + step, None] >> np.arange(p)) & 1).astype(bool)
            out[s:s + step] = self.savings_masks(m)
        return out
