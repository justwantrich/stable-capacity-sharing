"""Tariff constants, connection classification and the tariff-design specification (Sections 3.2.2-3.2.3, 4.1.2).

Dutch coefficients: Liander 2026 transport tariffs for large-user connections (public document cited in the
manuscript), combined with the group-billing provisions of the Tarievencode elektriciteit 2026 (Arts. 3.1, 3.8, 3.15).
German coefficients are not hard-coded: they are the two-part schedule of the designated host site and are read from the
public VEA master data (see ``parks.make_tariff``).
"""
from dataclasses import dataclass

import numpy as np

from .kernels import T

# index: 0 = LS (<= 50 kW), 1 = MS/LS (> 50-136 kW), 2 = MS (> 136-2000 kW), 3 = TS/MS (> 2000 kW)
CAT_NAMES = ["LS", "MS/LS", "MS", "TS/MS"]
CAT_A = np.array([1.49, 3.74, 2.29, 3.99]) * 12      # EUR/kW/yr on the annual contracted capacity
CAT_B = np.array([0.0, 3.57, 3.57, 6.24])            # EUR/kW per month on the monthly maximum
CAT_E = np.array([0.0806, 0.0226, 0.0226, 0.0])      # EUR/kWh (LS: peak-hour rate; LS is not used for individuals)
CAT_FIX = np.array([1.50, 36.75, 36.75, 230.0]) * 12  # EUR/yr fixed transport charge per connection
LS_E_HIGH, LS_E_LOW = 0.0806, 0.0430                 # LS energy tariff: high 07-23 h Monday-Friday, low otherwise
UPLIFT = 3.74 - 2.29                                 # EUR/kW/month, transformer uplift (Art. 3.15(2))
REF = 2                                              # reference category of the neutral design (MS)
NREF = 4                                             # row of the neutral reference in the game's rate table
BATTERY_LEVEL = 2                                    # shared battery connected at MS level (modelling assumption)

_t = np.arange(T)
_HOUR = (_t % 96) // 4
_WEEKDAY = ((_t // 96) + 4) % 7                      # 1 January 2016 is a Friday (0 = Monday)
HIGH_MASK = (_WEEKDAY < 5) & (_HOUR >= 7) & (_HOUR < 23)


def capacity_category(contract_kw):
    """Individual tariff category from contracted capacity (own annual peak)."""
    contract_kw = np.asarray(contract_kw, float)
    return np.select([contract_kw > 2000, contract_kw > 136, contract_kw > 50], [3, 2, 1], 0)


def physical_level(grid_level, contract_kw):
    """Physical connection level from the recorded grid level; connections above 2 MW are TS/MS and unknown levels
    are mapped to medium voltage (modelling assumptions of Section 4.1.2)."""
    base = np.array([{"NS": 0, "MS/NS": 1, "MS": 2}.get(g, 2) for g in grid_level])
    return np.where(np.asarray(contract_kw) > 2000, 3, base)


@dataclass
class GTATariff:
    """Tariff-design specification.

    kind: "NLGTA" (Dutch group transport agreement as legislated), "NLNEUTRAL" (composition-neutral Dutch design),
          "DE" (German original with the 2,500 h rule), "DENEUTRAL" (composition-neutral German design).
    """
    kind: str = "NLGTA"
    # Local remedies of Section 4.2.2 (defaults = legislated rule)
    uplift: bool = True            # NLGTA: transformer uplift for MS/LS-level members of a higher-level group
    fixed_by_level: bool = True    # NLGTA: group fixed charges by physical level (True) or by own category (False)
    # Neutral designs
    ref_scale: float = 1.0         # scale on the reference capacity charges (revenue calibration, Eq. (20))
    # Tariff-rate perturbation (Section 4.1.3): multipliers on (A, B, E) per category, shape (4, 3); the uplift follows A
    rate_mult: tuple = None
    # German original with storage: peak-shaving dispatch only (no deliberate cycling or peak raising)
    de_battery: bool = False
    # German host schedule (EUR/kW/yr, EUR/kWh) for utilisation >= / < 2,500 h
    P_over: float = 80.0
    e_over: float = 0.010
    P_under: float = 14.0
    e_under: float = 0.037

    def lp_representable(self):
        """True for the designs whose coalition cost is a linear programme with additive resources (Eq. (16))."""
        return self.kind in ("NLNEUTRAL", "DENEUTRAL")
