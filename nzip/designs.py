"""Named tariff-design variants and revenue calibration (Sections 3.5, 4.1.2, 4.2.2).

Dutch variants
  NLGTA            group transport agreement as legislated (original design)
  NL_no_uplift     local remedy 1: transformer uplift removed
  NL_only_catrule  local remedy 2: uplift removed and group fixed charges assigned by each enterprise's own category
  NLNEUTRAL        composition-neutral design on the reference schedule (scale s = 1)
  NLNEUTRAL_rev    composition-neutral design with the revenue-matched scale of Eq. (20)
German variants
  DE_legal         original two-part tariff with the legal 2,500 h rule
  DENEUTRAL        composition-neutral design on the upper-utilisation line (scale s = 1)
  DENEUTRAL_rev    composition-neutral design with the revenue-matched scale of Eq. (20)
"""
import numpy as np

from .game import GTAGame
from .parks import make_tariff, park_inputs
from .tariffs import GTATariff

NL_VARIANTS = {
    "NLGTA": dict(),
    "NL_no_uplift": dict(uplift=False),
    "NL_only_catrule": dict(uplift=False, fixed_by_level=False),
    "NLNEUTRAL": dict(kind="NLNEUTRAL"),
    "NLNEUTRAL_rev": dict(kind="NLNEUTRAL", ref_scale="rev"),
}
DE_NEUTRAL_VARIANTS = {"DENEUTRAL": 1.0, "DENEUTRAL_rev": "rev"}


def make_game(name, X, host, cats, levels, P_abs=0.0, scales=None, rate_mult=None, duration_h=2.0):
    """Build the park game of a named variant.

    X: (n, T) loads in kW; host: master-data row of the host site (German variants only); P_abs: storage power in kW
    (0 = no storage); scales: dict(nl=..., de=...) revenue-matched scales used by the ``_rev`` variants;
    rate_mult: optional (4, 3) multipliers on the Dutch (A, B, E) coefficients per category.
    """
    if name == "DE_legal":
        tf = make_tariff("DE", host)
        tf.de_battery = P_abs > 0                    # legal rule with peak-shaving dispatch only
        return GTAGame(X, tf, cats=cats, levels=levels, P_abs=P_abs, duration_h=duration_h)
    if name in DE_NEUTRAL_VARIANTS:
        tf = make_tariff("DE", host)
        tf.kind = "DENEUTRAL"
        tf.ref_scale = scales["de"] if DE_NEUTRAL_VARIANTS[name] == "rev" else DE_NEUTRAL_VARIANTS[name]
        return GTAGame(X, tf, cats=cats, levels=levels, P_abs=P_abs, duration_h=duration_h)
    kw = dict(NL_VARIANTS[name])
    if kw.get("ref_scale") == "rev":
        kw["ref_scale"] = scales["nl"]
    tf = GTATariff(**kw, rate_mult=rate_mult)
    return GTAGame(X, tf, cats=cats, levels=levels, P_abs=P_abs, duration_h=duration_h)


def calibrate_scales(L, m, parks):
    """Revenue-matched reference scales of Eq. (20), one per country, from the baseline without storage or added load.

    s_c = (sum over parks of grand-coalition savings under the original design) /
          (sum over parks of grand-coalition savings under the neutral design at s = 1).
    Because neutral savings scale linearly in s (Eq. (18)), this equates aggregate baseline revenue under the original
    and neutral designs (Eq. (19)). Returns dict(nl=..., de=...).
    """
    tot = dict.fromkeys(("NLGTA", "NLNEUTRAL", "DE_legal", "DENEUTRAL"), 0.0)
    for pk in parks:
        X, cats, levels = park_inputs(L, m, pk["members"])
        host = m.loc[pk["host"]]
        for name in tot:
            g = make_game(name, X, host, cats, levels)
            tot[name] += float(g.savings_masks(np.ones((1, g.n_players), bool))[0])
    return dict(nl=tot["NLGTA"] / tot["NLNEUTRAL"], de=tot["DE_legal"] / tot["DENEUTRAL"])
