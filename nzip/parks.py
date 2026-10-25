"""Construction of site-disjoint synthetic parks, electrification profile and park inputs (Sections 3.6, 4.1.1, 4.1.3).

Parks are constructed analytical samples built from measured sites; they are not observations of existing parks.
"""
import numpy as np

from .tariffs import GTATariff, capacity_category, physical_level


def disjoint_parks(m, rng, n_parks, nmin, nmax, min_kw=50.0):
    """Site-disjoint parks from sites with annual peak >= ``min_kw``.

    Each park has a size drawn uniformly from nmin ... nmax and a composition rule drawn independently: regional
    (same two-digit postal code, probability 0.50), same sector (0.25) or unrestricted random (0.25). For regional and
    sector rules one group with enough remaining sites is chosen at random (otherwise the rule reverts to random
    sampling); members are drawn without replacement and removed from the pool. One member is drawn as the host whose
    tariff schedule is used in the German scenario.
    """
    pool = set(m.index[m.peak_kw >= min_kw])
    parks = []
    for k in range(n_parks):
        n = int(rng.integers(nmin, nmax + 1))
        comp = rng.choice(["regional", "regional", "sector", "random"])
        avail = m.loc[sorted(pool)]
        if comp == "regional":
            cnt = avail.zip_code.value_counts()
            ok = cnt.index[cnt >= n]
            if len(ok) == 0:
                comp = "random"
            else:
                z = rng.choice(ok)
                cand = avail.index[avail.zip_code == z].to_numpy()
        if comp == "sector":
            cnt = avail.sector_group_id.value_counts()
            ok = cnt.index[cnt >= n]
            if len(ok) == 0:
                comp = "random"
            else:
                s = rng.choice(ok)
                cand = avail.index[avail.sector_group_id == s].to_numpy()
        if comp == "random":
            cand = avail.index.to_numpy()
        if len(cand) < n:
            break
        mem = rng.choice(cand, n, replace=False)
        pool -= set(mem.tolist())
        parks.append(dict(members=[int(x) for x in mem], composition=str(comp), host=int(rng.choice(mem))))
    return parks


STUDY_SEED, STUDY_PARKS, STUDY_SIZES = 1, 500, (4, 10)     # the 500 matched parks used by every experiment


def study_parks(m):
    """The matched sample of 500 site-disjoint parks (4-10 members) shared by all experiments."""
    return disjoint_parks(m, np.random.default_rng(STUDY_SEED), STUDY_PARKS, *STUDY_SIZES)


def electrify(loads, alpha):
    """Add the winter-weighted component alpha * (mean load) * (1 + 0.5 cos(2 pi t / T)) to every member, Eq. (30)."""
    t = np.arange(loads.shape[1])
    season = 1 + 0.5 * np.cos(2 * np.pi * t / loads.shape[1])
    return loads + alpha * loads.mean(1, keepdims=True) * season


def make_tariff(kind, host):
    """Tariff specification of a design; the German schedule is read from the host row of the master data."""
    if kind == "DE":
        return GTATariff("DE", P_over=host.capacity_price_over_2500h_eur_per_kw,
                         e_over=host.energy_price_over_2500h_eur_per_kwh,
                         P_under=host.capacity_price_under_2500h_eur_per_kw,
                         e_under=host.energy_price_under_2500h_eur_per_kwh)
    return GTATariff(kind)


def park_inputs(L, m, members):
    """Loads (kW), individual capacity categories and physical connection levels of a park's members."""
    X = np.asarray(L[members], dtype=np.float64)
    own = X.max(1)
    return X, capacity_category(own), physical_level(m.loc[members, "grid_level"].tolist(), own)
