"""Economic accounting of revenue-matched designs (Section 3.5, Eqs. (19)-(25); results in Section 4.2.3).

Input: the baseline table written by ``scripts/run_baseline_designs.py`` (one row per park and design variant, without
storage or added load, with the per-enterprise vectors ``standalone`` and ``lc``). Enterprise payments are
stand-alone bill minus allocated savings, Eq. (23), using the least-core allocation closest to the Shapley value.
"""
import numpy as np
import pandas as pd

# (country, original design, revenue-matched neutral design, neutral design at s = 1, column holding the scale)
PAIRS = [("Netherlands", "NLGTA", "NLNEUTRAL_rev", "NLNEUTRAL", "s_nl"),
         ("Germany", "DE_legal", "DENEUTRAL_rev", "DENEUTRAL", "s_de")]
PAYING_MORE_THRESHOLD = 0.001      # an enterprise pays more if its payment rises by more than 0.1% of the original


def economic_accounting(data, n_parks=500, n_split=2000, seed=0):
    """Park-, operator- and enterprise-level accounting for each country; returns a dict of data frames.

    summary     aggregate stand-alone bills, revenue (Eq. (19)), savings rates (Eq. (22)), median park savings,
                median enterprise bill change and share of enterprises paying more (Eqs. (23)-(24)) per design
    supplement  revenue-level effect, 5th / 95th percentile of park-level revenue change, total positive / negative
                bill changes, allocation-independent lower bound of Eq. (25) and the split-half calibration check
                of Section 3.5
    firms       enterprise-level payments under both designs
    parks       park-level savings and revenue
    The random split-half partitions use one generator (``seed``) continued across the two countries.
    """
    rng = np.random.default_rng(seed)
    summary, supplement, firms, parks = [], [], [], []
    for country, original, neutral, reference, scale_col in PAIRS:
        frames = {v: data.loc[data.variant.eq(v)].set_index("park").sort_index() for v in (original, neutral, reference)}
        a, b, r = (frames[v] for v in (original, neutral, reference))
        assert len(a) == len(b) == len(r) == n_parks and a.index.equals(b.index) and a.index.equals(r.index)
        for g in frames.values():
            assert not g.battery.astype(bool).any(), "accounting is defined for the baseline without storage"
            np.testing.assert_allclose(g.revenue, g.scale - g.vN, rtol=0, atol=1e-6)
        np.testing.assert_allclose(a.revenue.sum(), b.revenue.sum(), rtol=1e-9, atol=1e-6)   # revenue matching
        np.testing.assert_allclose(a.scale, b.scale, rtol=0, atol=1e-6)                       # stand-alone bills preserved
        ref_scale = float(b[scale_col].iloc[0])
        np.testing.assert_allclose(b.vN, ref_scale * r.vN, rtol=1e-8, atol=1e-6)               # Eq. (18)
        pay, max_balance = {}, 0.0
        for variant, g in ((original, a), (neutral, b)):
            pieces = []
            for park, row in g.iterrows():
                sa, alloc = np.asarray(row.standalone, float), np.asarray(row.lc, float)
                payment = sa - alloc                                           # Eq. (23)
                balance = abs(payment.sum() - row.revenue)                     # budget balance with Eq. (19)
                assert balance <= max(1e-6, 1e-9 * row.scale)
                max_balance = max(max_balance, balance)
                pieces.extend((int(park), i, float(x), float(y)) for i, (x, y) in enumerate(zip(sa, payment)))
                parks.append(dict(country=country, variant=variant, park=int(park), standalone_eur=float(row.scale),
                                  savings_eur=float(row.vN), revenue_eur=float(row.revenue),
                                  savings_rate_pct=100 * float(row.save_rel)))
            pay[variant] = pd.DataFrame(pieces, columns=["park", "firm", "standalone", "payment"]).set_index(["park", "firm"])
        fa, fb = pay[original], pay[neutral]
        assert fa.index.equals(fb.index)
        assert (fa.payment > 0).all(), "relative changes require positive original allocated payments"
        delta = fb.payment - fa.payment                                        # Eq. (24)
        rel = delta / fa.payment
        firms.append(pd.DataFrame(dict(country=country, park=fa.index.get_level_values(0), firm=fa.index.get_level_values(1),
                                       standalone_eur=fa.standalone.to_numpy(), original_payment_eur=fa.payment.to_numpy(),
                                       neutral_payment_eur=fb.payment.to_numpy(), change_eur=delta.to_numpy(),
                                       change_pct=100 * rel.to_numpy(),
                                       paying_more=(rel > PAYING_MORE_THRESHOLD).to_numpy())))
        for variant, g in ((original, a), (neutral, b)):
            is_original = variant == original
            summary.append(dict(
                country=country, design="original" if is_original else "neutral, revenue-matched", variant=variant,
                n_parks=len(g), n_firms=len(fa), reference_capacity_scale=np.nan if is_original else ref_scale,
                standalone_eur=float(g.scale.sum()), revenue_eur=float(g.revenue.sum()),
                revenue_index=100 * float(g.revenue.sum() / a.revenue.sum()), savings_eur=float(g.vN.sum()),
                aggregate_savings_pct=100 * float(g.vN.sum() / g.scale.sum()),        # Eq. (22)
                median_park_savings_pct=100 * float(g.save_rel.median()),
                median_firm_bill_change_pct=np.nan if is_original else 100 * float(rel.median()),
                firms_paying_more_pct=np.nan if is_original else 100 * float((rel > PAYING_MORE_THRESHOLD).mean()),
                nonpositive_allocated_bills=int((pay[variant].payment <= 0).sum())))
        # split-half check: fit the scale on 250 random parks, evaluate revenue error on the other half (Section 3.5)
        errors = []
        half = n_parks // 2
        for _ in range(n_split):
            perm = rng.permutation(n_parks)
            train, test = perm[:half], perm[half:]
            fitted = a.vN.iloc[train].sum() / r.vN.iloc[train].sum()
            errors.append(100 * ((a.scale.iloc[test].sum() - fitted * r.vN.iloc[test].sum()) / a.revenue.iloc[test].sum() - 1))
        park_change = b.revenue.to_numpy() / a.revenue.to_numpy() - 1             # park-level revenue change
        lower = float((a.vN - b.vN).clip(lower=0).sum())                          # Eq. (25), allocation-independent bound
        up, down = float(delta.clip(lower=0).sum()), float(-delta.clip(upper=0).sum())
        assert up + max_balance * n_parks >= lower
        supplement.append(dict(
            country=country, reference_capacity_scale=ref_scale, original_total_revenue_eur=float(a.revenue.sum()),
            neutral_total_revenue_eur=float(b.revenue.sum()),
            aggregate_revenue_change_pct=100 * float(b.revenue.sum() / a.revenue.sum() - 1),
            park_revenue_change_p5_pct=100 * float(np.quantile(park_change, 0.05)),
            park_revenue_change_p95_pct=100 * float(np.quantile(park_change, 0.95)),
            positive_bill_changes_eur=up, negative_bill_changes_eur=down, eq25_lower_bound_eur=lower,
            allocation_dependent_gap_eur=up - lower, split_half_repetitions=n_split,
            held_out_revenue_error_p5_pct=float(np.percentile(errors, 5)),
            held_out_revenue_error_median_pct=float(np.median(errors)),
            held_out_revenue_error_p95_pct=float(np.percentile(errors, 95))))
    return dict(summary=pd.DataFrame(summary), supplement=pd.DataFrame(supplement),
                firms=pd.concat(firms, ignore_index=True), parks=pd.DataFrame(parks))
