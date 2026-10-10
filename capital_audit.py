#!/usr/bin/env python3
"""
capital_audit.py
================

Capital allocation audit: legacy studio slate vs. micro-budget independent slate.

Scope
-----
Converts title-level production, marketing, and box-office data into
producer-realized cash flows, then evaluates capital efficiency, recovery
velocity, and demand-decay metrics for two cohorts (``LEGACY`` and ``MICRO``).

Provenance control
------------------
The built-in generator produces SYNTHETIC data from explicit, user-editable
parameters (``SegmentParams``). Synthetic output is a conditional scenario, not
an observation. Every report produced from generated data is stamped as such.
For empirical use, pass a CSV via ``--input`` that follows the canonical schema
documented in ``README.md``; the metric engine is agnostic to data origin.

Units: INR crore (Rs Cr) unless stated otherwise.

Usage
-----
    python capital_audit.py                       # synthetic scenario
    python capital_audit.py --input titles.csv    # empirical dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

LOG = logging.getLogger("capital_audit")

# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #
WEEKS: int = 8
GROSS_COLS: list[str] = [f"wk_gross_{i}" for i in range(1, WEEKS + 1)]
SEGMENTS: tuple[str, str] = ("LEGACY", "MICRO")
ID_COLS: list[str] = ["title_id", "segment"]
NUMERIC_COLS: list[str] = [
    "production_budget_cr",
    "marketing_cr",
    "distribution_overhead_cr",
    "cash_cycle_months",
    *GROSS_COLS,
    "overseas_gross_cr",
    "non_theatrical_cr",
]
REQUIRED_COLS: list[str] = [*ID_COLS, *NUMERIC_COLS]

PALETTE = {
    "LEGACY": "#ff4d6d",
    "MICRO": "#2ec4b6",
    "bg": "#0b0f14",
    "panel": "#121821",
    "grid": "#222c3a",
    "text": "#d8dee9",
    "muted": "#8d99ae",
    "accent": "#f4d35e",
}
SEGMENT_LABEL = {"LEGACY": "Legacy studio slate", "MICRO": "Micro-budget independent slate"}


class SchemaError(ValueError):
    """Raised when an input dataset violates the canonical schema."""


# --------------------------------------------------------------------------- #
# Parameter objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReleaseEconomics:
    """
    Contractual cash-flow waterfall between headline gross and producer receipts.

    All values are scenario parameters. Actual terms are contract-specific and
    should be overridden with deal-level data where available.

    Attributes
    ----------
    gst_rate:
        Blended GST wedge between gross and nett collections (tickets are taxed
        at tiered rates; this is a single blended approximation).
    distributor_share_by_week:
        Share of nett remitted to the distributor/producer side, by release week.
        Decays with run length (multiplex revenue-share schedules are
        front-loaded in favour of the producer side, then taper).
    distributor_commission:
        Fraction of the distributor share retained by the distribution
        intermediary before cash reaches the producer.
    overseas_realization:
        Producer-side realization on overseas gross after exhibitor split,
        sub-distributor fees and remittance friction.
    """

    gst_rate: float = 0.16
    distributor_share_by_week: tuple[float, ...] = (0.50, 0.425, 0.375, 0.30, 0.30, 0.30, 0.30, 0.30)
    distributor_commission: float = 0.10
    overseas_realization: float = 0.40

    def __post_init__(self) -> None:
        if len(self.distributor_share_by_week) != WEEKS:
            raise ValueError(f"distributor_share_by_week must have {WEEKS} entries.")
        scalars = [self.gst_rate, self.distributor_commission, self.overseas_realization,
                   *self.distributor_share_by_week]
        if any(not 0.0 <= s <= 1.0 for s in scalars):
            raise ValueError("All ReleaseEconomics rates must lie in [0, 1].")


@dataclass(frozen=True)
class OutcomeTier:
    """
    One component of a worldwide-gross outcome mixture.

    ``median_multiple`` is worldwide gross / production budget. ``tail`` marks
    right-tail tiers for stress scaling in the sensitivity surface.
    """

    name: str
    prob: float
    median_multiple: float
    sigma: float
    tail: bool = False


@dataclass(frozen=True)
class SegmentParams:
    """Generative assumptions for one cohort. Every field is an explicit assumption."""

    label: str
    budget_median_cr: float
    budget_sigma: float
    marketing_pct: tuple[float, float]       # of production budget
    overhead_pct: tuple[float, float]        # distribution bloat, of production budget
    tiers: tuple[OutcomeTier, ...]
    overseas_share: tuple[float, float]      # of worldwide gross
    retention_mean: float                    # mean week-over-week demand retention
    retention_concentration: float           # Beta concentration (higher = tighter)
    nt_sale_prob: float                      # P(non-theatrical / digital yield realized)
    nt_median_pct: float                     # median yield, fraction of budget, at 1.0x gross
    nt_sigma: float
    nt_perf_elasticity: float                # sensitivity of yield to theatrical outcome
    cycle_months: tuple[float, float]        # capital outstanding, start to final receipt

    def with_tail_stress(self, dud_prob: float, tail_scale: float) -> "SegmentParams":
        """Return a copy with the DUD tier probability and tail-tier medians re-set."""
        if not 0.0 < dud_prob < 1.0:
            raise ValueError("dud_prob must lie in (0, 1).")
        dud = next(t for t in self.tiers if t.name == "DUD")
        rest = [t for t in self.tiers if t.name != "DUD"]
        mass = sum(t.prob for t in rest)
        tiers = [replace(dud, prob=dud_prob)]
        for t in rest:
            tiers.append(replace(
                t,
                prob=t.prob / mass * (1.0 - dud_prob),
                median_multiple=t.median_multiple * (tail_scale if t.tail else 1.0),
            ))
        return replace(self, tiers=tuple(tiers))


# Default scenario. Encodes the structural premises under test: high fixed
# marketing intensity, front-loaded demand and thin tail for the legacy slate;
# low fixed cost, higher retention, performance-linked digital yield and a
# heavy right tail for the micro slate. These are premises, not measurements.
DEFAULT_LEGACY = SegmentParams(
    label="LEGACY",
    budget_median_cr=150.0, budget_sigma=0.45,
    marketing_pct=(0.15, 0.30), overhead_pct=(0.06, 0.14),
    tiers=(
        OutcomeTier("DUD", 0.35, 0.50, 0.35),
        OutcomeTier("PAR", 0.35, 1.20, 0.30),
        OutcomeTier("HIT", 0.24, 2.40, 0.30),
        OutcomeTier("MEGA", 0.06, 5.00, 0.30, tail=True),
    ),
    overseas_share=(0.10, 0.30),
    retention_mean=0.45, retention_concentration=20.0,
    nt_sale_prob=0.90, nt_median_pct=0.55, nt_sigma=0.35, nt_perf_elasticity=0.15,
    cycle_months=(22.0, 34.0),
)

DEFAULT_MICRO = SegmentParams(
    label="MICRO",
    budget_median_cr=2.0, budget_sigma=0.55,
    marketing_pct=(0.05, 0.15), overhead_pct=(0.02, 0.06),
    tiers=(
        OutcomeTier("DUD", 0.55, 0.15, 0.60),
        OutcomeTier("MODEST", 0.30, 1.00, 0.40),
        OutcomeTier("BREAKOUT", 0.13, 6.00, 0.70, tail=True),
        OutcomeTier("OUTLIER", 0.02, 45.00, 0.80, tail=True),
    ),
    overseas_share=(0.0, 0.08),
    retention_mean=0.60, retention_concentration=15.0,
    nt_sale_prob=0.60, nt_median_pct=0.50, nt_sigma=0.80, nt_perf_elasticity=0.70,
    cycle_months=(8.0, 16.0),
)


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
def simulate_cohort(p: SegmentParams, n: int, rng: np.random.Generator, id_prefix: str) -> pd.DataFrame:
    """
    Generate ``n`` synthetic titles in the canonical schema.

    Worldwide gross is drawn from a lognormal mixture over ``p.tiers`` applied
    to production budget. India gross is spread across ``WEEKS`` with geometric
    retention ``h ~ Beta``. Non-theatrical yield is conditioned on the
    theatrical outcome through a power-law elasticity.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    budget = rng.lognormal(np.log(p.budget_median_cr), p.budget_sigma, n)
    marketing = budget * rng.uniform(*p.marketing_pct, n)
    overhead = budget * rng.uniform(*p.overhead_pct, n)

    probs = np.array([t.prob for t in p.tiers], dtype=float)
    probs /= probs.sum()
    tier_idx = rng.choice(len(p.tiers), size=n, p=probs)
    med = np.array([t.median_multiple for t in p.tiers])[tier_idx]
    sig = np.array([t.sigma for t in p.tiers])[tier_idx]
    multiple = med * rng.lognormal(0.0, 1.0, n) ** sig
    ww_gross = budget * multiple

    overseas = ww_gross * rng.uniform(*p.overseas_share, n)
    india = ww_gross - overseas

    k = p.retention_concentration
    hold = np.clip(rng.beta(p.retention_mean * k, (1 - p.retention_mean) * k, n), 0.05, 0.95)
    geom = hold[:, None] ** np.arange(WEEKS)[None, :]
    weekly = india[:, None] * geom / geom.sum(axis=1, keepdims=True)

    sale = rng.random(n) < p.nt_sale_prob
    nt = (budget * p.nt_median_pct * np.maximum(multiple, 1e-6) ** p.nt_perf_elasticity
          * rng.lognormal(0.0, p.nt_sigma, n) * sale)

    out = pd.DataFrame({
        "title_id": [f"{id_prefix}-{i:05d}" for i in range(1, n + 1)],
        "segment": p.label,
        "production_budget_cr": budget,
        "marketing_cr": marketing,
        "distribution_overhead_cr": overhead,
        "cash_cycle_months": rng.uniform(*p.cycle_months, n),
    })
    out[GROSS_COLS] = weekly
    out["overseas_gross_cr"] = overseas
    out["non_theatrical_cr"] = nt
    return out


def simulate_market(n_legacy: int, n_micro: int, rng: np.random.Generator,
                    legacy: SegmentParams = DEFAULT_LEGACY,
                    micro: SegmentParams = DEFAULT_MICRO) -> pd.DataFrame:
    """Generate both cohorts and concatenate into one canonical frame."""
    return pd.concat(
        [simulate_cohort(legacy, n_legacy, rng, "LEG"), simulate_cohort(micro, n_micro, rng, "MIC")],
        ignore_index=True,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and type-coerce an input frame. Returns a cleaned copy.

    Raises
    ------
    SchemaError
        On missing columns, non-numeric/NaN values, negatives, unknown
        segments, duplicate IDs, non-positive budgets or cycle times.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    out = df[REQUIRED_COLS].copy()
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    bad_nan = [c for c in NUMERIC_COLS if out[c].isna().any()]
    if bad_nan:
        raise SchemaError(f"Non-numeric or null values in: {bad_nan}")
    bad_neg = [c for c in NUMERIC_COLS if (out[c] < 0).any()]
    if bad_neg:
        raise SchemaError(f"Negative values in: {bad_neg}")
    if not set(out["segment"].unique()) <= set(SEGMENTS):
        raise SchemaError(f"segment must be one of {SEGMENTS}; got {sorted(out['segment'].unique())}")
    if out["title_id"].duplicated().any():
        raise SchemaError("Duplicate title_id values.")
    if (out["production_budget_cr"] <= 0).any():
        raise SchemaError("production_budget_cr must be strictly positive.")
    if (out["cash_cycle_months"] <= 0).any():
        raise SchemaError("cash_cycle_months must be strictly positive.")
    return out


# --------------------------------------------------------------------------- #
# Metric engine
# --------------------------------------------------------------------------- #
def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise division returning NaN where the denominator is zero."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=den != 0)
    return out


def estimate_weekly_decay(weekly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-title log-linear OLS of weekly gross against week index.

    Returns ``(lambda, hold)`` where ``ln(g_w) = a - lambda * w`` and
    ``hold = exp(-lambda)`` is the implied week-over-week retention. Weeks with
    zero gross are excluded; titles with fewer than three positive weeks
    return NaN.
    """
    x = np.arange(weekly.shape[1], dtype=float)[None, :]
    mask = weekly > 0
    y = np.log(np.where(mask, weekly, 1.0))
    n = mask.sum(axis=1).astype(float)
    sx, sy = (x * mask).sum(1), (y * mask).sum(1)
    sxy, sxx = (x * y * mask).sum(1), (x * x * mask).sum(1)
    denom = n * sxx - sx**2
    slope = np.full(weekly.shape[0], np.nan)
    ok = (n >= 3) & (denom > 0)
    slope[ok] = (n[ok] * sxy[ok] - sx[ok] * sy[ok]) / denom[ok]
    lam = np.clip(-slope, 0.0, None)
    return lam, np.exp(-lam)


class CapitalAuditEngine:
    """Converts canonical title-level data into producer-realized capital metrics."""

    def __init__(self, economics: ReleaseEconomics | None = None) -> None:
        self.econ = economics or ReleaseEconomics()

    def compute(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Compute title-level metrics.

        Definitions (INR crore)
        -----------------------
        C_T      = production + marketing + distribution overhead
        R        = producer theatrical receipts + non-theatrical/digital yield
        P_net    = R - C_T
        ROI      = P_net / C_T
        CEI      = R / C_T                       (realized yield per Rs 1 Cr invested)
        BVS      = 100 * (marketing / C_T) * (1 - h)
        Velocity = CEI * 12 / cash_cycle_months  (capital turns per year)
        """
        df = validate_schema(raw)
        e = self.econ

        wk = df[GROSS_COLS].to_numpy(float)
        share = np.asarray(e.distributor_share_by_week)[None, :]
        india_gross = wk.sum(axis=1)
        nett_wk = wk * (1.0 - e.gst_rate)

        gst_leak = india_gross * e.gst_rate
        dist_pool = (nett_wk * share).sum(axis=1)
        exhibitor_leak = (nett_wk * (1.0 - share)).sum(axis=1)
        commission_leak = dist_pool * e.distributor_commission
        india_producer = dist_pool - commission_leak

        os_gross = df["overseas_gross_cr"].to_numpy(float)
        os_producer = os_gross * e.overseas_realization
        os_leak = os_gross - os_producer

        gross_ww = india_gross + os_gross
        producer_theatrical = india_producer + os_producer
        nt = df["non_theatrical_cr"].to_numpy(float)
        receipts = producer_theatrical + nt

        prod = df["production_budget_cr"].to_numpy(float)
        mkt = df["marketing_cr"].to_numpy(float)
        ovh = df["distribution_overhead_cr"].to_numpy(float)
        c_t = prod + mkt + ovh
        p_net = receipts - c_t

        lam, hold = estimate_weekly_decay(wk)
        realization = _safe_div(producer_theatrical, gross_ww)
        breakeven_gross = _safe_div(np.maximum(c_t - nt, 0.0), realization)
        cycle = df["cash_cycle_months"].to_numpy(float)

        return pd.DataFrame({
            "title_id": df["title_id"].to_numpy(),
            "segment": df["segment"].to_numpy(),
            "production_budget_cr": prod,
            "marketing_cr": mkt,
            "c_t_cr": c_t,
            "gross_ww_cr": gross_ww,
            "gst_leak_cr": gst_leak,
            "exhibitor_leak_cr": exhibitor_leak,
            "commission_leak_cr": commission_leak,
            "overseas_leak_cr": os_leak,
            "producer_theatrical_cr": producer_theatrical,
            "non_theatrical_cr": nt,
            "receipts_cr": receipts,
            "p_net_cr": p_net,
            "roi_multiple": p_net / c_t,
            "cei": receipts / c_t,
            "headline_roi": (gross_ww - prod) / prod,
            "realization_rate": realization,
            "breakeven_gross_cr": breakeven_gross,
            "breakeven_multiple": breakeven_gross / prod,
            "weekly_retention": hold,
            "decay_lambda": lam,
            "week1_share": _safe_div(wk[:, 0], india_gross),
            "burn_velocity_score": 100.0 * (mkt / c_t) * (1.0 - hold),
            "capital_velocity": receipts / c_t * 12.0 / cycle,
            "cash_cycle_months": cycle,
        })


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def bootstrap_pooled_roi(p_net: np.ndarray, c_t: np.ndarray, rng: np.random.Generator,
                         n_boot: int) -> np.ndarray:
    """Bootstrap distribution of pooled ROI (sum P_net / sum C_T), resampling titles."""
    n = len(p_net)
    out = np.empty(n_boot)
    chunk = max(1, int(4_000_000 // n))
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = p_net[idx].sum(axis=1) / c_t[idx].sum(axis=1)
    return out


def summarize_cohorts(m: pd.DataFrame, rng: np.random.Generator,
                      n_boot: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Cohort-level statistics plus bootstrap draws of pooled ROI per segment."""
    cols: dict[str, dict[str, float]] = {}
    boots: dict[str, np.ndarray] = {}
    for seg in SEGMENTS:
        s = m[m["segment"] == seg]
        if s.empty:
            raise SchemaError(f"Segment {seg} has no rows; both cohorts are required.")
        p, c, roi = s["p_net_cr"].to_numpy(), s["c_t_cr"].to_numpy(), s["roi_multiple"].to_numpy()
        k = max(1, int(np.ceil(0.10 * len(roi))))
        boot = bootstrap_pooled_roi(p, c, rng, n_boot)
        boots[seg] = boot
        cols[seg] = {
            "titles": float(len(s)),
            "capital_deployed_cr": c.sum(),
            "receipts_cr": s["receipts_cr"].sum(),
            "pooled_roi": p.sum() / c.sum(),
            "pooled_roi_ci_lo": np.percentile(boot, 2.5),
            "pooled_roi_ci_hi": np.percentile(boot, 97.5),
            "pooled_cei": s["receipts_cr"].sum() / c.sum(),
            "median_roi": np.median(roi),
            "p10_roi": np.percentile(roi, 10),
            "p90_roi": np.percentile(roi, 90),
            "cvar10_roi": np.sort(roi)[:k].mean(),
            "loss_rate": float((roi < 0).mean()),
            "median_headline_roi": s["headline_roi"].median(),
            "median_realization": s["realization_rate"].median(),
            "median_breakeven_multiple": s["breakeven_multiple"].median(),
            "median_cash_cycle_m": s["cash_cycle_months"].median(),
            "pooled_capital_velocity": (s["receipts_cr"].sum() / c.sum()) * 12.0
                                       / ((s["cash_cycle_months"] * s["c_t_cr"]).sum() / c.sum()),
            "median_capital_velocity": s["capital_velocity"].median(),
            "median_weekly_retention": s["weekly_retention"].median(),
            "median_week1_share": s["week1_share"].median(),
            "mean_bvs": s["burn_velocity_score"].mean(),
        }
    return pd.DataFrame(cols), boots


def capital_pool_monte_carlo(m: pd.DataFrame, segment: str, pool_cr: float, trials: int,
                             rng: np.random.Generator) -> pd.DataFrame:
    """
    Deploy an identical capital pool into each cohort by resampling titles.

    Titles are drawn with replacement until the next title would exceed
    ``pool_cr``. Returns per-trial pooled ROI and title count. Differences in
    dispersion therefore combine unit economics AND the diversification effect
    of title count; both are reported, neither is separated.
    """
    s = m[m["segment"] == segment]
    c, p = s["c_t_cr"].to_numpy(), s["p_net_cr"].to_numpy()
    k = int(np.ceil(pool_cr / c.min())) + 1
    chunk = max(25, int(2_000_000 // k))
    roi, count = [], []
    for start in range(0, trials, chunk):
        t = min(chunk, trials - start)
        idx = rng.integers(0, len(c), size=(t, k))
        keep = c[idx].cumsum(axis=1) <= pool_cr
        deployed = (c[idx] * keep).sum(axis=1)
        profit = (p[idx] * keep).sum(axis=1)
        roi.append(_safe_div(profit, deployed))
        count.append(keep.sum(axis=1))
    return pd.DataFrame({"pooled_roi": np.concatenate(roi), "titles": np.concatenate(count)})


def sensitivity_surface(engine: CapitalAuditEngine, base: SegmentParams, dud_probs: Sequence[float],
                        tail_scales: Sequence[float], n: int, seed: int) -> pd.DataFrame:
    """
    Micro cohort pooled ROI over (DUD probability x tail-median scale).

    Common random numbers (fixed seed per cell) limit cell-to-cell sampling noise.
    """
    grid = pd.DataFrame(index=list(dud_probs), columns=list(tail_scales), dtype=float)
    for d in dud_probs:
        for s in tail_scales:
            params = base.with_tail_stress(d, s)
            df = simulate_cohort(params, n, np.random.default_rng(seed), "SEN")
            met = engine.compute(df)
            grid.loc[d, s] = met["p_net_cr"].sum() / met["c_t_cr"].sum()
    return grid


# --------------------------------------------------------------------------- #
# Terminal report
# --------------------------------------------------------------------------- #
_ROWS: list[tuple[str, str, str]] = [
    ("titles", "Titles", "{:,.0f}"),
    ("capital_deployed_cr", "Capital deployed C_T (Rs Cr)", "{:,.1f}"),
    ("receipts_cr", "Producer-realized receipts (Rs Cr)", "{:,.1f}"),
    ("pooled_roi", "Pooled ROI  (sum P_net / sum C_T)", "{:+.3f}"),
    ("pooled_roi_ci_lo", "  95% bootstrap CI, lower", "{:+.3f}"),
    ("pooled_roi_ci_hi", "  95% bootstrap CI, upper", "{:+.3f}"),
    ("pooled_cei", "Pooled CEI (Rs realized per Rs 1 Cr)", "{:.3f}"),
    ("median_roi", "Median title ROI", "{:+.3f}"),
    ("p10_roi", "P10 title ROI", "{:+.3f}"),
    ("p90_roi", "P90 title ROI", "{:+.3f}"),
    ("cvar10_roi", "CVaR10 title ROI (worst decile mean)", "{:+.3f}"),
    ("loss_rate", "Title loss rate (P_net < 0)", "{:.1%}"),
    ("median_headline_roi", "Median HEADLINE ROI (gross, pre-leakage)", "{:+.3f}"),
    ("median_realization", "Median gross-to-producer realization", "{:.1%}"),
    ("median_breakeven_multiple", "Median breakeven gross / budget", "{:.2f}x"),
    ("median_cash_cycle_m", "Median cash cycle (months)", "{:.1f}"),
    ("pooled_capital_velocity", "Pooled capital velocity (turns/yr)", "{:.2f}"),
    ("median_capital_velocity", "Median title capital velocity (turns/yr)", "{:.2f}"),
    ("median_weekly_retention", "Median weekly demand retention h", "{:.3f}"),
    ("median_week1_share", "Median week-1 share of 8-wk India gross", "{:.1%}"),
    ("mean_bvs", "Mean Burn Velocity Score", "{:.2f}"),
]


def build_report(provenance: str, summary: pd.DataFrame, boots: dict[str, np.ndarray],
                 mc: dict[str, pd.DataFrame], pool_cr: float, sens: pd.DataFrame | None,
                 rng: np.random.Generator) -> str:
    """Assemble the plain-text audit report. Verdicts are computed, never hard-coded."""
    bar = "=" * 86
    lines = [bar, "CAPITAL AUDIT: LEGACY STUDIO SLATE vs MICRO-BUDGET INDEPENDENT SLATE", bar, provenance, ""]
    lines.append(f"{'Metric':<46}{'LEGACY':>18}{'MICRO':>18}")
    lines.append("-" * 82)
    for key, label, fmt in _ROWS:
        vals = [fmt.format(summary.loc[key, seg]) for seg in SEGMENTS]
        lines.append(f"{label:<46}{vals[0]:>18}{vals[1]:>18}")

    # Inference
    leg, mic = boots["LEGACY"], boots["MICRO"]
    n_pair = min(len(leg), len(mic))
    diff = mic[:n_pair] - leg[:n_pair]  # draws are iid and independent across cohorts
    d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
    lines += ["", "INFERENCE (bootstrap over titles, independent cohorts)", "-" * 82]
    lo, hi, pt = (summary.loc["pooled_roi_ci_lo", "LEGACY"], summary.loc["pooled_roi_ci_hi", "LEGACY"],
                  summary.loc["pooled_roi", "LEGACY"])
    if hi < 0:
        lines.append(f"Legacy pooled ROI {pt:+.3f}: 95% CI entirely below zero. Capital destruction not rejected by noise.")
    elif pt < 0:
        lines.append(f"Legacy pooled ROI {pt:+.3f}: negative point estimate; 95% CI [{lo:+.3f}, {hi:+.3f}] spans zero.")
    else:
        lines.append(f"Legacy pooled ROI {pt:+.3f}: non-negative. The capital-destruction hypothesis is NOT supported in this dataset.")
    lo, hi, pt = (summary.loc["pooled_roi_ci_lo", "MICRO"], summary.loc["pooled_roi_ci_hi", "MICRO"],
                  summary.loc["pooled_roi", "MICRO"])
    if lo > 0:
        lines.append(f"Micro pooled ROI {pt:+.3f}: 95% CI [{lo:+.3f}, {hi:+.3f}] strictly positive.")
    else:
        lines.append(f"Micro pooled ROI {pt:+.3f}: 95% CI [{lo:+.3f}, {hi:+.3f}] includes zero; positive return not established.")
    if d_lo > 0:
        lines.append(f"Micro - Legacy pooled ROI: 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]. Micro advantage distinguishable from zero.")
    elif d_hi < 0:
        lines.append(f"Micro - Legacy pooled ROI: 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]. Legacy outperforms micro.")
    else:
        lines.append(f"Micro - Legacy pooled ROI: 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]. Cohorts not separable at 95%.")

    # Equal-capital portfolios
    lines += ["", f"EQUAL-CAPITAL PORTFOLIO MONTE CARLO (pool = Rs {pool_cr:,.0f} Cr, titles resampled)", "-" * 82]
    lines.append(f"{'':<46}{'LEGACY':>18}{'MICRO':>18}")
    stats = {
        "Titles per portfolio (median)": lambda d: f"{d['titles'].median():.0f}",
        "Median portfolio ROI": lambda d: f"{d['pooled_roi'].median():+.3f}",
        "P(portfolio ROI < 0)": lambda d: f"{(d['pooled_roi'] < 0).mean():.1%}",
        "P(portfolio ROI < -25%)": lambda d: f"{(d['pooled_roi'] < -0.25).mean():.1%}",
        "5th percentile portfolio ROI": lambda d: f"{d['pooled_roi'].quantile(0.05):+.3f}",
    }
    for label, fn in stats.items():
        lines.append(f"{label:<46}{fn(mc['LEGACY']):>18}{fn(mc['MICRO']):>18}")

    if sens is not None:
        adv = sens - summary.loc["pooled_roi", "LEGACY"]
        share = float((adv.to_numpy() > 0).mean())
        lines += ["", "SENSITIVITY (micro DUD probability x tail-median scale)", "-" * 82,
                  f"Micro pooled ROI exceeds legacy in {share:.0%} of {adv.size} grid cells.",
                  f"Micro pooled ROI range across grid: {sens.min().min():+.3f} to {sens.max().max():+.3f}."]
        if share < 1.0:
            lines.append("Thesis-reversal region exists; see 06_sensitivity_frontier for its boundary.")
    lines.append(bar)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["panel"],
        "axes.edgecolor": PALETTE["grid"], "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"], "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "axes.axisbelow": True, "grid.color": PALETTE["grid"], "grid.linewidth": 0.6,
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
        "text.color": PALETTE["text"], "legend.facecolor": PALETTE["panel"],
        "legend.edgecolor": PALETTE["grid"], "font.family": "DejaVu Sans", "font.size": 9,
        "savefig.facecolor": PALETTE["bg"],
    })


def _save(fig: plt.Figure, stem: str, outdir: Path, formats: Sequence[str], dpi: int,
          footer: str) -> list[Path]:
    fig.text(0.01, 0.005, footer, fontsize=7, color=PALETTE["muted"], ha="left", va="bottom")
    paths = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_leakage_waterfall(m: pd.DataFrame, footer: str, outdir: Path, formats: Sequence[str], dpi: int):
    """Per Rs 100 of pooled headline gross: where cash is lost before and after invested capital."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, seg in zip(axes, SEGMENTS):
        s = m[m["segment"] == seg]
        scale = 100.0 / s["gross_ww_cr"].sum()
        steps = [
            ("Headline\ngross", 100.0, "total"),
            ("GST", -s["gst_leak_cr"].sum() * scale, "delta"),
            ("Exhibitor\nshare", -s["exhibitor_leak_cr"].sum() * scale, "delta"),
            ("Distributor\ncommission", -s["commission_leak_cr"].sum() * scale, "delta"),
            ("Overseas\nleakage", -s["overseas_leak_cr"].sum() * scale, "delta"),
            ("Rights /\ndigital", s["non_theatrical_cr"].sum() * scale, "delta"),
            ("Invested\nC_T", -s["c_t_cr"].sum() * scale, "delta"),
        ]
        run = 0.0
        for i, (label, val, kind) in enumerate(steps):
            if kind == "total":
                ax.bar(i, val, color=PALETTE["muted"], width=0.62)
                run = val
                ax.text(i, val + 1.5, f"{val:.1f}", ha="center", fontsize=8)
            else:
                bottom = run if val >= 0 else run + val
                color = PALETTE["MICRO"] if val >= 0 else PALETTE["LEGACY"]
                ax.bar(i, abs(val), bottom=bottom, color=color, width=0.62)
                ax.text(i, max(run, run + val) + 1.5, f"{val:+.1f}", ha="center", fontsize=8)
                run += val
        n = len(steps)
        net_color = PALETTE["MICRO"] if run >= 0 else PALETTE["LEGACY"]
        ax.bar(n, run, color=net_color, width=0.62, edgecolor=PALETTE["text"], linewidth=0.8)
        ax.text(n, max(run, 0) + 1.5, f"{run:+.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.axhline(0, color=PALETTE["text"], linewidth=0.8)
        ax.set_xticks(range(n + 1), [s_[0] for s_ in steps] + ["Net\nP_net"], fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.08)
        ax.set_title(SEGMENT_LABEL[seg])
    axes[0].set_ylabel("Rs per Rs 100 of pooled headline gross")
    fig.suptitle("Headline gross to net cash: pooled leakage waterfall", fontsize=14, fontweight="bold")
    return _save(fig, "01_leakage_waterfall", outdir, formats, dpi, footer)


def plot_roi_ecdf(m: pd.DataFrame, summary: pd.DataFrame, footer: str, outdir: Path,
                  formats: Sequence[str], dpi: int):
    """Empirical CDF of title-level ROI multiples, full cohort including non-releases and failures."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for seg in SEGMENTS:
        r = np.sort(m.loc[m["segment"] == seg, "roi_multiple"].to_numpy())
        y = np.arange(1, len(r) + 1) / len(r)
        ax.step(r, y, where="post", color=PALETTE[seg], linewidth=2,
                label=f"{SEGMENT_LABEL[seg]}  (loss rate {summary.loc['loss_rate', seg]:.0%}, "
                      f"pooled ROI {summary.loc['pooled_roi', seg]:+.2f})")
    ax.axvline(0, color=PALETTE["accent"], linestyle="--", linewidth=1.2)
    ax.text(0, 0.02, " breakeven", color=PALETTE["accent"], fontsize=8)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(left=-1.02)
    ax.set_xlabel("Title ROI multiple  (P_net / C_T), symlog axis")
    ax.set_ylabel("Cumulative share of titles")
    ax.set_title("Outcome distribution: complete cohorts, no survivor filtering")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, "02_roi_ecdf", outdir, formats, dpi, footer)


def plot_cei_scatter(m: pd.DataFrame, footer: str, outdir: Path, formats: Sequence[str], dpi: int):
    """Capital Efficiency Index against capital invested."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for seg in SEGMENTS:
        s = m[m["segment"] == seg]
        ax.scatter(s["c_t_cr"], np.clip(s["cei"], 1e-3, None), s=12, alpha=0.35,
                   color=PALETTE[seg], label=SEGMENT_LABEL[seg], linewidths=0)
        ax.scatter([s["c_t_cr"].median()], [s["cei"].median()], s=140, marker="D",
                   color=PALETTE[seg], edgecolor="white", linewidth=1.2, zorder=5)
    ax.axhline(1.0, color=PALETTE["accent"], linestyle="--", linewidth=1.2)
    ax.text(ax.get_xlim()[0], 1.04, " CEI = 1.0 (capital returned)", color=PALETTE["accent"], fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Invested capital C_T (Rs Cr, log)")
    ax.set_ylabel("CEI: realized yield per Rs 1 Cr invested (log)")
    ax.set_title("Capital Efficiency Index vs. capital at risk  (diamonds: cohort medians)")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    return _save(fig, "03_cei_scatter", outdir, formats, dpi, footer)


def plot_burn_velocity(m: pd.DataFrame, footer: str, outdir: Path, formats: Sequence[str], dpi: int,
                       weekly_cols: pd.DataFrame):
    """Demand decay curves (week 1 = 100) and Burn Velocity Score distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})
    weeks = np.arange(1, WEEKS + 1)
    for seg in SEGMENTS:
        idx = m.index[m["segment"] == seg]
        w = weekly_cols.loc[idx, GROSS_COLS].to_numpy()
        ok = w[:, 0] > 0
        norm = 100.0 * w[ok] / w[ok, 0][:, None]
        q25, q50, q75 = np.percentile(norm, [25, 50, 75], axis=0)
        ax1.fill_between(weeks, q25, q75, color=PALETTE[seg], alpha=0.18)
        ax1.plot(weeks, q50, color=PALETTE[seg], linewidth=2.2, marker="o", label=SEGMENT_LABEL[seg])
    ax1.set_xlabel("Release week")
    ax1.set_ylabel("India gross indexed to week 1 = 100")
    ax1.set_title("Demand decay (median, IQR band)")
    ax1.legend(fontsize=8)

    data = [m.loc[m["segment"] == seg, "burn_velocity_score"].dropna().to_numpy() for seg in SEGMENTS]
    bp = ax2.boxplot(data, tick_labels=["Legacy", "Micro"], patch_artist=True, showfliers=False, widths=0.5)
    for patch, seg in zip(bp["boxes"], SEGMENTS):
        patch.set(facecolor=PALETTE[seg], alpha=0.6, edgecolor=PALETTE["text"])
    for item in bp["medians"]:
        item.set(color="white", linewidth=1.8)
    for item in bp["whiskers"] + bp["caps"]:
        item.set(color=PALETTE["muted"])
    ax2.set_ylabel("BVS = 100 x (marketing / C_T) x (1 - h)")
    ax2.set_title("Burn Velocity Score")
    fig.suptitle("Marketing-driven demand decay vs. retention", fontsize=14, fontweight="bold")
    return _save(fig, "04_burn_velocity", outdir, formats, dpi, footer)


def plot_portfolio_mc(mc: dict[str, pd.DataFrame], pool_cr: float, footer: str, outdir: Path,
                      formats: Sequence[str], dpi: int):
    """Distribution of pooled ROI for identical capital pools."""
    fig, ax = plt.subplots(figsize=(10, 6))
    allv = np.concatenate([mc[s]["pooled_roi"].dropna().to_numpy() for s in SEGMENTS])
    bins = np.linspace(max(-1.0, np.percentile(allv, 0.2)), np.percentile(allv, 99.8), 70)
    for seg in SEGMENTS:
        d = mc[seg]["pooled_roi"].dropna()
        ax.hist(d, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=PALETTE[seg])
        ax.hist(d, bins=bins, density=True, histtype="step", linewidth=1.8, color=PALETTE[seg],
                label=f"{SEGMENT_LABEL[seg]}: {int(mc[seg]['titles'].median())} titles, "
                      f"P(ROI<0) = {(d < 0).mean():.0%}")
    ax.axvline(0, color=PALETTE["accent"], linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"Pooled ROI of a Rs {pool_cr:,.0f} Cr capital pool")
    ax.set_ylabel("Density")
    ax.set_title("Equal-capital portfolio outcomes (unit economics + diversification, not separated)")
    ax.legend(fontsize=8, loc="upper right")
    return _save(fig, "05_portfolio_monte_carlo", outdir, formats, dpi, footer)


def plot_sensitivity(grid: pd.DataFrame, legacy_roi: float, base_dud: float, base_scale: float,
                     footer: str, outdir: Path, formats: Sequence[str], dpi: int):
    """Heatmap of (micro pooled ROI - legacy pooled ROI) over tail stress parameters."""
    adv = grid.to_numpy(float) - legacy_roi
    lim = max(abs(np.nanmin(adv)), abs(np.nanmax(adv)), 1e-6)
    cmap = LinearSegmentedColormap.from_list("div", [PALETTE["LEGACY"], "#2b2f3a", PALETTE["MICRO"]])
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    im = ax.imshow(adv, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), aspect="auto", origin="lower")
    ax.grid(False)
    ax.set_xticks(range(grid.shape[1]), [f"{c:.2f}x" for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{r:.0%}" for r in grid.index])
    for i in range(adv.shape[0]):
        for j in range(adv.shape[1]):
            ax.text(j, i, f"{adv[i, j]:+.2f}", ha="center", va="center", fontsize=8, color="white")
    if base_dud in list(grid.index) and base_scale in list(grid.columns):
        bi, bj = list(grid.index).index(base_dud), list(grid.columns).index(base_scale)
        ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=PALETTE["accent"], linewidth=2.2))
    ax.set_xlabel("Scale applied to micro right-tail median multiples")
    ax.set_ylabel("Micro DUD-tier probability")
    ax.set_title("Micro minus legacy pooled ROI (outlined cell: default scenario)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Delta pooled ROI")
    return _save(fig, "06_sensitivity_frontier", outdir, formats, dpi, footer)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Capital allocation audit: legacy vs micro-budget slates.")
    ap.add_argument("--input", type=Path, help="CSV in the canonical schema (skips synthetic generation).")
    ap.add_argument("--n-legacy", type=int, default=400)
    ap.add_argument("--n-micro", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, default=Path("audit_output"))
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pool-cr", type=float, default=1000.0, help="Capital pool for the portfolio Monte Carlo.")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--mc-trials", type=int, default=5000)
    ap.add_argument("--skip-sensitivity", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    rng = np.random.default_rng(args.seed)
    try:
        if args.input is not None:
            if not args.input.exists():
                raise FileNotFoundError(f"Input file not found: {args.input}")
            raw = pd.read_csv(args.input)
            provenance = f"DATA PROVENANCE : USER-SUPPLIED DATASET ({args.input.name}), {len(raw):,} rows"
            footer = f"Source: {args.input.name}"
            run_sensitivity = False
            LOG.info("Sensitivity surface disabled for user-supplied data (it re-simulates the micro cohort).")
        else:
            raw = simulate_market(args.n_legacy, args.n_micro, rng)
            provenance = (f"DATA PROVENANCE : SYNTHETIC SCENARIO (seed={args.seed}). Outputs are conditional on the\n"
                          f"                  stated generator parameters. They are not observed market data.")
            footer = f"SYNTHETIC SCENARIO, seed {args.seed}. Conditional on stated assumptions; not observed data."
            run_sensitivity = not args.skip_sensitivity

        engine = CapitalAuditEngine()
        metrics = engine.compute(raw)
        validated = validate_schema(raw)
        summary, boots = summarize_cohorts(metrics, rng, args.bootstrap)
        mc = {seg: capital_pool_monte_carlo(metrics, seg, args.pool_cr, args.mc_trials, rng) for seg in SEGMENTS}

        sens = None
        if run_sensitivity:
            sens = sensitivity_surface(engine, DEFAULT_MICRO, [0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
                                       [0.25, 0.50, 0.75, 1.00, 1.50, 2.00], n=4000, seed=args.seed)

        args.outdir.mkdir(parents=True, exist_ok=True)
        report = build_report(provenance, summary, boots, mc, args.pool_cr, sens, rng)
        print(report)
        (args.outdir / "audit_report.txt").write_text(report + "\n", encoding="utf-8")
        metrics.to_csv(args.outdir / "title_level_metrics.csv", index=False)
        summary.to_csv(args.outdir / "cohort_summary.csv")

        _apply_theme()
        written: list[Path] = []
        written += plot_leakage_waterfall(metrics, footer, args.outdir, args.formats, args.dpi)
        written += plot_roi_ecdf(metrics, summary, footer, args.outdir, args.formats, args.dpi)
        written += plot_cei_scatter(metrics, footer, args.outdir, args.formats, args.dpi)
        written += plot_burn_velocity(metrics, footer, args.outdir, args.formats, args.dpi, validated)
        written += plot_portfolio_mc(mc, args.pool_cr, footer, args.outdir, args.formats, args.dpi)
        if sens is not None:
            written += plot_sensitivity(sens, float(summary.loc["pooled_roi", "LEGACY"]), 0.55, 1.0,
                                        footer, args.outdir, args.formats, args.dpi)
        LOG.info("Wrote %d chart files and 3 data/report files to %s", len(written), args.outdir.resolve())
        return 0
    except (SchemaError, FileNotFoundError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
