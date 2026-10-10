# Capital Audit: Legacy Studio Slate vs. Micro-Budget Independent Slate

A reproducible audit of capital allocation in Indian film production. The toolkit converts title-level production, marketing, and box-office data into **producer-realized cash flows**, then measures return, capital efficiency, recovery velocity, and demand decay for two cohorts: `LEGACY` (studio-scale slate) and `MICRO` (micro-budget independent slate).

Headline gross is a marketing statistic. Producer cash is what remains after tax, exhibitor share, intermediary commission, overseas leakage, and the capital that financed the release. This toolkit measures the second quantity.

---

## 1. Data provenance (read first)

| Mode | Command | Status of output |
|---|---|---|
| Synthetic scenario | `python capital_audit.py` | **Conditional.** Results follow from generator parameters in `SegmentParams`. They are not observations. |
| Empirical | `python capital_audit.py --input titles.csv` | Observational, subject to dataset quality and selection. |

The default synthetic scenario encodes the structural premises under test (high fixed marketing intensity and front-loaded demand for the legacy slate; lower fixed cost, higher retention, performance-linked digital yield, and a heavy right tail for the micro slate). A model that assumes a mechanism cannot independently demonstrate it. Synthetic runs are stamped in the terminal report and on every chart footer. Any publication of results should state the mode used. Claims about the real market require real data run through the same engine.

Generator parameters are calibration placeholders and have not been audited against a ledger of actual budgets, P&A, or producer receipts.

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python capital_audit.py
```

Python 3.10+. Dependencies: `numpy`, `pandas`, `matplotlib`.

### CLI

| Flag | Default | Meaning |
|---|---|---|
| `--input PATH` | none | Canonical-schema CSV. Disables synthetic generation and the sensitivity surface. |
| `--n-legacy` / `--n-micro` | 400 / 3000 | Synthetic cohort sizes. |
| `--seed` | 42 | RNG seed. Identical seeds reproduce identical output. |
| `--outdir` | `audit_output` | Output directory. |
| `--formats` | `png pdf` | Chart formats (`png`, `pdf`, `svg`). |
| `--dpi` | 300 | Raster export resolution. |
| `--pool-cr` | 1000 | Capital pool (Rs Cr) for the equal-capital Monte Carlo. |
| `--bootstrap` | 2000 | Bootstrap resamples. |
| `--mc-trials` | 5000 | Portfolio Monte Carlo trials. |
| `--skip-sensitivity` | off | Skip the sensitivity surface. |

### Outputs

| File | Content |
|---|---|
| `audit_report.txt` | Terminal report, saved. |
| `title_level_metrics.csv` | Every metric for every title. |
| `cohort_summary.csv` | Cohort statistics. |
| `01_leakage_waterfall` | Pooled headline gross to net cash, per Rs 100 of gross. |
| `02_roi_ecdf` | Full-cohort ROI distribution (no survivor filtering). |
| `03_cei_scatter` | CEI vs. capital at risk. |
| `04_burn_velocity` | Demand decay curves and Burn Velocity Score. |
| `05_portfolio_monte_carlo` | Equal-capital portfolio outcomes. |
| `06_sensitivity_frontier` | Micro-minus-legacy ROI over tail-stress parameters. |

---

## 3. Input schema

One row per title. All monetary values in INR crore. Both segments are required.

| Column | Type | Definition |
|---|---|---|
| `title_id` | str | Unique identifier. |
| `segment` | str | `LEGACY` or `MICRO`. |
| `production_budget_cr` | float > 0 | Negative cost of production. |
| `marketing_cr` | float >= 0 | Publicity and advertising. |
| `distribution_overhead_cr` | float >= 0 | Delivery/print charges, release-slot costs, minimum-guarantee shortfalls, allocated overhead. |
| `cash_cycle_months` | float > 0 | Months from first capital outlay to final receipt. |
| `wk_gross_1` ... `wk_gross_8` | float >= 0 | Weekly India gross collections. |
| `overseas_gross_cr` | float >= 0 | Overseas gross. |
| `non_theatrical_cr` | float >= 0 | Realized streaming, satellite, music, and direct-digital yield, net of platform splits. |

Validation rejects missing columns, null or non-numeric values, negatives, unknown segments, duplicate IDs, and non-positive budgets or cycle times.

---

## 4. Formula specification

Let *G* denote weekly India gross, *O* overseas gross, *NT* non-theatrical yield, and *w* index release weeks 1..8.

**Release waterfall** (`ReleaseEconomics`)

$$N_w = G_w\,(1-\tau) \qquad D = \sum_w N_w\,s_w \qquad R_{\text{India}} = D\,(1-\kappa)$$

$$R_{\text{theatrical}} = R_{\text{India}} + \rho\,O \qquad R = R_{\text{theatrical}} + NT$$

where τ is the blended GST wedge, *s_w* the producer-side share of nett in week *w*, κ the distributor commission, and ρ overseas realization. Defaults (`gst 16%`, `s = 50 / 42.5 / 37.5 / 30%...`, `κ = 10%`, `ρ = 40%`) are scenario parameters; contract-level terms should replace them.

**Total Invested Capital**

$$C_T = \text{Production} + \text{Marketing} + \text{Distribution Overhead}$$

**Net Realized Profit and ROI Multiple**

$$P_{\text{net}} = R - C_T \qquad \text{ROI} = \frac{P_{\text{net}}}{C_T}$$

**Capital Efficiency Index** (producer-realized yield per Rs 1 Cr invested)

$$\text{CEI} = \frac{R}{C_T} \qquad \text{CEI} = 1 + \text{ROI}$$

**Gross-to-producer realization and breakeven**

$$\phi = \frac{R_{\text{theatrical}}}{G_{\text{WW}}} \qquad G^{*}_{\text{WW}} = \frac{\max(C_T - NT,\,0)}{\phi}$$

`breakeven_multiple` = *G\** / production budget. `headline_roi` = (*G_WW* − production) / production is reported only as the naive comparator; it omits marketing, overhead, and every leakage layer.

**Demand decay and Burn Velocity Score**

Per-title OLS on `ln(G_w) = a - λ·w` gives decay rate λ and weekly retention *h = e^(−λ)*.

$$\text{BVS} = 100 \cdot \frac{M}{C_T} \cdot (1-h)$$

BVS is a composite heuristic, not an accounting standard: the share of invested capital that is marketing, multiplied by the fraction of demand lost per week. Higher values indicate more capital committed to demand that decays faster.

**Capital velocity**

$$V = \text{CEI}\cdot\frac{12}{\text{cash cycle (months)}}$$

Cohort-level velocity uses pooled CEI and the capital-weighted mean cycle.

**Cohort statistics.** Pooled ROI = Σ*P_net* / Σ*C_T* (capital-weighted; this is the portfolio-relevant figure, not the mean of title ROIs). Confidence intervals are percentile bootstraps over titles. CVaR10 is the mean ROI of the worst decile of titles. The equal-capital Monte Carlo resamples titles with replacement until a fixed capital pool is exhausted.

---

## 5. Reference run (synthetic scenario, seed 42, defaults)

Conditional on the default generator parameters. Not observed data.

| Metric | LEGACY | MICRO |
|---|---|---|
| Pooled ROI | -0.207 (95% CI -0.253 to -0.158) | +0.216 (95% CI +0.044 to +0.415) |
| Pooled CEI | 0.79 | 1.22 |
| Title loss rate | 78.0% | 79.4% |
| Median title ROI | -0.29 | -0.74 |
| CVaR10 title ROI | -0.75 | -0.97 |
| Median gross-to-producer realization | 35.3% | 32.2% |
| Median breakeven gross / budget | 2.25x | 3.23x |
| Pooled capital velocity (turns/yr) | 0.35 | 1.21 |
| Median weekly retention *h* | 0.44 | 0.60 |
| Mean Burn Velocity Score | 9.4 | 3.5 |
| P(Rs 1,000 Cr portfolio ROI < 0) | 83.6% | 21.2% |

Observations the output supports under this scenario:

1. Legacy cohort return is negative with a confidence interval excluding zero. The mechanism is the leakage stack (waterfall chart): headline gross converts to producer cash at roughly one-third, while marketing and overhead raise C_T to ~1.3x production.
2. Headline ROI misstates outcomes in both directions. The median legacy title shows positive headline ROI (+0.11) and negative realized ROI (-0.29).
3. The micro cohort's title-level loss rate is **not lower** than legacy (79.4% vs. 78.0%), and its median title ROI is worse. Its pooled return is positive because of the right tail. The return profile is tail-dependent by construction.
4. The micro-minus-legacy advantage reverses in part of the sensitivity grid (see chart 06). With the DUD-tier probability at 85% and tail multiples at 0.25x the default, the micro cohort underperforms legacy. The thesis holds in 64% of the grid cells.
5. The sensitivity surface is a separate draw (n = 4000, fixed seed) from the headline run, so its default cell differs from the headline pooled-ROI gap by sampling noise inherent to a heavy right tail.

---

## 6. Threats to validity

- **Survivorship.** Micro-budget datasets assembled from released or celebrated titles overstate returns. The tool is designed for complete cohorts, including titles that never obtained a theatrical release. Input data should be audited for this before any ROI claim.
- **Tail dependence.** Micro-cohort pooled ROI is dominated by a small fraction of titles. The bootstrap quantifies resampling noise but cannot quantify the probability that the tail itself is mis-specified.
- **Diversification confound.** The equal-capital Monte Carlo compares ~4 titles against ~370. Lower portfolio dispersion for micro reflects title count as well as unit economics; the two are not separated.
- **Parameter risk.** Share schedules, GST wedge, commission, overseas realization, and non-theatrical elasticity are scenario inputs. Results move with them.
- **Capital definition.** C_T excludes financing cost. Cash-cycle length enters through capital velocity only.
- **Non-theatrical correlation.** The generator links digital yield to theatrical outcome through a power-law elasticity (0.15 legacy, 0.70 micro). Real pre-sale and licensing contracts may differ materially.
- **Burn Velocity Score.** A heuristic composite. It is monotone in marketing intensity and weekly demand loss and should be reported with its definition.

---

## 7. Extending

- Replace `DEFAULT_LEGACY` / `DEFAULT_MICRO` with calibrated `SegmentParams`, or bypass the generator and ingest deal-level data.
- Override `ReleaseEconomics` with contract-level revenue-share schedules.
- `CapitalAuditEngine.compute` is stateless and accepts any frame passing `validate_schema`.

## 8. Licence

Add a licence file appropriate to the repository (MIT or Apache-2.0 are conventional for analytical tooling).
