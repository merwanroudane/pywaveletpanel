<p align="center">
  <h1 align="center">🌊 PyWaveletPanel</h1>
  <p align="center">
    <strong>Wavelet-Based Panel Data Econometrics in Python</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
    <a href="https://github.com/merwanroudane/pywaveletpanel"><img src="https://img.shields.io/badge/GitHub-Repository-black.svg" alt="GitHub"></a>
  </p>
</p>

---

**PyWaveletPanel** is a Python library for wavelet-based panel data analysis. It implements econometric methods from five papers, providing tools for scale-by-scale panel regression, structural break detection, and panel unit root testing, together with journal-quality tables and publication-grade plots.

This document is a **complete usage guide with full syntax** for every public function and class.

**Links:** [GitHub Repository](https://github.com/merwanroudane/pywaveletpanel) · [PyPI](https://pypi.org/project/pywaveletpanel/) · [Issue Tracker](https://github.com/merwanroudane/pywaveletpanel/issues)

## 📚 Table of Contents

- [Implemented Papers](#-implemented-papers)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Data Conventions](#-data-conventions)
- [API Reference](#-api-reference)
  - [1. Wavelet Transforms (`wavelets`)](#1-wavelet-transforms-wavelets)
  - [2. Panel Regression (`panel_regression`)](#2-panel-regression-panel_regression)
  - [3. Structural Breaks (`structural_breaks`)](#3-structural-breaks-structural_breaks)
  - [4. Unit Root Tests (`unit_root`)](#4-unit-root-tests-unit_root)
  - [5. Tables (`tables`)](#5-tables-tables)
  - [6. Visualisation (`visualization`)](#6-visualisation-visualization)
  - [7. Utilities (`utils`)](#7-utilities-utils)
- [Scale Interpretation](#-scale-interpretation)
- [Examples](#-examples)
- [References](#-references)

## 📑 Implemented Papers

| # | Paper | Method | Module |
|---|-------|--------|--------|
| 1 | **Bada et al. (2021)** — *A Wavelet Method for Panel Models with Jump Discontinuities* | SAW Estimator, Post-SAW | `structural_breaks` |
| 2 | **Karlsson et al. (2020)** — *Unveiling Time-dependent Dynamics: Oil Prices & Exchange Rates* | MODWT Panel OLS | `panel_regression` |
| 3 | **Almasri et al. (2016)** — *Wavelet-based Panel Unit-root Test with Structural Breaks* | WDWT, WMODWT | `unit_root` |
| 4 | **Gallegati et al. (2015)** — *Productivity and Unemployment: Scale-by-scale Panel Analysis* | Scale-by-scale Panel FE | `panel_regression` |
| 5 | **Li & Shukur (2013)** — *Testing for Unit Roots in Panel Data Using Wavelet Ratio* | Wavelet Ratio IPS | `unit_root` |

## 🚀 Installation

```bash
git clone https://github.com/merwanroudane/pywaveletpanel.git
cd pywaveletpanel

# Install in development mode (recommended — makes `import pywaveletpanel` work anywhere)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

> **Note:** If you do not `pip install`, you must run scripts from the repo root or set `PYTHONPATH` to the repo directory, otherwise `import pywaveletpanel` fails with `ModuleNotFoundError`.

**Dependencies:** `numpy>=1.20`, `scipy>=1.7`, `pandas>=1.3`, `statsmodels>=0.13`, `matplotlib>=3.5`, `PyWavelets>=1.1`, `rich>=12.0`, `tabulate>=0.9`. Requires Python ≥ 3.9.

## ⚡ Quick Start

```python
import numpy as np
from pywaveletpanel import WaveletPanelOLS, set_journal_style

set_journal_style()

model = WaveletPanelOLS(wavelet='sym4', level=3, robust=True)
result = model.fit(y=y, X=X, entity_ids=entity_ids, time_ids=time_ids,
                   regressor_names=['Productivity'])

print(result.summary())     # journal-quality console table
result.plot()               # scale-dependent coefficient forest plot
print(result.to_latex())    # LaTeX export
df = result.summary_df()    # tidy DataFrame
```

## 📐 Data Conventions

Two distinct data layouts are used across the library:

| Layout | Used by | Shape | Description |
|--------|---------|-------|-------------|
| **Stacked panel** | `WaveletPanelOLS`, `SAWEstimator`, `PostSAWEstimator` | `(N*T,)` for `y`; `(N*T,)` or `(N*T, k)` for `X` | One row per (entity, time) observation. Paired with `entity_ids` (length `N*T`) and optional `time_ids`. |
| **Matrix panel** | All unit-root tests, `simulate_panel_ar1` | `(N, T)` | Each **row** is one entity's full time series. |

- **Balanced panels only:** every entity must have the same number of periods `T`. `WaveletPanelOLS.fit` raises `ValueError` on unbalanced data.
- If `time_ids` is omitted, observations are assumed already sorted in time order within each entity.
- `X.ndim == 1` is automatically reshaped to a single column `(N*T, 1)`.

---

# 📖 API Reference

## 1. Wavelet Transforms (`wavelets`)

Low-level transforms. All operate on a 1-D series `x` of shape `(T,)`.

### `haar_dwt(x, level=1) -> (V_J, W)`

Decimated Haar Discrete Wavelet Transform up to `level` J.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `ndarray (T,)` | — | Input series (ideally length divisible by `2**level`; odd lengths are boundary-reflected). |
| `level` | `int` | `1` | Decomposition level J. |

**Returns:** `V_J` — scaling (approximation) coefficients at level J; `W` — list `[W_1, …, W_J]` of detail coefficients (each halves in length per level).

### `haar_idwt(V_J, W) -> x`

Inverse Haar DWT. Reconstructs the signal from coarsest scaling coefficients `V_J` and detail list `W`.

### `haar_modwt(x, level=1) -> (V_J, W)`

Maximal-Overlap Haar DWT — translation-invariant, **no downsampling** (every level returns `T` coefficients). Uses rescaled, circularly-filtered Haar coefficients.

**Returns:** `V_J` of shape `(T,)`; `W` list of `(T,)` arrays.

### `haar_imodwt(V_J, W) -> x`

Inverse Haar MODWT.

### `modwt(x, wavelet="haar", level=1) -> (V_J, W)`

General MODWT using any PyWavelets filter (non-decimated / stationary transform).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `ndarray (T,)` | — | Input series. |
| `wavelet` | `str` | `"haar"` | Filter name from `pywt.wavelist()`. Use `"sym4"` for LA(8). |
| `level` | `int` | `1` | Decomposition level J. |

### `la8_modwt(x, level=4) -> (V_J, W)`

Convenience wrapper: `modwt(x, wavelet="sym4", level=level)` — the LA(8) filter from Gallegati et al. (2015) and Karlsson et al. (2020).

### `modwt_mra(x, wavelet="sym4", level=4) -> dict`

MODWT-based **Multiresolution Analysis**. Decomposes `x` into additive components such that `x ≈ D1 + D2 + … + DJ + SJ` (implemented via PyWavelets SWT with reflective padding).

**Returns** a `dict` with:
- Keys `'D1', 'D2', …, 'DJ'` → detail-component arrays (length `T`).
- Key `'SJ'` (e.g. `'S4'`) → smooth/trend component.
- Key `'labels'` → dict mapping each scale name to a frequency-band string (e.g. `'D1' → '2–4 periods'`).

```python
from pywaveletpanel import modwt_mra
comp = modwt_mra(x, wavelet='sym4', level=4)
print(comp['labels'])        # {'D1': '2–4 periods', ..., 'S4': '>32 periods (trend)'}
trend = comp['S4']
```

### `pad_dyadic(x, mode="reflect") -> x_padded`

Pads `x` to the next power-of-two length. `mode` is any NumPy pad mode (`"reflect"`, `"constant"`, `"edge"`). Returns `x` unchanged if already dyadic.

---

## 2. Panel Regression (`panel_regression`)

### `class WaveletPanelOLS`

Scale-by-scale wavelet panel regression with fixed effects (Papers 2, 4). Decomposes each variable via `modwt_mra`, then runs a fixed-effects OLS at each scale.

**Constructor**

```python
WaveletPanelOLS(
    wavelet="sym4",          # filter; 'sym4'=LA(8) (Papers 2,4), 'haar' (Papers 1,3)
    level=3,                 # decomposition level J
    robust=True,             # Newey-West HAC standard errors
    nw_lags=None,            # NW lag truncation; None = automatic rule-of-thumb
    include_aggregate=True,  # also estimate on raw (non-decomposed) data
)
```

**`.fit(y, X, entity_ids, time_ids=None, regressor_names=None) -> ScaleRegressionResult`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `ndarray (N*T,)` | Dependent variable (stacked). |
| `X` | `ndarray (N*T,)` or `(N*T, k)` | Regressors (stacked). |
| `entity_ids` | `ndarray (N*T,)` | Entity identifiers. |
| `time_ids` | `ndarray (N*T,)`, optional | Time identifiers (used to sort within entity). |
| `regressor_names` | `list[str]`, optional | Defaults to `['x1', 'x2', …]`. |

Raises `ValueError` if the panel is unbalanced.

### `@dataclass ScaleRegressionResult`

Returned by `WaveletPanelOLS.fit`.

**Attributes:** `scale_results` (`dict[str, dict]`), `aggregate_result` (`dict | None`), `scale_labels` (`dict[str, str]`), `n_entities`, `n_periods`, `wavelet`, `level`, `regressor_names`.

Each per-scale result `dict` contains: `coef`, `se`, `t_stat`, `pvalue` (arrays of length `k`), plus `r_squared`, `adj_r_squared`, `residuals`, `nobs`, `df`.

**Methods**

| Method | Returns | Description |
|--------|---------|-------------|
| `.summary(decimals=3)` | `str` | Rendered journal-quality table (columns: Aggregate, SJ, DJ…D1). |
| `.summary_df()` | `pd.DataFrame` | Tidy long-format results (one row per scale × regressor). |
| `.to_latex(decimals=3)` | `str` | LaTeX `table` environment. |
| `.plot(figsize=(10,6), **kwargs)` | `plt.Figure` | Forest plot of coefficients by scale. |

```python
from pywaveletpanel import WaveletPanelOLS

model = WaveletPanelOLS(wavelet='sym4', level=3, robust=True, nw_lags=None)
res = model.fit(y, X, entity_ids, time_ids, regressor_names=['Productivity'])

print(res.summary(decimals=3))
res.summary_df().to_csv('scale_results.csv', index=False)
fig = res.plot(figsize=(10, 7))
```

---

## 3. Structural Breaks (`structural_breaks`)

### `class SAWEstimator`

Structure-Adapted Wavelet estimator for detecting breaks in panel coefficients (Paper 1). First-differences out fixed effects, expands the cross-sectional coefficient estimates `γ̂_t` in a Haar basis, hard-thresholds the detail coefficients, and reads breaks off the reconstructed piecewise-constant path.

**Constructor**

```python
SAWEstimator(
    threshold_method="adaptive",  # see note below
    kappa_adjustment=True,        # log-log correction to kappa (eq. 3.1); affects the analytic threshold
    min_segment_length=2,         # minimum periods between consecutive breaks
)
```

**Threshold:** the noise in `γ̂_t` is estimated robustly from the finest detail level (median-absolute-deviation, Donoho & Johnstone 1994) and the universal threshold `σ_w·√(2 log T)` is applied. `threshold_method="universal"` additionally takes the max with the analytic threshold from Theorem 2. This MAD calibration is what keeps detection from over-segmenting when `γ̂_t`, being a cross-sectional average, has a much lower noise floor than the per-observation residual.

**`.detect(y, X, entity_ids, time_ids=None, regressor_names=None) -> BreakDetectionResult`**

Inputs use the **stacked-panel** layout (see [Data Conventions](#-data-conventions)).

### `class PostSAWEstimator`

Re-estimates the panel model on the stability intervals found by `SAWEstimator`, achieving the oracle property (Paper 1, Theorem 3).

**Constructor**

```python
PostSAWEstimator(
    variance_type="homoskedastic",  # 'homoskedastic' | 'cross_hetero' | 'time_hetero' | 'both'
    chow_test=True,                 # run Chow tests between consecutive intervals
)
```

**`.fit(y, X, entity_ids, time_ids=None, breaks=None) -> dict`**

`breaks` is a `BreakDetectionResult` from `SAWEstimator.detect`. Returns a `dict` with keys:
- `interval_results` — `{regressor_idx: [ {interval, coef, se, t_stat, pvalue, nobs}, … ]}`
- `chow_tests` — `{(p, seg_i, seg_j): {F_stat, pvalue, break_time}}`
- `full_coefficients` — `ndarray (T, k)` time-varying coefficient path
- `n_entities`, `n_periods`

### `@dataclass BreakDetectionResult`

**Attributes:** `n_breaks` (`dict[int,int]`), `break_locations` (`dict[int, list[int]]`), `stability_intervals` (`dict[int, list[(start,end)]]`), `coefficients` (`dict[int, list[float]]`), `threshold` (`float`), `wavelet_coeffs` (`dict[int, ndarray]`), `n_entities`, `n_periods`, `regressor_names`.

**Methods:** `.summary() -> str`, `.plot(figsize=(14,5), **kwargs) -> plt.Figure`, `.total_breaks() -> int`.

```python
from pywaveletpanel import SAWEstimator, PostSAWEstimator

saw = SAWEstimator(threshold_method='adaptive', min_segment_length=2)
breaks = saw.detect(y, X, entity_ids, time_ids, regressor_names=['AT_share'])

print(breaks.summary())
print("Total breaks:", breaks.total_breaks())
breaks.plot()

post = PostSAWEstimator(variance_type='both', chow_test=True)
final = post.fit(y, X, entity_ids, time_ids, breaks)
for (p, i, j), c in final['chow_tests'].items():
    print(f"reg {p}, {i}->{j}: F={c['F_stat']:.2f}, p={c['pvalue']:.4f}")
```

---

## 4. Unit Root Tests (`unit_root`)

All test classes share the signature `.test(data, n_mc=10000, seed=None) -> UnitRootResult`, where `data` is a **matrix panel** of shape `(N, T)` and critical values are obtained by `n_mc` Monte Carlo replications under H0 (independent random walks).

> **H0:** all entities have a unit root. **H1:** at least some entities are stationary.

| Class | Constructor | Statistic | Test direction | Reference |
|-------|-------------|-----------|----------------|-----------|
| `WaveletRatioIPS` | `WaveletRatioIPS()` | Mean Fan–Gençay wavelet ratio `S_NT` | left-tail (reject if `S_NT ≤ CV`) | Li & Shukur (2013) |
| `WaveletWaldDWT` | `WaveletWaldDWT()` | `W_DWT = T·tr[(H'H)⁻¹E'E] − N` | right-tail (reject if `stat ≥ CV`) | Almasri et al. (2016) |
| `WaveletWaldMODWT` | `WaveletWaldMODWT()` | MODWT analogue of `W_DWT` | right-tail | Almasri et al. (2016) |
| `PanelADF` | `PanelADF(trend="c")` | Mean ADF t-stat (IPS) | left-tail | Im, Pesaran & Shin (2003) |

`PanelADF` accepts `trend`: `"c"` (constant, default), `"ct"` (constant + trend), or any other value for no deterministic term.

**`.test` parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `ndarray (N, T)` | — | Panel matrix, one entity per row. |
| `n_mc` | `int` | `10000` | Monte Carlo replications for critical values. |
| `seed` | `int`, optional | `None` | RNG seed for reproducibility. |

### `@dataclass UnitRootResult`

**Attributes:** `test_name` (`str`), `statistic` (`float`), `pvalue` (`float`), `critical_values` (`{0.01, 0.05, 0.10 → float}`), `reject_null` (`{level → bool}`), `n_entities`, `n_periods`, `individual_stats` (`ndarray | None`, per-entity stats where applicable).

**Method:** `.summary() -> str`.

```python
import numpy as np
from pywaveletpanel import (
    WaveletRatioIPS, WaveletWaldDWT, WaveletWaldMODWT, PanelADF,
    plot_unit_root_comparison,
)
from pywaveletpanel.tables import UnitRootTable

data = np.random.randn(5, 128)   # (N, T)

res_adf    = PanelADF(trend='c').test(data, n_mc=5000, seed=0)
res_wr     = WaveletRatioIPS().test(data, n_mc=5000, seed=0)
res_wdwt   = WaveletWaldDWT().test(data, n_mc=5000, seed=0)
res_wmodwt = WaveletWaldMODWT().test(data, n_mc=5000, seed=0)

print(UnitRootTable.from_multiple_results(
    [res_adf, res_wr, res_wdwt, res_wmodwt]).render())
plot_unit_root_comparison([res_adf, res_wr, res_wdwt, res_wmodwt])
```

---

## 5. Tables (`tables`)

Four table builders. Each renders to console (via `rich`, falling back to `tabulate`) and exports to LaTeX/HTML/DataFrame. Console rendering auto-highlights significance stars and reject/accept decisions.

### `class RegressionTable`

| Member | Signature | Description |
|--------|-----------|-------------|
| classmethod | `RegressionTable.from_scale_result(result, decimals=3)` | Build from a `ScaleRegressionResult`. |
| method | `.render() -> str` | Console table. |
| method | `.to_latex() -> str` | LaTeX. |
| method | `.to_html() -> str` | Bootstrap-styled HTML. |
| method | `.to_dataframe() -> pd.DataFrame` | Underlying frame. |

### `class UnitRootTable`

| Member | Signature | Description |
|--------|-----------|-------------|
| classmethod | `UnitRootTable.from_single_result(result)` | Single `UnitRootResult`. |
| classmethod | `UnitRootTable.from_multiple_results(results, title="")` | Side-by-side comparison from a list of results. |
| method | `.render() -> str` / `.to_latex() -> str` | Output. |

### `class BreakTable`

| Member | Signature | Description |
|--------|-----------|-------------|
| classmethod | `BreakTable.from_break_result(result)` | From a `BreakDetectionResult`. |
| method | `.render() -> str` / `.to_latex() -> str` | Output. |

### `class SimulationTable`

| Member | Signature | Description |
|--------|-----------|-------------|
| classmethod | `SimulationTable.from_simulation(results, title="Monte Carlo Simulation Results")` | `results = {test_name: {scenario: rejection_rate}}`. |
| method | `.render() -> str` / `.to_latex() -> str` | Output. |

```python
from pywaveletpanel.tables import SimulationTable
sim = SimulationTable.from_simulation({
    "WDWT":   {"rho=1.00": 0.051, "rho=0.95": 0.62},
    "WMODWT": {"rho=1.00": 0.049, "rho=0.95": 0.71},
})
print(sim.render())
```

---

## 6. Visualisation (`visualization`)

All plotting functions return a `matplotlib.figure.Figure` and accept an optional `save_path` to write a 300-dpi image.

### `set_journal_style()`

Applies the light journal/paper publication theme globally to matplotlib (white background, serif fonts, subtle grey grid, Okabe-Ito colorblind-safe palette). Call once at the top of a script.

### `plot_wavelet_decomposition(x, components, title="MODWT Multiresolution Decomposition", time_index=None, figsize=(14,10), save_path=None)`

Stacked panels of the original series and each MRA component. `components` is the dict returned by `modwt_mra`.

### `plot_scale_coefficients(result, figsize=(10,7), ci_level=0.05, save_path=None, **kwargs)`

Forest plot of coefficients per scale with confidence intervals; significant points highlighted. `result` is a `ScaleRegressionResult`. (Also reachable via `result.plot()`.)

### `plot_structural_breaks(result, figsize=(14,5), time_index=None, save_path=None, **kwargs)`

Step-function coefficient paths with vertical break lines. `result` is a `BreakDetectionResult`. (Also reachable via `result.plot()`.)

### `plot_unit_root_comparison(results, figsize=(12,6), save_path=None)`

Two-panel grouped bar chart (p-values and 5% decisions) across a list of `UnitRootResult`.

### `plot_loess_by_country(x_dict, y_dict, xlabel="Productivity growth", ylabel="Unemployment rate", title="Nonparametric Loess Fit by Country", span=0.5, figsize=(16,10), save_path=None)`

Per-country scatter with a smoothed fit. `x_dict`/`y_dict` map `country_name → array`. `span` (0–1) controls smoothing window.

```python
from pywaveletpanel import (
    set_journal_style, modwt_mra,
    plot_wavelet_decomposition, plot_loess_by_country,
)
set_journal_style()
comp = modwt_mra(series, wavelet='sym4', level=4)
plot_wavelet_decomposition(series, comp, title="Oil Price Decomposition",
                           save_path="decomp.png")
```

---

## 7. Utilities (`utils`)

Lower-level helpers (importable from `pywaveletpanel.utils`).

| Function | Signature | Description |
|----------|-----------|-------------|
| `newey_west_se` | `newey_west_se(X, residuals, n_lags=None) -> ndarray` | HAC standard errors (Bartlett kernel). `n_lags=None` → `floor(4·(T/100)^(2/9))`. |
| `fixed_effects_transform` | `fixed_effects_transform(y, entity_ids) -> ndarray` | Within (entity-demeaning) transform. |
| `first_difference` | `first_difference(y, entity_ids, time_ids=None) -> (dy, mask)` | First-difference within each entity. |
| `ols_fit` | `ols_fit(y, X, robust=True, n_lags=None) -> dict` | OLS with optional NW SEs; returns `coef, se, t_stat, pvalue, r_squared, adj_r_squared, residuals, nobs`. |
| `panel_fixed_effects_ols` | `panel_fixed_effects_ols(y, X, entity_ids, robust=True, n_lags=None) -> dict` | Core FE panel estimator (adds `df`). |
| `significance_stars` | `significance_stars(pvalue) -> str` | `***`/`**`/`*`/`""`. |
| `format_coef` | `format_coef(value, pvalue, decimals=4) -> str` | Coefficient + stars. |
| `simulate_panel_ar1` | `simulate_panel_ar1(N, T, rho=1.0, cross_corr=0.0, seed=None) -> ndarray (N,T)` | AR(1) panel; `rho=1` → unit root, `cross_corr` adds equi-correlation. |

```python
from pywaveletpanel.utils import simulate_panel_ar1
# Near-integrated panel with cross-sectional dependence
data = simulate_panel_ar1(N=10, T=200, rho=0.95, cross_corr=0.3, seed=42)
```

---

## 📊 Scale Interpretation

**Annual data (J=3):**

| Scale | Period | Interpretation |
|-------|--------|----------------|
| D1 | 2–4 years | Short-run / business cycle |
| D2 | 4–8 years | Business cycle |
| D3 | 8–16 years | Medium-run |
| S3 | >16 years | Long-run trend |

**Monthly data (J=4):**

| Scale | Period | Interpretation |
|-------|--------|----------------|
| D1 | 1–2 months | Very short-run |
| D2 | 2–4 months | Short-run |
| D3 | 4–8 months | Medium-run |
| D4 | 8–16 months | Long-run |

### Unit Root Test Comparison

| Test | Robust to cross-dep? | Robust to breaks? | Reference |
|------|---------------------|-------------------|-----------|
| IPS (ADF) | ✗ | ✗ | Im, Pesaran & Shin (2003) |
| Wavelet Ratio IPS | Partial | ✗ | Li & Shukur (2013) |
| **WDWT** | **✓** | **✓** | Almasri et al. (2016) |
| **WMODWT** | **✓** | **✓** | Almasri et al. (2016) |

## 📁 Examples

Run from the repo root (or after `pip install -e .`):

```bash
python examples/example_scale_regression.py   # Papers 2, 4
python examples/example_structural_breaks.py   # Paper 1
python examples/example_unit_root.py           # Papers 3, 5
```

## 🏗️ Library Architecture

```
pywaveletpanel/
├── wavelets.py             # Haar DWT/MODWT, LA(8), MODWT-MRA, dyadic padding
├── panel_regression.py     # WaveletPanelOLS, ScaleRegressionResult
├── structural_breaks.py    # SAWEstimator, PostSAWEstimator, BreakDetectionResult
├── unit_root.py            # WaveletRatioIPS, WaveletWaldDWT/MODWT, PanelADF, UnitRootResult
├── tables.py               # RegressionTable, UnitRootTable, BreakTable, SimulationTable
├── visualization.py        # set_journal_style + 5 plot functions
└── utils.py                # Newey-West HAC, panel transforms, OLS, MC simulation
```

## 📚 References

1. Bada, O., Kneip, A., Liebl, D., Mensinger, T., Gualtieri, J. & Sickles, R.C. (2021). A Wavelet Method for Panel Models with Jump Discontinuities in the Parameters. *arXiv:2109.10950v1*.
2. Karlsson, H.K., Månsson, K. & Sjölander, P. (2020). Unveiling the Time-dependent Dynamics between Oil Prices and Exchange Rates: A Wavelet-based Panel Analysis. *The Energy Journal*, 41(1), 87–106.
3. Almasri, A., Månsson, K., Sjölander, P. & Shukur, G. (2016). A wavelet-based panel unit-root test in the presence of an unknown structural break and cross-sectional dependency. *Applied Economics*, DOI:10.1080/00036846.2016.1231908.
4. Gallegati, M., Gallegati, M., Ramsey, J.B. & Semmler, W. (2015). Productivity and unemployment: a scale-by-scale panel data analysis for the G7 countries. *Studies in Nonlinear Dynamics & Econometrics*, DOI:10.1515/snde-2014-0053.
5. Li, Y. & Shukur, G. (2013). Testing for Unit Roots in Panel Data Using a Wavelet Ratio Method. *Computational Economics*, 41, 59–69.

## 👤 Author

**Dr. Merwan Roudane** — 📧 merwanroudane920@gmail.com — 🔗 [github.com/merwanroudane](https://github.com/merwanroudane)

## 📄 License

MIT License — see [LICENSE](LICENSE).

```
@software{roudane2024pywaveletpanel,
  author = {Roudane, Merwan},
  title  = {PyWaveletPanel: Wavelet-Based Panel Data Econometrics in Python},
  year   = {2024},
  url    = {https://github.com/merwanroudane/pywaveletpanel}
}
```
