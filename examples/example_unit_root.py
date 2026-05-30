"""
Example: Wavelet-based panel unit root tests.

Compares four panel unit root tests from the literature:
1. IPS (ADF-based) -- Im, Pesaran & Shin (2003)
2. Wavelet Ratio IPS -- Li & Shukur (2013) [Paper 5]
3. WDWT -- Almasri et al. (2016) [Paper 3]
4. WMODWT -- Almasri et al. (2016) [Paper 3]

Demonstrates:
- Superior power of wavelet tests for near-integrated alternatives
- Comparison table and visualisation
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from pywaveletpanel import (
    WaveletRatioIPS,
    WaveletWaldDWT,
    WaveletWaldMODWT,
    PanelADF,
    set_journal_style,
    plot_unit_root_comparison,
)
from pywaveletpanel.tables import UnitRootTable

set_journal_style()

print("=" * 70)
print("  PyWaveletPanel -- Panel Unit Root Tests Comparison")
print("  Papers: Li & Shukur (2013), Almasri et al. (2016)")
print("=" * 70)

# -- Simulate near-integrated panel data ----------------------------------------
N = 5       # cross-sectional units
T = 128     # time periods
rho = 0.95  # near-integrated (stationary but close to unit root)
seed = 42

print(f"\n>> Simulating panel data: N={N}, T={T}, rho={rho}")
print(f"  (Near-integrated: should reject H0 of unit root)")

np.random.seed(seed)
data = np.zeros((N, T))
for i in range(N):
    for t in range(1, T):
        data[i, t] = rho * data[i, t - 1] + np.random.randn()

# -- Run all four tests ---------------------------------------------------------
n_mc = 5000  # Monte Carlo replications for critical values
print(f"\n>> Running tests with {n_mc} Monte Carlo replications ...")

print("  -> IPS (ADF) ...")
res_adf = PanelADF().test(data, n_mc=n_mc, seed=seed)

print("  -> Wavelet Ratio IPS ...")
res_wr = WaveletRatioIPS().test(data, n_mc=n_mc, seed=seed)

print("  -> WDWT ...")
res_wdwt = WaveletWaldDWT().test(data, n_mc=n_mc, seed=seed)

print("  -> WMODWT ...")
res_wmodwt = WaveletWaldMODWT().test(data, n_mc=n_mc, seed=seed)

# -- Display comparison table ---------------------------------------------------
results = [res_adf, res_wr, res_wdwt, res_wmodwt]
comparison = UnitRootTable.from_multiple_results(results)
print("\n" + comparison.render())

# -- Individual summaries -------------------------------------------------------
for r in results:
    print(f"\n{'--' * 25}")
    print(r.summary())

# -- LaTeX export ---------------------------------------------------------------
print("\n>> LaTeX output:")
print(comparison.to_latex())

# -- Visualisation --------------------------------------------------------------
print("\n>> Generating comparison plot ...")
fig = plot_unit_root_comparison(results)
fig.savefig("unit_root_comparison.png", dpi=150, bbox_inches="tight")
print("[OK] Saved: unit_root_comparison.png")

# -- Now test with actual unit root data ----------------------------------------
print("\n" + "=" * 70)
print("  Test 2: Pure random walk (should NOT reject H0)")
print("=" * 70)

data_rw = np.cumsum(np.random.randn(N, T), axis=1)
print(f"\n>> N={N}, T={T}, rho=1.0 (true unit root)")

res_adf_rw = PanelADF().test(data_rw, n_mc=n_mc, seed=seed + 1)
res_wr_rw = WaveletRatioIPS().test(data_rw, n_mc=n_mc, seed=seed + 1)
res_wmodwt_rw = WaveletWaldMODWT().test(data_rw, n_mc=n_mc, seed=seed + 1)

results_rw = [res_adf_rw, res_wr_rw, res_wmodwt_rw]
comparison_rw = UnitRootTable.from_multiple_results(
    results_rw, title="Unit Root Tests -- True Random Walk"
)
print("\n" + comparison_rw.render())

print("\n" + "=" * 70)
print("  Example complete.")
print("=" * 70)
