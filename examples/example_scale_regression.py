"""
Example: Scale-by-scale wavelet panel regression.

Reproduces the methodology from:
- Gallegati et al. (2015), Table 1 -- Productivity & Unemployment in G7
- Karlsson et al. (2020), Table 2 -- Oil prices & exchange rates

This example simulates a panel where the relationship between x and y
varies across time scales:
  - Short-run (D1, D2): near-zero or positive effect
  - Long-run (S3): strong negative effect
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from pywaveletpanel import (
    WaveletPanelOLS,
    modwt_mra,
    set_journal_style,
    plot_wavelet_decomposition,
)

# -- Apply publication theme ---------------------------------------------------
set_journal_style()

# -- Simulate panel data -------------------------------------------------------
np.random.seed(42)
N = 7       # number of entities (e.g. G7 countries)
T = 128     # time periods (annual data)

print("=" * 70)
print("  PyWaveletPanel -- Scale-by-Scale Panel Regression Example")
print("  Methodology: Gallegati et al. (2015) / Karlsson et al. (2020)")
print("=" * 70)

# Create x with multiple frequency components
t = np.arange(T, dtype=float)
y_all, x_all, ids_all, time_all = [], [], [], []

for i in range(N):
    # Regressor: mix of frequencies
    x_i = (
        0.5 * np.sin(2 * np.pi * t / 6)    # short-run (D1)
        + 1.0 * np.sin(2 * np.pi * t / 12)  # business cycle (D2)
        + 1.5 * np.sin(2 * np.pi * t / 30)  # medium-run (D3)
        + 0.02 * t                           # trend (S3)
        + np.random.randn(T) * 0.3
    )

    # Response: DIFFERENT relationships at different scales
    y_i = (
        0.05  * 0.5 * np.sin(2 * np.pi * t / 6)    # weak positive at D1
        + 0.2 * 1.0 * np.sin(2 * np.pi * t / 12)    # positive at D2
        + 0.4 * 1.5 * np.sin(2 * np.pi * t / 30)    # positive at D3
        - 0.9 * 0.02 * t                             # strong NEGATIVE at trend
        + (i * 0.5)                                   # fixed effect
        + np.random.randn(T) * 0.5
    )

    y_all.append(y_i)
    x_all.append(x_i)
    ids_all.append(np.full(T, i))
    time_all.append(t.copy())

y = np.concatenate(y_all)
X = np.concatenate(x_all)
entity_ids = np.concatenate(ids_all)
time_ids = np.concatenate(time_all)

# -- Fit the scale-by-scale panel regression -----------------------------------
print("\n>> Fitting WaveletPanelOLS with LA(8) wavelet, level=3 ...")
model = WaveletPanelOLS(wavelet="sym4", level=3, robust=True)
result = model.fit(
    y=y, X=X,
    entity_ids=entity_ids,
    time_ids=time_ids,
    regressor_names=["Productivity"],
)

# -- Display results -----------------------------------------------------------
print("\n" + result.summary())

# -- Export to LaTeX -----------------------------------------------------------
print("\n>> LaTeX output:")
print(result.to_latex())

# -- Tidy DataFrame ------------------------------------------------------------
print("\n>> Results as DataFrame:")
print(result.summary_df().to_string(index=False))

# -- Visualisations ------------------------------------------------------------
print("\n>> Generating plots ...")

# 1. Wavelet decomposition of first entity
mra = modwt_mra(x_all[0], wavelet="sym4", level=3)
fig1 = plot_wavelet_decomposition(
    x_all[0], mra,
    title="MODWT Decomposition -- Entity 1 (Productivity)",
)
fig1.savefig("wavelet_decomposition.png", dpi=150, bbox_inches="tight")

# 2. Scale-dependent coefficients
fig2 = result.plot()
fig2.savefig("scale_coefficients.png", dpi=150, bbox_inches="tight")

print("\n[OK] Saved: wavelet_decomposition.png")
print("[OK] Saved: scale_coefficients.png")
print("\n" + "=" * 70)
print("  Example complete.")
print("=" * 70)
