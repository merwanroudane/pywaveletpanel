"""
Example: SAW structural break detection in panel data.

Reproduces the methodology from:
  Bada et al. (2021), "A Wavelet Method for Panel Models with Jump
  Discontinuities in the Parameters", arXiv:2109.10950v1.

Simulates a panel where the slope coefficient gamma_t changes at two
known break points, then detects them using the SAW estimator.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from pywaveletpanel import (
    SAWEstimator,
    PostSAWEstimator,
    set_journal_style,
)

set_journal_style()

# -- Simulate DGP similar to Paper 1, DGP1 ------------------------------------
np.random.seed(123)
N = 50      # cross-sectional units
T = 64      # time periods

print("=" * 70)
print("  PyWaveletPanel -- SAW Structural Break Detection Example")
print("  Methodology: Bada et al. (2021), arXiv:2109.10950v1")
print("=" * 70)

# True coefficient path with 2 breaks
# gamma_t = 1.0  for t in [0, 20)
# gamma_t = -0.5 for t in [20, 45)
# gamma_t = 2.0  for t in [45, 64)
gamma_true = np.ones(T)
gamma_true[20:45] = -0.5
gamma_true[45:] = 2.0

print(f"\n>> True coefficient path:")
print(f"  gamma = 1.0  for t in [0, 20)")
print(f"  gamma = -0.5 for t in [20, 45)")
print(f"  gamma = 2.0  for t in [45, 64)")
print(f"  True breaks at t = 20, 45")

# Generate panel data
alpha_i = np.random.randn(N) * 2  # fixed effects
y_all, x_all, ids_all, time_all = [], [], [], []

for i in range(N):
    x_i = np.random.randn(T)
    eps_i = np.random.randn(T) * 0.5
    y_i = alpha_i[i] + gamma_true * x_i + eps_i
    y_all.append(y_i)
    x_all.append(x_i)
    ids_all.append(np.full(T, i))
    time_all.append(np.arange(T))

y = np.concatenate(y_all)
X = np.concatenate(x_all)
entity_ids = np.concatenate(ids_all)
time_ids = np.concatenate(time_all)

# -- Detect breaks -------------------------------------------------------------
print(f"\n>> Running SAW estimator (N={N}, T={T}) ...")
saw = SAWEstimator(threshold_method="adaptive", min_segment_length=5)
breaks = saw.detect(
    y, X, entity_ids, time_ids,
    regressor_names=["x1"],
)

# -- Display results -----------------------------------------------------------
print("\n" + breaks.summary())
print(f"\n>> Detected {breaks.total_breaks()} break(s):")
for p, locs in breaks.break_locations.items():
    name = breaks.regressor_names[p]
    print(f"  {name}: breaks at t = {locs}")
    for (s, e), c in zip(breaks.stability_intervals[p], breaks.coefficients[p]):
        print(f"    [{s:3d}, {e:3d}): gamma_hat = {c:.4f}")

# -- Post-SAW estimation ------------------------------------------------------
print("\n>> Running Post-SAW estimation ...")
post = PostSAWEstimator(variance_type="homoskedastic", chow_test=True)
post_result = post.fit(y, X, entity_ids, time_ids, breaks)

if post_result["chow_tests"]:
    print("\n>> Chow tests between consecutive intervals:")
    for key, chow in post_result["chow_tests"].items():
        p_idx, seg_from, seg_to = key
        print(f"  Regressor {p_idx}, interval {seg_from}->{seg_to}: "
              f"F = {chow['F_stat']:.2f}, p = {chow['pvalue']:.4f}")

# -- Visualise -----------------------------------------------------------------
print("\n>> Generating structural break plot ...")
fig = breaks.plot(time_index=np.arange(T))
fig.savefig("structural_breaks.png", dpi=150, bbox_inches="tight")
print("[OK] Saved: structural_breaks.png")

print("\n" + "=" * 70)
print("  Example complete.")
print("=" * 70)
