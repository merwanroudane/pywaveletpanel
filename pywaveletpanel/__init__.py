"""
PyWaveletPanel — Wavelet-Based Panel Data Econometrics
======================================================

A Python library for wavelet-based panel data analysis implementing methods
from the following papers:

1. Bada et al. (2021) — SAW estimator for structural breaks in panel models
2. Karlsson et al. (2020) — Wavelet-based panel analysis of oil–exchange rates
3. Almasri et al. (2016) — Wavelet panel unit-root tests with structural breaks
4. Gallegati et al. (2015) — Scale-by-scale panel regression for G7 countries
5. Li & Shukur (2013) — Wavelet ratio panel unit-root test

Author: Dr. Merwan Roudane (merwanroudane920@gmail.com)
Repository: https://github.com/merwanroudane/pywaveletpanel
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Dr. Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"
__url__ = "https://github.com/merwanroudane/pywaveletpanel"

# ---------------------------------------------------------------------------
# Public API — wavelet transforms
# ---------------------------------------------------------------------------
from pywaveletpanel.wavelets import (
    haar_dwt,
    haar_idwt,
    haar_modwt,
    haar_imodwt,
    la8_modwt,
    modwt,
    modwt_mra,
    pad_dyadic,
)

# ---------------------------------------------------------------------------
# Public API — panel regression (Papers 2, 4)
# ---------------------------------------------------------------------------
from pywaveletpanel.panel_regression import (
    WaveletPanelOLS,
    ScaleRegressionResult,
)

# ---------------------------------------------------------------------------
# Public API — structural breaks (Paper 1)
# ---------------------------------------------------------------------------
from pywaveletpanel.structural_breaks import (
    SAWEstimator,
    PostSAWEstimator,
    BreakDetectionResult,
)

# ---------------------------------------------------------------------------
# Public API — unit-root tests (Papers 3, 5)
# ---------------------------------------------------------------------------
from pywaveletpanel.unit_root import (
    WaveletRatioIPS,
    WaveletWaldDWT,
    WaveletWaldMODWT,
    PanelADF,
    UnitRootResult,
)

# ---------------------------------------------------------------------------
# Public API — tables & visualisation
# ---------------------------------------------------------------------------
from pywaveletpanel.tables import (
    RegressionTable,
    UnitRootTable,
    BreakTable,
    SimulationTable,
)
from pywaveletpanel.visualization import (
    plot_wavelet_decomposition,
    plot_scale_coefficients,
    plot_structural_breaks,
    plot_unit_root_comparison,
    plot_loess_by_country,
    set_journal_style,
)

__all__ = [
    # Wavelets
    "haar_dwt", "haar_idwt", "haar_modwt", "haar_imodwt",
    "la8_modwt", "modwt", "modwt_mra", "pad_dyadic",
    # Panel regression
    "WaveletPanelOLS", "ScaleRegressionResult",
    # Structural breaks
    "SAWEstimator", "PostSAWEstimator", "BreakDetectionResult",
    # Unit-root tests
    "WaveletRatioIPS", "WaveletWaldDWT", "WaveletWaldMODWT",
    "PanelADF", "UnitRootResult",
    # Tables
    "RegressionTable", "UnitRootTable", "BreakTable", "SimulationTable",
    # Visualisation
    "plot_wavelet_decomposition", "plot_scale_coefficients",
    "plot_structural_breaks", "plot_unit_root_comparison",
    "plot_loess_by_country", "set_journal_style",
]
