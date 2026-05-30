"""
Wavelet-based panel unit root tests.

Implements:
- Wavelet Ratio IPS test (Li & Shukur, 2013) [Paper 5]
- Wavelet Wald DWT test — WDWT (Almasri et al., 2016) [Paper 3]
- Wavelet Wald MODWT test — WMODWT (Almasri et al., 2016) [Paper 3]
- Wavelet IPS-DWT test — IPSDWT (Li & Shukur, 2013; Almasri et al., 2016)
- Standard IPS / ADF panel test for comparison

All tests use Monte Carlo simulated critical values.

References
----------
Fan, Y. & Gençay, R. (2010). Unit root tests with wavelets.
    Econometric Theory, 26, 1305–1331.
Li, Y. & Shukur, G. (2013). Testing for unit roots in panel data
    using a wavelet ratio method. Computational Economics, 41, 59–69.
Almasri et al. (2016). A wavelet-based panel unit-root test in the
    presence of unknown structural breaks and cross-sectional dependency.
    Applied Economics.

Example
-------
>>> from pywaveletpanel import WaveletRatioIPS, WaveletWaldMODWT, PanelADF
>>> data = np.random.randn(5, 200)  # 5 entities, 200 periods
>>> # Wavelet ratio IPS test
>>> wr = WaveletRatioIPS()
>>> res_wr = wr.test(data, n_mc=5000)
>>> # WMODWT test
>>> wm = WaveletWaldMODWT()
>>> res_wm = wm.test(data, n_mc=5000)
>>> # Standard ADF-based IPS
>>> adf = PanelADF()
>>> res_adf = adf.test(data)
>>> # Compare
>>> print(res_wr.summary())
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from scipy import stats

from pywaveletpanel.wavelets import haar_dwt, haar_modwt
from pywaveletpanel.utils import simulate_panel_ar1


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnitRootResult:
    """
    Result of a panel unit root test.

    Attributes
    ----------
    test_name : str
    statistic : float
    pvalue : float
    critical_values : dict
        ``{significance_level: critical_value}``, e.g. ``{0.01: ..., 0.05: ..., 0.10: ...}``.
    reject_null : dict
        ``{significance_level: bool}``.
    n_entities : int
    n_periods : int
    individual_stats : ndarray or None
        Per-entity test statistics (if applicable).
    """
    test_name: str
    statistic: float
    pvalue: float
    critical_values: Dict[float, float]
    reject_null: Dict[float, bool]
    n_entities: int
    n_periods: int
    individual_stats: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Generate a formatted summary."""
        from pywaveletpanel.tables import UnitRootTable
        return UnitRootTable.from_single_result(self).render()


# ══════════════════════════════════════════════════════════════════════════════
# Fan-Gençay Wavelet Ratio (univariate helper)
# ══════════════════════════════════════════════════════════════════════════════

def _wavelet_ratio_stat(x: np.ndarray) -> float:
    """
    Compute the wavelet ratio statistic s_T for a univariate series.

    s_T = ||V_1||^2 / (||V_1||^2 + ||W_1||^2)

    Under H0 (unit root): s_T → 1
    Under H1 (stationary): s_T < 1

    Reference: Fan & Gençay (2010), Equation (2.7) in Paper 3.
    """
    V, W = haar_dwt(x, level=1)
    v_energy = np.sum(V ** 2)
    w_energy = np.sum(W[0] ** 2)
    total = v_energy + w_energy
    if total < 1e-15:
        return 1.0
    return v_energy / total


def _wavelet_ratio_stat_modwt(x: np.ndarray) -> float:
    """Wavelet ratio statistic using MODWT (Paper 3, Eq. 2.8)."""
    V, W = haar_modwt(x, level=1)
    v_energy = np.sum(V ** 2)
    w_energy = np.sum(W[0] ** 2)
    total = v_energy + w_energy
    if total < 1e-15:
        return 1.0
    return v_energy / total


def _adf_t_stat(x: np.ndarray, trend: str = "c") -> float:
    """Compute the Augmented Dickey-Fuller t-statistic for a single series."""
    T = len(x)
    dy = np.diff(x)
    y_lag = x[:-1]
    if trend == "c":
        X = np.column_stack([np.ones(T - 1), y_lag])
    elif trend == "ct":
        X = np.column_stack([np.ones(T - 1), np.arange(1, T), y_lag])
    else:
        X = y_lag.reshape(-1, 1)

    beta = np.linalg.lstsq(X, dy, rcond=None)[0]
    resid = dy - X @ beta
    sigma2 = np.sum(resid ** 2) / (T - 1 - X.shape[1])
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))

    # t-stat on the coefficient of y_{t-1} (last column)
    return beta[-1] / se[-1]


# ══════════════════════════════════════════════════════════════════════════════
# Wavelet Ratio IPS Test (Li & Shukur, 2013) — Paper 5
# ══════════════════════════════════════════════════════════════════════════════

class WaveletRatioIPS:
    """
    IPS panel unit root test using wavelet ratio statistics.

    Instead of the Dickey-Fuller t-statistic, uses the Fan-Gençay
    wavelet ratio statistic for each entity, then averages:

        S_NT = (1/N) * Σ S_iT

    Under H0 (all unit roots): S_NT is near 1.
    Under H1 (some stationary): S_NT < 1.

    References
    ----------
    Li, Y. & Shukur, G. (2013), Computational Economics, 41, 59–69.
    """

    def test(
        self,
        data: np.ndarray,
        n_mc: int = 10000,
        seed: Optional[int] = None,
    ) -> UnitRootResult:
        """
        Perform the wavelet ratio IPS test.

        Parameters
        ----------
        data : ndarray of shape (N, T)
            Panel data matrix. Each row is an entity's time series.
        n_mc : int
            Number of Monte Carlo replications for critical values.
        seed : int, optional

        Returns
        -------
        UnitRootResult
        """
        data = np.asarray(data, dtype=float)
        N, T = data.shape

        # Compute individual wavelet ratio stats
        individual_stats = np.array([_wavelet_ratio_stat(data[i]) for i in range(N)])
        S_NT = np.mean(individual_stats)

        # Monte Carlo critical values under H0
        rng = np.random.default_rng(seed)
        mc_stats = np.zeros(n_mc)
        for m in range(n_mc):
            # Simulate N independent random walks
            panel_h0 = np.cumsum(rng.standard_normal((N, T)), axis=1)
            mc_stats[m] = np.mean([
                _wavelet_ratio_stat(panel_h0[i]) for i in range(N)
            ])

        # P-value: proportion of MC stats ≤ observed (left-tail test)
        pvalue = float(np.mean(mc_stats <= S_NT))

        # Critical values
        critical_values = {
            0.01: float(np.percentile(mc_stats, 1)),
            0.05: float(np.percentile(mc_stats, 5)),
            0.10: float(np.percentile(mc_stats, 10)),
        }
        reject = {k: S_NT <= v for k, v in critical_values.items()}

        return UnitRootResult(
            test_name="Wavelet Ratio IPS (Li & Shukur, 2013)",
            statistic=S_NT,
            pvalue=pvalue,
            critical_values=critical_values,
            reject_null=reject,
            n_entities=N,
            n_periods=T,
            individual_stats=individual_stats,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Wavelet Wald Tests (Almasri et al., 2016) — Paper 3
# ══════════════════════════════════════════════════════════════════════════════

class WaveletWaldDWT:
    """
    Wavelet Wald panel unit root test using DWT (WDWT).

    Test statistic:
        W_DWT = T * tr[(H'H)^{-1} E'E] - C

    where H contains scaling coefficients and E contains total
    (wavelet + scaling) coefficients across entities.

    Robust to:
    - Cross-sectional dependence (non-diagonal structure)
    - Unknown structural breaks (wavelet-based averages/differences)

    Reference: Almasri et al. (2016), Equation (2.10).
    """

    def test(
        self,
        data: np.ndarray,
        n_mc: int = 10000,
        seed: Optional[int] = None,
    ) -> UnitRootResult:
        """
        Perform the WDWT panel unit root test.

        Parameters
        ----------
        data : ndarray of shape (N, T)
        n_mc : int
        seed : int, optional

        Returns
        -------
        UnitRootResult
        """
        data = np.asarray(data, dtype=float)
        N, T = data.shape
        stat = self._compute_stat(data)

        # Monte Carlo critical values
        rng = np.random.default_rng(seed)
        mc_stats = np.zeros(n_mc)
        for m in range(n_mc):
            panel_h0 = np.cumsum(rng.standard_normal((N, T)), axis=1)
            mc_stats[m] = self._compute_stat(panel_h0)

        pvalue = float(np.mean(mc_stats >= stat))
        critical_values = {
            0.01: float(np.percentile(mc_stats, 99)),
            0.05: float(np.percentile(mc_stats, 95)),
            0.10: float(np.percentile(mc_stats, 90)),
        }
        reject = {k: stat >= v for k, v in critical_values.items()}

        return UnitRootResult(
            test_name="Wavelet Wald DWT — WDWT (Almasri et al., 2016)",
            statistic=stat,
            pvalue=pvalue,
            critical_values=critical_values,
            reject_null=reject,
            n_entities=N,
            n_periods=T,
        )

    @staticmethod
    def _compute_stat(data: np.ndarray) -> float:
        """Compute the WDWT test statistic."""
        N, T = data.shape
        # Haar DWT level 1 for each entity
        H_rows, E_rows = [], []
        for i in range(N):
            V, W = haar_dwt(data[i], level=1)
            H_rows.append(V)
            E_rows.append(np.concatenate([V, W[0]]))

        # Ensure equal lengths
        min_h = min(len(r) for r in H_rows)
        min_e = min(len(r) for r in E_rows)
        H = np.column_stack([r[:min_h] for r in H_rows])  # (T/2, N)
        E = np.column_stack([r[:min_e] for r in E_rows])   # (T, N)

        try:
            HtH_inv = np.linalg.inv(H.T @ H)
            stat = T * np.trace(HtH_inv @ (E.T @ E)) - N
        except np.linalg.LinAlgError:
            stat = 0.0

        return stat


class WaveletWaldMODWT:
    """
    Wavelet Wald panel unit root test using MODWT (WMODWT).

    Same structure as WDWT but uses MODWT coefficients, which are
    more efficient (Percival & Mofjeld, 1997) and yield higher power
    for near-integrated alternatives.

    Reference: Almasri et al. (2016), Equation (2.11).
    """

    def test(
        self,
        data: np.ndarray,
        n_mc: int = 10000,
        seed: Optional[int] = None,
    ) -> UnitRootResult:
        """
        Perform the WMODWT panel unit root test.

        Parameters
        ----------
        data : ndarray of shape (N, T)
        n_mc : int
        seed : int, optional

        Returns
        -------
        UnitRootResult
        """
        data = np.asarray(data, dtype=float)
        N, T = data.shape
        stat = self._compute_stat(data)

        rng = np.random.default_rng(seed)
        mc_stats = np.zeros(n_mc)
        for m in range(n_mc):
            panel_h0 = np.cumsum(rng.standard_normal((N, T)), axis=1)
            mc_stats[m] = self._compute_stat(panel_h0)

        pvalue = float(np.mean(mc_stats >= stat))
        critical_values = {
            0.01: float(np.percentile(mc_stats, 99)),
            0.05: float(np.percentile(mc_stats, 95)),
            0.10: float(np.percentile(mc_stats, 90)),
        }
        reject = {k: stat >= v for k, v in critical_values.items()}

        return UnitRootResult(
            test_name="Wavelet Wald MODWT — WMODWT (Almasri et al., 2016)",
            statistic=stat,
            pvalue=pvalue,
            critical_values=critical_values,
            reject_null=reject,
            n_entities=N,
            n_periods=T,
        )

    @staticmethod
    def _compute_stat(data: np.ndarray) -> float:
        """Compute the WMODWT test statistic."""
        N, T = data.shape
        H_rows, E_rows = [], []
        for i in range(N):
            V, W = haar_modwt(data[i], level=1)
            H_rows.append(V)
            E_rows.append(V + W[0])

        H = np.column_stack(H_rows)  # (T, N)
        E = np.column_stack(E_rows)  # (T, N)

        try:
            HtH_inv = np.linalg.inv(H.T @ H)
            stat = T * np.trace(HtH_inv @ (E.T @ E)) - N
        except np.linalg.LinAlgError:
            stat = 0.0

        return stat


# ══════════════════════════════════════════════════════════════════════════════
# Standard IPS / ADF Panel Test (comparison baseline)
# ══════════════════════════════════════════════════════════════════════════════

class PanelADF:
    """
    Standard Im-Pesaran-Shin (IPS) panel unit root test using
    Dickey-Fuller t-statistics.

    This is the benchmark test against which the wavelet methods
    are compared in Papers 3 and 5.

    H0: All entities have a unit root (α_i = 0 for all i).
    H1: At least some entities are stationary.
    """

    def __init__(self, trend: str = "c"):
        self.trend = trend

    def test(
        self,
        data: np.ndarray,
        n_mc: int = 10000,
        seed: Optional[int] = None,
    ) -> UnitRootResult:
        """
        Perform the IPS panel unit root test.

        Parameters
        ----------
        data : ndarray of shape (N, T)
        n_mc : int
        seed : int, optional

        Returns
        -------
        UnitRootResult
        """
        data = np.asarray(data, dtype=float)
        N, T = data.shape

        individual_stats = np.array([
            _adf_t_stat(data[i], trend=self.trend) for i in range(N)
        ])
        t_NT = np.mean(individual_stats)

        # Monte Carlo critical values
        rng = np.random.default_rng(seed)
        mc_stats = np.zeros(n_mc)
        for m in range(n_mc):
            panel_h0 = np.cumsum(rng.standard_normal((N, T)), axis=1)
            mc_stats[m] = np.mean([
                _adf_t_stat(panel_h0[i], trend=self.trend) for i in range(N)
            ])

        pvalue = float(np.mean(mc_stats <= t_NT))
        critical_values = {
            0.01: float(np.percentile(mc_stats, 1)),
            0.05: float(np.percentile(mc_stats, 5)),
            0.10: float(np.percentile(mc_stats, 10)),
        }
        reject = {k: t_NT <= v for k, v in critical_values.items()}

        return UnitRootResult(
            test_name="IPS Panel ADF (Im, Pesaran & Shin, 2003)",
            statistic=t_NT,
            pvalue=pvalue,
            critical_values=critical_values,
            reject_null=reject,
            n_entities=N,
            n_periods=T,
            individual_stats=individual_stats,
        )
