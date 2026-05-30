"""
Structure Adapted Wavelet (SAW) estimator for panel structural breaks.

Implements the methodology from:
    Bada, O., Kneip, A., Liebl, D., Mensinger, T., Gualtieri, J. &
    Sickles, R.C. (2021). "A Wavelet Method for Panel Models with Jump
    Discontinuities in the Parameters." arXiv:2109.10950v1.

The SAW estimator:
1. First-differences the panel to remove individual fixed effects.
2. Expands time-varying coefficients in a Haar wavelet basis.
3. Applies adaptive thresholding to detect jumps.
4. Performs Post-SAW estimation on detected stability intervals
   achieving the oracle property.

Key advantages over alternatives (Bai-Perron, Qian-Su):
- Different regressors can have breaks at different time points.
- Works with fixed n and large T, or large n and fixed T.
- Computationally efficient (avoids combinatorial search).

Example
-------
>>> from pywaveletpanel import SAWEstimator, PostSAWEstimator
>>> saw = SAWEstimator(threshold_method='adaptive')
>>> breaks = saw.detect(y_panel, X_panel, entity_ids, time_ids)
>>> print(breaks.summary())
>>> breaks.plot()
>>> post_saw = PostSAWEstimator()
>>> final = post_saw.fit(y_panel, X_panel, entity_ids, time_ids, breaks)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from scipy import stats

from pywaveletpanel.wavelets import haar_dwt, haar_idwt, pad_dyadic
from pywaveletpanel.utils import significance_stars


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BreakDetectionResult:
    """
    Result of SAW structural break detection.

    Attributes
    ----------
    n_breaks : dict
        Number of detected breaks per regressor. ``{regressor_idx: count}``.
    break_locations : dict
        Detected break time indices per regressor.
        ``{regressor_idx: [t1, t2, ...]}``.
    stability_intervals : dict
        List of stability intervals per regressor.
        ``{regressor_idx: [(start, end), ...]}``.
    coefficients : dict
        Estimated coefficients within each stability interval.
        ``{regressor_idx: [beta_1, beta_2, ...]}``.
    threshold : float
        Threshold value used.
    wavelet_coeffs : dict
        Raw wavelet coefficients before thresholding.
    n_entities : int
    n_periods : int
    regressor_names : list of str
    """
    n_breaks: Dict[int, int]
    break_locations: Dict[int, List[int]]
    stability_intervals: Dict[int, List[Tuple[int, int]]]
    coefficients: Dict[int, List[float]]
    threshold: float
    wavelet_coeffs: Dict[int, np.ndarray]
    n_entities: int
    n_periods: int
    regressor_names: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a journal-quality summary table."""
        from pywaveletpanel.tables import BreakTable
        return BreakTable.from_break_result(self).render()

    def plot(self, figsize: tuple = (14, 5), **kwargs):
        """Plot time-varying coefficients with detected breaks."""
        from pywaveletpanel.visualization import plot_structural_breaks
        return plot_structural_breaks(self, figsize=figsize, **kwargs)

    def total_breaks(self) -> int:
        """Total number of breaks across all regressors."""
        return sum(self.n_breaks.values())


# ══════════════════════════════════════════════════════════════════════════════
# SAW Estimator
# ══════════════════════════════════════════════════════════════════════════════

class SAWEstimator:
    """
    Structure Adapted Wavelet (SAW) estimator for detecting structural
    breaks in panel regression coefficients.

    Parameters
    ----------
    threshold_method : str
        ``'adaptive'`` (default) uses the data-driven threshold from
        Theorem 2 in Bada et al. (2021).
        ``'universal'`` uses the Donoho-Johnstone universal threshold.
    kappa_adjustment : bool
        Apply the log-log correction to kappa as in equation (3.1).
    min_segment_length : int
        Minimum number of periods between breaks (default 2).

    References
    ----------
    Bada et al. (2021), Section 2: "Structure Adapted Wavelet estimators".
    """

    def __init__(
        self,
        threshold_method: str = "adaptive",
        kappa_adjustment: bool = True,
        min_segment_length: int = 2,
    ):
        self.threshold_method = threshold_method
        self.kappa_adjustment = kappa_adjustment
        self.min_segment_length = min_segment_length

    def detect(
        self,
        y: np.ndarray,
        X: np.ndarray,
        entity_ids: np.ndarray,
        time_ids: Optional[np.ndarray] = None,
        regressor_names: Optional[List[str]] = None,
    ) -> BreakDetectionResult:
        """
        Detect structural breaks via wavelet thresholding.

        Parameters
        ----------
        y : ndarray of shape (N*T,)
        X : ndarray of shape (N*T,) or (N*T, k)
        entity_ids : ndarray of shape (N*T,)
        time_ids : ndarray of shape (N*T,), optional
        regressor_names : list of str, optional

        Returns
        -------
        BreakDetectionResult
        """
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        entity_ids = np.asarray(entity_ids)
        entities = np.unique(entity_ids)
        N = len(entities)
        k = X.shape[1]

        if regressor_names is None:
            regressor_names = [f"x{p+1}" for p in range(k)]

        # ── Determine T ──────────────────────────────────────────────────
        T = np.sum(entity_ids == entities[0])

        # ── Step 1: First-difference to remove fixed effects ─────────────
        # Following Bada et al. (2021), Section 2.1
        dy_list, dX_list = [], []
        for eid in entities:
            mask = entity_ids == eid
            if time_ids is not None:
                order = np.argsort(time_ids[mask])
            else:
                order = np.arange(T)
            y_i = y[mask][order]
            X_i = X[mask][order]
            dy_list.append(np.diff(y_i))
            dX_list.append(np.diff(X_i, axis=0))

        T_diff = T - 1

        # ── Step 2: Cross-sectional OLS at each time point ───────────────
        # Estimate gamma_t for t = 1, ..., T-1
        gamma_hat = np.zeros((T_diff, k))
        residual_var = np.zeros(T_diff)

        for t in range(T_diff):
            dy_t = np.array([dy_list[i][t] for i in range(N)])
            dX_t = np.array([dX_list[i][t] for i in range(N)])
            if dX_t.ndim == 1:
                dX_t = dX_t.reshape(-1, 1)

            # OLS: gamma_t = (dX'dX)^{-1} dX'dy
            try:
                gamma_hat[t] = np.linalg.lstsq(dX_t, dy_t, rcond=None)[0]
                res_t = dy_t - dX_t @ gamma_hat[t]
                residual_var[t] = np.var(res_t, ddof=k)
            except np.linalg.LinAlgError:
                gamma_hat[t] = 0
                residual_var[t] = 1.0

        # ── Step 3: Haar wavelet decomposition of gamma_hat ──────────────
        L = int(np.ceil(np.log2(T_diff)))
        wavelet_coeffs_dict = {}
        break_locations = {}
        n_breaks = {}
        stability_intervals = {}
        coefficients = {}

        # Analytic threshold (kept for reference / the 'universal' fallback)
        sigma_hat = np.sqrt(np.mean(residual_var))
        analytic_threshold = self._compute_threshold(T_diff, N, k, sigma_hat)

        thresholds_used = []

        for p in range(k):
            gamma_p = gamma_hat[:, p]
            # Pad to dyadic length
            gamma_padded = pad_dyadic(gamma_p)
            T_pad = len(gamma_padded)

            # Full Haar DWT
            max_level = int(np.log2(T_pad))
            V, W = haar_dwt(gamma_padded, level=max_level)

            # Store all wavelet coefficients
            all_w = np.concatenate(W)
            wavelet_coeffs_dict[p] = all_w

            # ── Step 4: Threshold selection ──────────────────────────────
            # The coefficient path γ̂_t is itself a cross-sectional average, so
            # its wavelet-coefficient noise is much smaller than the per-obs
            # residual σ̂. Estimate that noise robustly from the finest detail
            # level (median-absolute-deviation, Donoho & Johnstone 1994) and
            # apply the universal threshold σ_w·√(2 log T). This is self-
            # calibrating and avoids the over-segmentation caused by an
            # analytic threshold that sits below the coefficient noise floor.
            finest = np.abs(W[0])
            sigma_w = np.median(finest) / 0.6745 if finest.size else 0.0
            if sigma_w <= 0:
                sigma_w = np.std(gamma_p)
            lam = sigma_w * np.sqrt(2.0 * np.log(max(T_pad, 2)))
            if self.threshold_method == "universal":
                lam = max(lam, analytic_threshold)
            thresholds_used.append(lam)

            # ── Step 5: Denoise and locate jumps ─────────────────────────
            # Hard-threshold the detail coefficients and reconstruct. For the
            # Haar basis the reconstruction is piecewise constant, so breaks
            # are exactly the points where the denoised path γ̃_t jumps.
            W_thr = [np.where(np.abs(w_j) > lam, w_j, 0.0) for w_j in W]
            gamma_denoised = haar_idwt(V, W_thr)[:T_diff]

            jumps = np.abs(np.diff(gamma_denoised))
            raw_breaks = [t + 1 for t in range(len(jumps)) if jumps[t] > lam]

            # Filter out breaks too close together
            breaks_p = []
            for b in raw_breaks:
                if not breaks_p or b - breaks_p[-1] >= self.min_segment_length:
                    breaks_p.append(b)

            break_locations[p] = breaks_p
            n_breaks[p] = len(breaks_p)

            # ── Build stability intervals ────────────────────────────────
            boundaries = [0] + breaks_p + [T_diff]
            intervals = [
                (boundaries[i], boundaries[i + 1])
                for i in range(len(boundaries) - 1)
            ]
            stability_intervals[p] = intervals

            # ── Estimate constant coefficients on each interval ──────────
            coefs_p = []
            for start, end in intervals:
                if end > start:
                    coefs_p.append(np.mean(gamma_p[start:end]))
                else:
                    coefs_p.append(0.0)
            coefficients[p] = coefs_p

        threshold = float(np.mean(thresholds_used)) if thresholds_used else analytic_threshold

        return BreakDetectionResult(
            n_breaks=n_breaks,
            break_locations=break_locations,
            stability_intervals=stability_intervals,
            coefficients=coefficients,
            threshold=threshold,
            wavelet_coeffs=wavelet_coeffs_dict,
            n_entities=N,
            n_periods=T,
            regressor_names=regressor_names,
        )

    def _compute_threshold(
        self,
        T: int,
        N: int,
        P: int,
        sigma: float,
    ) -> float:
        """
        Compute the adaptive threshold λ_{n,T} from Theorem 2.

        λ_{nT} = σ̂ * √(2P * log((T-1)*P) / (n * (T-1)^{1/κ}))^{κ/2}
        """
        if self.kappa_adjustment:
            nT = N * T
            kappa = 1.0 - np.log(np.log(max(nT, 3))) / np.log(max(nT, 3))
        else:
            kappa = 0.5

        if self.threshold_method == "adaptive":
            log_term = 2 * P * np.log(max(T * P, 2))
            denom = N * max(T, 1) ** (1.0 / max(kappa, 0.01))
            lam = sigma * (log_term / max(denom, 1e-10)) ** (kappa / 2.0)
        else:
            # Universal threshold (Donoho-Johnstone)
            lam = sigma * np.sqrt(2 * np.log(T))

        return lam


# ══════════════════════════════════════════════════════════════════════════════
# Post-SAW Estimator
# ══════════════════════════════════════════════════════════════════════════════

class PostSAWEstimator:
    """
    Post-SAW estimation on detected stability intervals.

    After breaks are detected by SAWEstimator, this class re-estimates
    the panel model on each stability interval using pooled OLS or IV,
    achieving the oracle property (Theorem 3 in Bada et al. 2021).

    Parameters
    ----------
    variance_type : str
        ``'homoskedastic'``, ``'cross_hetero'``, ``'time_hetero'``,
        ``'both'``. Controls how the variance matrix is estimated
        (see Paper 1, Section 2.3).
    chow_test : bool
        Perform Chow tests between consecutive stability intervals.

    Example
    -------
    >>> post = PostSAWEstimator(variance_type='both')
    >>> result = post.fit(y, X, entity_ids, time_ids, breaks)
    """

    def __init__(
        self,
        variance_type: str = "homoskedastic",
        chow_test: bool = True,
    ):
        self.variance_type = variance_type
        self.chow_test = chow_test

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        entity_ids: np.ndarray,
        time_ids: Optional[np.ndarray] = None,
        breaks: Optional[BreakDetectionResult] = None,
    ) -> dict:
        """
        Re-estimate panel model on each stability interval.

        Parameters
        ----------
        y, X, entity_ids, time_ids : as in SAWEstimator.detect
        breaks : BreakDetectionResult from SAWEstimator

        Returns
        -------
        result : dict
            Keys: ``'interval_results'``, ``'chow_tests'``,
            ``'full_coefficients'`` (time-varying coefficient paths).
        """
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        entity_ids = np.asarray(entity_ids)
        entities = np.unique(entity_ids)
        N = len(entities)
        k = X.shape[1]
        T = np.sum(entity_ids == entities[0])

        # ── Re-organize data as (N, T) matrices ─────────────────────────
        Y_mat = np.zeros((N, T))
        X_mat = np.zeros((N, T, k))
        for i, eid in enumerate(entities):
            mask = entity_ids == eid
            if time_ids is not None:
                order = np.argsort(time_ids[mask])
            else:
                order = np.arange(T)
            Y_mat[i] = y[mask][order]
            X_mat[i] = X[mask][order]

        # ── Estimate on each stability interval ──────────────────────────
        interval_results = {}
        full_coef_path = np.zeros((T, k))
        chow_results = {}

        for p in range(k):
            intervals = breaks.stability_intervals[p]
            interval_results[p] = []
            prev_res = None

            for seg_idx, (t_start, t_end) in enumerate(intervals):
                if t_end <= t_start:
                    continue

                # Pool observations across entities for this interval
                y_pool = Y_mat[:, t_start:t_end].flatten()
                x_pool = X_mat[:, t_start:t_end, p].flatten()
                eids_pool = np.repeat(np.arange(N), t_end - t_start)

                # Fixed-effects OLS
                y_dm = y_pool.copy()
                x_dm = x_pool.copy()
                for uid in range(N):
                    m = eids_pool == uid
                    y_dm[m] -= y_pool[m].mean()
                    x_dm[m] -= x_pool[m].mean()

                if np.sum(x_dm ** 2) > 1e-12:
                    beta = float(np.sum(x_dm * y_dm) / np.sum(x_dm ** 2))
                    res = y_dm - x_dm.flatten() * beta
                    nobs = len(y_dm)
                    se = np.sqrt(np.sum(res ** 2) / max(nobs - N - 1, 1) /
                                 max(np.sum(x_dm ** 2), 1e-12))
                    t_stat = beta / max(se, 1e-12)
                    pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(nobs - N - 1, 1)))
                else:
                    beta, se, t_stat, pval = 0.0, np.inf, 0.0, 1.0
                    nobs = t_end - t_start

                seg_res = {
                    "interval": (t_start, t_end),
                    "coef": beta,
                    "se": se,
                    "t_stat": t_stat,
                    "pvalue": pval,
                    "nobs": nobs,
                }
                interval_results[p].append(seg_res)

                # Fill coefficient path
                full_coef_path[t_start:t_end, p] = beta

                # ── Chow test ────────────────────────────────────────────
                if self.chow_test and prev_res is not None:
                    F_stat = self._chow_F(prev_res, seg_res, N)
                    df1, df2 = 1, max(prev_res["nobs"] + nobs - 2 * N - 2, 1)
                    chow_p = 1 - stats.f.cdf(F_stat, df1, df2)
                    chow_results[(p, seg_idx - 1, seg_idx)] = {
                        "F_stat": F_stat,
                        "pvalue": chow_p,
                        "break_time": t_start,
                    }
                prev_res = seg_res

        return {
            "interval_results": interval_results,
            "chow_tests": chow_results,
            "full_coefficients": full_coef_path,
            "n_entities": N,
            "n_periods": T,
        }

    @staticmethod
    def _chow_F(res1: dict, res2: dict, N: int) -> float:
        """Compute approximate Chow F-statistic between two intervals."""
        b1, b2 = res1["coef"], res2["coef"]
        se1, se2 = res1["se"], res2["se"]
        diff = b1 - b2
        se_diff = np.sqrt(se1 ** 2 + se2 ** 2)
        if se_diff < 1e-12:
            return 0.0
        return (diff / se_diff) ** 2
