"""
Utility functions for panel data econometrics.

Provides:
- Newey-West (HAC) standard errors
- Fixed-effects (within) transformation
- First-difference transformation
- Significance stars formatting
- Monte Carlo critical-value simulation helpers

References
----------
Newey, W.K. & West, K.D. (1987). A Simple, Positive Semi-definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
    Econometrica, 55(3), 703–708.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple


# ── Newey-West HAC standard errors ───────────────────────────────────────────

def newey_west_se(
    X: np.ndarray,
    residuals: np.ndarray,
    n_lags: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Newey-West heteroskedasticity and autocorrelation consistent
    (HAC) standard errors.

    Parameters
    ----------
    X : ndarray of shape (T, k)
        Regressor matrix.
    residuals : ndarray of shape (T,)
        OLS residuals.
    n_lags : int, optional
        Number of lags for the Bartlett kernel. Defaults to
        ``floor(4 * (T / 100)^(2/9))`` (Newey-West rule of thumb).

    Returns
    -------
    se : ndarray of shape (k,)
        HAC standard errors for each coefficient.
    """
    T, k = X.shape
    if n_lags is None:
        n_lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

    # Meat: S_0
    e = residuals.reshape(-1, 1)
    Xe = X * e  # (T, k)
    S = Xe.T @ Xe / T  # (k, k)

    # Bartlett kernel
    for lag in range(1, n_lags + 1):
        w = 1.0 - lag / (n_lags + 1.0)
        Gamma = Xe[lag:].T @ Xe[:-lag] / T
        S += w * (Gamma + Gamma.T)

    # Bread
    XtX_inv = np.linalg.inv(X.T @ X / T)
    V = XtX_inv @ S @ XtX_inv / T

    return np.sqrt(np.diag(V))


# ── Panel transformations ────────────────────────────────────────────────────

def fixed_effects_transform(
    y: np.ndarray,
    entity_ids: np.ndarray,
) -> np.ndarray:
    """
    Apply within (fixed-effects) transformation: demean each entity.

    Parameters
    ----------
    y : ndarray of shape (N*T,) or (N*T, k)
        Panel data in stacked form.
    entity_ids : ndarray of shape (N*T,)
        Integer entity identifiers (0, 1, ..., N-1).

    Returns
    -------
    y_demean : ndarray, same shape as *y*
        Demeaned data.
    """
    y = np.asarray(y, dtype=float)
    ids = np.asarray(entity_ids)
    y_demean = y.copy()
    for uid in np.unique(ids):
        mask = ids == uid
        if y.ndim == 1:
            y_demean[mask] -= y[mask].mean()
        else:
            y_demean[mask] -= y[mask].mean(axis=0)
    return y_demean


def first_difference(
    y: np.ndarray,
    entity_ids: np.ndarray,
    time_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    First-difference panel data within each entity.

    Parameters
    ----------
    y : ndarray of shape (N*T,) or (N*T, k)
    entity_ids : ndarray of shape (N*T,)
    time_ids : ndarray of shape (N*T,), optional
        If provided, used for sorting; otherwise assumes data is sorted.

    Returns
    -------
    dy : ndarray
        First-differenced data (shorter by N observations).
    mask : ndarray of bool
        Mask of retained observations.
    """
    y = np.asarray(y, dtype=float)
    ids = np.asarray(entity_ids)
    keep = np.ones(len(y), dtype=bool)
    dy = np.zeros_like(y)

    for uid in np.unique(ids):
        idx = np.where(ids == uid)[0]
        if time_ids is not None:
            idx = idx[np.argsort(time_ids[idx])]
        keep[idx[0]] = False
        if y.ndim == 1:
            dy[idx[1:]] = np.diff(y[idx])
        else:
            dy[idx[1:]] = np.diff(y[idx], axis=0)

    return dy[keep], keep


# ── Significance stars ───────────────────────────────────────────────────────

def significance_stars(pvalue: float) -> str:
    """Return significance stars: *** (1%), ** (5%), * (10%)."""
    if pvalue <= 0.01:
        return "***"
    elif pvalue <= 0.05:
        return "**"
    elif pvalue <= 0.10:
        return "*"
    return ""


def format_coef(value: float, pvalue: float, decimals: int = 4) -> str:
    """Format a coefficient with significance stars."""
    stars = significance_stars(pvalue)
    return f"{value:.{decimals}f}{stars}"


# ── OLS helpers ──────────────────────────────────────────────────────────────

def ols_fit(
    y: np.ndarray,
    X: np.ndarray,
    robust: bool = True,
    n_lags: Optional[int] = None,
) -> dict:
    """
    Simple OLS estimation with optional Newey-West standard errors.

    Parameters
    ----------
    y : ndarray of shape (T,)
    X : ndarray of shape (T, k)
    robust : bool
        If True, use Newey-West HAC standard errors.
    n_lags : int, optional

    Returns
    -------
    result : dict with keys 'coef', 'se', 't_stat', 'pvalue', 'r_squared',
             'residuals', 'nobs'.
    """
    T, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if robust:
        se = newey_west_se(X, residuals, n_lags=n_lags)
    else:
        sigma2 = ss_res / (T - k)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * np.diag(XtX_inv))

    t_stat = beta / se
    pvalue = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=T - k))

    return {
        "coef": beta,
        "se": se,
        "t_stat": t_stat,
        "pvalue": pvalue,
        "r_squared": r_sq,
        "adj_r_squared": 1.0 - (1.0 - r_sq) * (T - 1) / (T - k - 1),
        "residuals": residuals,
        "nobs": T,
    }


def panel_fixed_effects_ols(
    y: np.ndarray,
    X: np.ndarray,
    entity_ids: np.ndarray,
    robust: bool = True,
    n_lags: Optional[int] = None,
) -> dict:
    """
    Fixed-effects panel OLS with optional Newey-West standard errors.

    This is the core estimator used in Gallegati et al. (2015) and
    Karlsson et al. (2020).
    """
    y_dm = fixed_effects_transform(y, entity_ids)
    X_dm = fixed_effects_transform(X, entity_ids)
    N = len(np.unique(entity_ids))
    result = ols_fit(y_dm, X_dm, robust=robust, n_lags=n_lags)
    # Adjust degrees of freedom for entity dummies
    T = len(y)
    k = X.shape[1]
    result["df"] = T - N - k
    return result


# ── Monte Carlo helpers ──────────────────────────────────────────────────────

def simulate_panel_ar1(
    N: int,
    T: int,
    rho: float = 1.0,
    cross_corr: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate an AR(1) panel: y_{it} = rho * y_{i,t-1} + u_{it}.

    Parameters
    ----------
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    rho : float
        Autoregressive parameter. rho=1 gives unit root.
    cross_corr : float
        Equi-correlation across units (0 = independent).
    seed : int, optional

    Returns
    -------
    Y : ndarray of shape (N, T)
    """
    rng = np.random.default_rng(seed)
    # Covariance matrix
    Sigma = np.full((N, N), cross_corr)
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)

    Y = np.zeros((N, T))
    for t in range(1, T):
        eps = L @ rng.standard_normal(N)
        Y[:, t] = rho * Y[:, t - 1] + eps
    return Y
