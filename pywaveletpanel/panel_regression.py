"""
Scale-by-scale wavelet panel regression.

Implements the methodology from:
- Gallegati et al. (2015), "Productivity and unemployment: a scale-by-scale
  panel data analysis for the G7 countries", SNDE.
- Karlsson et al. (2020), "Unveiling the time-dependent dynamics between
  oil prices and exchange rates", The Energy Journal.

The approach:
1. Decompose each panel variable via MODWT into detail (D1..DJ) and smooth (SJ).
2. Stack each scale into a separate panel dataset.
3. Estimate fixed-effects OLS at each scale with Newey-West standard errors.

Example
-------
>>> import numpy as np
>>> from pywaveletpanel import WaveletPanelOLS
>>> model = WaveletPanelOLS(wavelet='sym4', level=3)
>>> result = model.fit(y=panel_y, X=panel_X, entity_ids=ids, time_ids=times)
>>> print(result.summary())
>>> result.plot()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from pywaveletpanel.wavelets import modwt_mra
from pywaveletpanel.utils import (
    panel_fixed_effects_ols,
    newey_west_se,
    significance_stars,
    format_coef,
)


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScaleRegressionResult:
    """
    Results from scale-by-scale panel regression.

    Attributes
    ----------
    scale_results : dict
        Mapping from scale name (e.g. ``'D1'``, ``'S3'``) to OLS result dict.
    aggregate_result : dict or None
        OLS result using raw (non-decomposed) aggregate data.
    scale_labels : dict
        Mapping from scale name to frequency-band description.
    n_entities : int
    n_periods : int
    wavelet : str
    level : int
    regressor_names : list of str
    """
    scale_results: Dict[str, dict]
    aggregate_result: Optional[dict]
    scale_labels: Dict[str, str]
    n_entities: int
    n_periods: int
    wavelet: str
    level: int
    regressor_names: List[str] = field(default_factory=list)

    # ── Summary table ────────────────────────────────────────────────────

    def summary(self, decimals: int = 3) -> str:
        """
        Generate a journal-quality summary table.

        Modeled after Table 1 in Gallegati et al. (2015) and
        Tables 2–3 in Karlsson et al. (2020).
        """
        from pywaveletpanel.tables import RegressionTable
        return RegressionTable.from_scale_result(self, decimals=decimals).render()

    def summary_df(self) -> pd.DataFrame:
        """Return results as a tidy DataFrame."""
        rows = []
        scales = (["Aggregate"] if self.aggregate_result else []) + \
                 [f"S{self.level}"] + [f"D{j}" for j in range(self.level, 0, -1)]

        for scale in scales:
            if scale == "Aggregate":
                res = self.aggregate_result
                label = "Raw data"
            else:
                if scale not in self.scale_results:
                    continue
                res = self.scale_results[scale]
                label = self.scale_labels.get(scale, scale)

            for p in range(len(res["coef"])):
                name = self.regressor_names[p] if p < len(self.regressor_names) else f"x{p+1}"
                rows.append({
                    "Scale": scale,
                    "Frequency band": label,
                    "Variable": name,
                    "Coefficient": res["coef"][p],
                    "Std. Error": res["se"][p],
                    "t-statistic": res["t_stat"][p],
                    "p-value": res["pvalue"][p],
                    "R²": res["r_squared"],
                    "Adj. R²": res.get("adj_r_squared", np.nan),
                    "N obs": res["nobs"],
                })
        return pd.DataFrame(rows)

    def to_latex(self, decimals: int = 3) -> str:
        """Export as a LaTeX table."""
        from pywaveletpanel.tables import RegressionTable
        return RegressionTable.from_scale_result(self, decimals=decimals).to_latex()

    def plot(self, figsize: tuple = (10, 6), **kwargs):
        """Plot scale-dependent coefficients with confidence intervals."""
        from pywaveletpanel.visualization import plot_scale_coefficients
        return plot_scale_coefficients(self, figsize=figsize, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Main estimator
# ══════════════════════════════════════════════════════════════════════════════

class WaveletPanelOLS:
    """
    Scale-by-scale wavelet panel regression with fixed effects.

    Decomposes panel variables via MODWT, then estimates a fixed-effects
    model at each wavelet scale separately.

    Parameters
    ----------
    wavelet : str
        Wavelet filter name. ``'sym4'`` for LA(8) as in Papers 2, 4.
        ``'haar'`` for Haar as in Papers 1, 3.
    level : int
        Decomposition level J.
        - J=3 for annual data (Paper 4): D1(2–4yr), D2(4–8yr), D3(8–16yr), S3(>16yr)
        - J=4 for monthly data (Paper 2): D1(1–2mo), D2(2–4mo), D3(4–8mo), D4(8–16mo)
    robust : bool
        Use Newey-West HAC standard errors (default True, as in Paper 2).
    nw_lags : int or None
        Newey-West lag truncation. None = automatic.
    include_aggregate : bool
        Also estimate the model using raw (non-decomposed) data for comparison.

    Example
    -------
    >>> model = WaveletPanelOLS(wavelet='sym4', level=3)
    >>> result = model.fit(y, X, entity_ids, time_ids)
    >>> print(result.summary())
    """

    def __init__(
        self,
        wavelet: str = "sym4",
        level: int = 3,
        robust: bool = True,
        nw_lags: Optional[int] = None,
        include_aggregate: bool = True,
    ):
        self.wavelet = wavelet
        self.level = level
        self.robust = robust
        self.nw_lags = nw_lags
        self.include_aggregate = include_aggregate

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        entity_ids: np.ndarray,
        time_ids: Optional[np.ndarray] = None,
        regressor_names: Optional[List[str]] = None,
    ) -> ScaleRegressionResult:
        """
        Fit the scale-by-scale panel regression.

        Parameters
        ----------
        y : ndarray of shape (N*T,)
            Dependent variable (stacked panel).
        X : ndarray of shape (N*T,) or (N*T, k)
            Regressors (stacked panel).
        entity_ids : ndarray of shape (N*T,)
            Entity (country/firm) identifiers.
        time_ids : ndarray of shape (N*T,), optional
            Time period identifiers.
        regressor_names : list of str, optional

        Returns
        -------
        ScaleRegressionResult
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

        # ── Determine T per entity ───────────────────────────────────────
        T_per = {}
        for eid in entities:
            T_per[eid] = np.sum(entity_ids == eid)
        T_vals = list(T_per.values())
        if len(set(T_vals)) > 1:
            raise ValueError(
                "Unbalanced panel detected. All entities must have the same "
                f"number of time periods. Found: {set(T_vals)}"
            )
        T = T_vals[0]

        # ── Decompose each entity's y and X via MODWT MRA ────────────────
        y_components: Dict[str, np.ndarray] = {}
        X_components: Dict[str, np.ndarray] = {}
        labels = {}

        for eid in entities:
            mask = entity_ids == eid
            if time_ids is not None:
                order = np.argsort(time_ids[mask])
            else:
                order = np.arange(T)

            y_i = y[mask][order]
            X_i = X[mask][order]

            # Decompose y
            mra_y = modwt_mra(y_i, wavelet=self.wavelet, level=self.level)
            if not labels:
                labels = mra_y["labels"]

            for key in mra_y:
                if key == "labels":
                    continue
                if key not in y_components:
                    y_components[key] = []
                y_components[key].append(mra_y[key])

            # Decompose each column of X
            for p in range(k):
                mra_x = modwt_mra(X_i[:, p], wavelet=self.wavelet, level=self.level)
                for key in mra_x:
                    if key == "labels":
                        continue
                    col_key = (key, p)
                    if col_key not in X_components:
                        X_components[col_key] = []
                    X_components[col_key].append(mra_x[key])

        # ── Build scale-specific panels and estimate ─────────────────────
        scale_names = [f"D{j}" for j in range(1, self.level + 1)] + [f"S{self.level}"]
        scale_results = {}

        for scale in scale_names:
            # Stack y
            y_scale = np.concatenate(y_components[scale])
            # Stack X
            X_scale = np.column_stack([
                np.concatenate(X_components[(scale, p)])
                for p in range(k)
            ])
            # Entity IDs for stacked data
            eids_scale = np.repeat(np.arange(N), T)

            # Fixed-effects OLS
            res = panel_fixed_effects_ols(
                y_scale, X_scale, eids_scale,
                robust=self.robust, n_lags=self.nw_lags,
            )
            scale_results[scale] = res

        # ── Aggregate (raw data) estimation ──────────────────────────────
        agg_result = None
        if self.include_aggregate:
            eids_all = np.repeat(np.arange(N), T)
            # Restack raw data in entity order
            y_raw_parts = []
            X_raw_parts = []
            for eid in entities:
                mask = entity_ids == eid
                if time_ids is not None:
                    order = np.argsort(time_ids[mask])
                else:
                    order = np.arange(T)
                y_raw_parts.append(y[mask][order])
                X_raw_parts.append(X[mask][order])
            y_raw = np.concatenate(y_raw_parts)
            X_raw = np.vstack(X_raw_parts)
            agg_result = panel_fixed_effects_ols(
                y_raw, X_raw, eids_all,
                robust=self.robust, n_lags=self.nw_lags,
            )

        return ScaleRegressionResult(
            scale_results=scale_results,
            aggregate_result=agg_result,
            scale_labels=labels,
            n_entities=N,
            n_periods=T,
            wavelet=self.wavelet,
            level=self.level,
            regressor_names=regressor_names,
        )
