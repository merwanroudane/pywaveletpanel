"""
Wavelet transforms for panel data analysis.

Implements:
- Haar DWT / IDWT (Papers 1, 3, 5)
- Haar MODWT / IMODWT (Paper 3)
- LA(8) MODWT (Papers 2, 4)
- General MODWT with arbitrary PyWavelets filters
- Multiresolution Analysis (MRA) reconstruction
- Dyadic padding with boundary reflection

References
----------
Percival, D.B. & Walden, A.T. (2000). Wavelet Methods for Time Series
    Analysis. Cambridge University Press.
Mallat, S.G. (1989). A theory for multiresolution signal decomposition.
    IEEE PAMI, 11(7), 674–693.
Fan, Y. & Gençay, R. (2010). Unit root tests with wavelets.
    Econometric Theory, 26, 1305–1331.
"""

from __future__ import annotations

import numpy as np
import pywt
from typing import List, Tuple, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Haar Discrete Wavelet Transform (DWT)
# ══════════════════════════════════════════════════════════════════════════════

def haar_dwt(x: np.ndarray, level: int = 1) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute the Haar DWT up to *level* J.

    The Haar filter is the simplest wavelet and is chosen because:
    1. It has no boundary coefficient issues (Paper 3, Section II).
    2. It directly computes averages and differences of adjacent pairs.
    3. It is optimal for detecting jump discontinuities (Paper 1).

    Parameters
    ----------
    x : ndarray of shape (T,)
        Input time series (length should be divisible by ``2**level``).
    level : int
        Decomposition level J.

    Returns
    -------
    V_J : ndarray
        Scaling (approximation) coefficients at level J.
    W : list of ndarray
        Wavelet (detail) coefficients ``[W_1, W_2, ..., W_J]``.
    """
    x = np.asarray(x, dtype=float)
    W_list = []
    v = x.copy()
    for _ in range(level):
        n = len(v)
        if n % 2 != 0:
            v = np.append(v, v[-1])  # reflect boundary
            n += 1
        v_even = v[0::2]
        v_odd = v[1::2]
        w_j = (v_even - v_odd) / np.sqrt(2)
        v_j = (v_even + v_odd) / np.sqrt(2)
        W_list.append(w_j)
        v = v_j
    return v, W_list


def haar_idwt(V_J: np.ndarray, W: List[np.ndarray]) -> np.ndarray:
    """
    Inverse Haar DWT — reconstruct the original signal.

    Parameters
    ----------
    V_J : ndarray
        Scaling coefficients at the coarsest level.
    W : list of ndarray
        Wavelet coefficients ``[W_1, W_2, ..., W_J]``.

    Returns
    -------
    x : ndarray
        Reconstructed signal.
    """
    v = V_J.copy()
    for w_j in reversed(W):
        n = len(w_j)
        x_even = (v[:n] + w_j) / np.sqrt(2)
        x_odd = (v[:n] - w_j) / np.sqrt(2)
        v = np.empty(2 * n)
        v[0::2] = x_even
        v[1::2] = x_odd
    return v


# ══════════════════════════════════════════════════════════════════════════════
# Haar MODWT (Maximal Overlap DWT)
# ══════════════════════════════════════════════════════════════════════════════

def haar_modwt(x: np.ndarray, level: int = 1) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute the Haar MODWT up to *level* J.

    Unlike the DWT, the MODWT:
    - Is translation-invariant (Paper 3, Section II)
    - Does not downsample (returns N coefficients at each level)
    - Uses rescaled filters: h_j = h / 2^(j/2), g_j = g / 2^(j/2)

    Parameters
    ----------
    x : ndarray of shape (T,)
    level : int

    Returns
    -------
    V_J : ndarray of shape (T,)
        MODWT scaling coefficients at level J.
    W : list of ndarray
        MODWT wavelet coefficients ``[W_1, ..., W_J]``, each of shape (T,).
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    W_list = []
    v = x.copy()
    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        w_j = np.zeros(T)
        v_new = np.zeros(T)
        for t in range(T):
            idx_lag = (t - step) % T  # circular filtering
            w_j[t] = (v[t] - v[idx_lag]) / 2.0
            v_new[t] = (v[t] + v[idx_lag]) / 2.0
        W_list.append(w_j)
        v = v_new
    return v, W_list


def haar_imodwt(V_J: np.ndarray, W: List[np.ndarray]) -> np.ndarray:
    """Inverse Haar MODWT."""
    v = V_J.copy()
    T = len(v)
    for j in range(len(W), 0, -1):
        step = 2 ** (j - 1)
        w_j = W[j - 1]
        v_new = np.zeros(T)
        for t in range(T):
            idx_lag = (t - step) % T
            v_new[t] = v[t] + w_j[t]
            # Average with shifted version
        # Proper inverse using circulant structure
        v_prev = np.zeros(T)
        for t in range(T):
            idx_fwd = (t + step) % T
            v_prev[t] = v[t] + w_j[t] + v[idx_fwd] - w_j[idx_fwd]
        v = v_prev / 2.0
    return v


# ══════════════════════════════════════════════════════════════════════════════
# General MODWT using PyWavelets
# ══════════════════════════════════════════════════════════════════════════════

def modwt(
    x: np.ndarray,
    wavelet: str = "haar",
    level: int = 1,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute the MODWT using any wavelet filter supported by PyWavelets.

    This implements the non-decimated (stationary) wavelet transform.
    Common choices:
    - ``'haar'`` for structural break detection (Papers 1, 3, 5)
    - ``'sym4'`` (≈ LA(8)) for smooth decomposition (Papers 2, 4)

    Parameters
    ----------
    x : ndarray of shape (T,)
    wavelet : str
        Wavelet name (see ``pywt.wavelist()``). Use ``'sym4'`` for LA(8).
    level : int
        Decomposition level J.

    Returns
    -------
    V_J : ndarray of shape (T,)
        Scaling coefficients at the coarsest level.
    W : list of ndarray
        Wavelet detail coefficients ``[W_1, ..., W_J]``.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    w = pywt.Wavelet(wavelet)

    # Get filter coefficients and rescale for MODWT
    dec_lo = np.array(w.dec_lo) / np.sqrt(2)
    dec_hi = np.array(w.dec_hi) / np.sqrt(2)

    W_list = []
    v = x.copy()

    for j in range(1, level + 1):
        # Up-sample filters by inserting 2^(j-1) - 1 zeros
        if j > 1:
            up_factor = 2 ** (j - 1)
            lo_j = np.zeros(len(dec_lo) * up_factor - (up_factor - 1))
            hi_j = np.zeros(len(dec_hi) * up_factor - (up_factor - 1))
            lo_j[::up_factor] = dec_lo
            hi_j[::up_factor] = dec_hi
        else:
            lo_j = dec_lo
            hi_j = dec_hi

        # Circular convolution
        w_j = np.zeros(T)
        v_new = np.zeros(T)
        L = len(lo_j)
        for t in range(T):
            for l in range(L):
                idx = (t - l) % T
                w_j[t] += hi_j[l] * v[idx]
                v_new[t] += lo_j[l] * v[idx]

        W_list.append(w_j)
        v = v_new

    return v, W_list


def la8_modwt(x: np.ndarray, level: int = 4) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    MODWT with the Daubechies Least Asymmetric filter of length 8 — LA(8).

    This is the filter used in Gallegati et al. (2015) and
    Karlsson et al. (2020).  In PyWavelets, LA(8) corresponds to ``'sym4'``.

    Parameters
    ----------
    x : ndarray of shape (T,)
    level : int
        Decomposition level (default 4 for monthly data).

    Returns
    -------
    V_J : ndarray
    W : list of ndarray
    """
    return modwt(x, wavelet="sym4", level=level)


# ══════════════════════════════════════════════════════════════════════════════
# Multiresolution Analysis (MRA)
# ══════════════════════════════════════════════════════════════════════════════

def modwt_mra(
    x: np.ndarray,
    wavelet: str = "sym4",
    level: int = 4,
) -> dict:
    """
    Compute the MODWT-based Multiresolution Analysis (MRA).

    Decomposes *x* into detail components D_1, ..., D_J and a smooth
    component S_J such that ``x ≈ D_1 + D_2 + ... + D_J + S_J``.

    For annual data with J=3 (Papers 4):
    - D1: 2–4 years (short-run)
    - D2: 4–8 years (business cycle)
    - D3: 8–16 years (medium-run)
    - S3: >16 years (long-run trend)

    For monthly data with J=4 (Paper 2):
    - D1: 1–2 months
    - D2: 2–4 months
    - D3: 4–8 months
    - D4: 8–16 months

    Parameters
    ----------
    x : ndarray of shape (T,)
    wavelet : str
    level : int

    Returns
    -------
    components : dict
        Keys ``'D1', 'D2', ..., 'DJ', 'SJ'`` mapping to ndarray.
        Also includes ``'labels'`` mapping scale names to frequency bands.
    """
    x = np.asarray(x, dtype=float)

    # Use pywt's SWT (stationary wavelet transform) for MRA
    # SWT requires length to be a multiple of 2^level
    orig_len = len(x)
    pad_len = int(np.ceil(orig_len / (2 ** level))) * (2 ** level)
    if pad_len > orig_len:
        x_padded = np.pad(x, (0, pad_len - orig_len), mode="reflect")
    else:
        x_padded = x

    coeffs = pywt.swt(x_padded, wavelet, level=level, trim_approx=True)
    # coeffs = [cA_J, cD_J, cD_{J-1}, ..., cD_1]

    components = {}

    # Detail components via inverse SWT of each level independently
    for j in range(1, level + 1):
        # Zero out all coefficients except level j detail
        zeros = [np.zeros_like(c) for c in coeffs]
        idx = level - j + 1  # index into coeffs list
        zeros[idx] = coeffs[idx].copy()
        zeros[0] = np.zeros_like(coeffs[0])  # zero approx
        d_j = pywt.iswt(zeros, wavelet)[:orig_len]
        components[f"D{j}"] = d_j

    # Smooth component
    zeros = [np.zeros_like(c) for c in coeffs]
    zeros[0] = coeffs[0].copy()  # keep only approximation
    s_j = pywt.iswt(zeros, wavelet)[:orig_len]
    components[f"S{level}"] = s_j

    # Generate labels
    labels = {}
    for j in range(1, level + 1):
        lo = 2 ** j
        hi = 2 ** (j + 1)
        labels[f"D{j}"] = f"{lo}–{hi} periods"
    labels[f"S{level}"] = f">{2 ** (level + 1)} periods (trend)"
    components["labels"] = labels

    return components


# ══════════════════════════════════════════════════════════════════════════════
# Dyadic padding
# ══════════════════════════════════════════════════════════════════════════════

def pad_dyadic(x: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """
    Pad a signal to the next power-of-two length.

    Used in Paper 1 (Section 2.1) to ensure T-1 = 2^L - 1 for the
    Haar basis expansion.

    Parameters
    ----------
    x : ndarray
    mode : str
        Padding mode: ``'reflect'``, ``'constant'``, ``'edge'``.

    Returns
    -------
    x_padded : ndarray
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    next_pow2 = int(2 ** np.ceil(np.log2(T)))
    if next_pow2 == T:
        return x
    pad_width = next_pow2 - T
    return np.pad(x, (0, pad_width), mode=mode)
