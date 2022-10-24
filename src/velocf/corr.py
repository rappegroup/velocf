"""Calculate time and frequency domain auto-correlation functions."""

import logging
from typing import Any, Mapping, Optional

import numpy as np
import scipy.constants


def get_time_grid(max_lag: int, time_step: float) -> np.ndarray:
    """Return time domain axis for correlation function.

    :param max_lag: Maximum lag in correlation function, in time steps
    :param time_step: MD time step length, in fs
    :returns: time domain axis, in fs
    """
    return time_step * np.arange(max_lag)


def _lagged_correlation(velocity: np.ndarray, lag: int) -> float:
    """Correlation with a given lag averaged over starting time and atom.

    :param velocity: matrix of velocities; shape [n_time, n_atom, 3]
    :param lag: time steps to lag correlation
    :returns: average lagged auto-correlation
    """
    # pylint: disable=import-outside-toplevel
    try:
        import numexpr as ne
    except ModuleNotFoundError:
        ne = None

    t_max = len(velocity) - lag
    # Reduce over both time and atom axis
    if ne:
        _v_head = velocity[:t_max]  # noqa: F841
        _v_lag = velocity[lag:]  # noqa: F841
        corr = ne.evaluate("sum(_v_head * _v_lag)")
    else:
        corr = np.sum(velocity[:t_max] * velocity[lag:])
    # Normalize by time steps
    corr /= t_max
    return float(corr)


def get_time_correlation(
    velocity: np.ndarray,
    max_lag: Optional[int] = None,
    mass: Optional[np.ndarray] = None,
    *,
    norm_nat: bool = False,
) -> np.ndarray:
    """Calculate time domain auto-correlation function.

    :param velocity: matrix of velocities, in a.u./fs; shape [n_time, n_atom, 3]
    :param max_lag: maximum time lag to include in correlation function
    :param mass: vector of element masses; shape [n_atom,]
    :param norm_nat: Normalize the correlation by the number of atoms in the sample
    :returns: time domain auto-correlation function
    """
    logger = logging.getLogger(__name__)
    n_atom = velocity.shape[1]
    if max_lag is None:
        max_lag = len(velocity) // 2
    if max_lag == -1:
        max_lag = len(velocity) - 1
    if max_lag < 0 or max_lag >= len(velocity):
        raise ValueError(f"Invalid correlation time lag {max_lag}")
    if mass is not None:
        if len(mass) != n_atom:
            raise ValueError(
                f"Length of mass matrix ({len(mass)}) does not match number of atoms ({n_atom}"
            )

    logger.debug(
        "Calculating time correlation for %d atoms with max lag of %d steps",
        n_atom,
        max_lag,
    )

    if mass is not None:
        logger.info("Weighting trajectory by element mass")
        velocity = np.sqrt(mass.reshape((-1, 1))) * velocity

    # Allocate space for time domain correlation
    corr = np.zeros(max_lag)

    # Calculate correlation for each lag
    # Could be done in parallel
    for lag in range(max_lag):
        corr[lag] = _lagged_correlation(velocity, lag)

    if norm_nat:
        corr /= n_atom

    return corr


def get_freq_grid(max_lag: int, time_step: float) -> np.ndarray:
    """Return frequency axis for correlation function.

    :param max_lag: Maximum lag in correlation function, in time steps
    :param time_step: MD time step length, in fs
    :returns: frequency axis in THz and cm^-1
    """
    # Note: (1 fs)^-1 = 1e3 THz
    freq_thz = 1e3 * np.fft.rfftfreq(max_lag, time_step)
    # Note: 1 cm^-1 = 1e10/c THz
    freq_wavenumber = 1e10 / scipy.constants.c * freq_thz
    return np.stack([freq_thz, freq_wavenumber])


def get_freq_correlation(time_correlation: np.ndarray) -> np.ndarray:
    """Return frequency domain auto-correlation function."""
    return np.fft.rfft(time_correlation)


def welch(
    x: np.ndarray,
    nperseg: int,
    noverlap: Optional[int] = None,
    *,
    window: str = "hann",
    axis: int = 0,
) -> np.ndarray:
    """Calculate a periodogram (squared Fourier transform) by Welch's method.

    Welch's method is defined by:

    .. math::
    P^i(f) = 1 / MU | F_f[w(t) * x_i(t) |^2

    where :math:`x_i(t)` is the windowed spectrum with windows of length M,
    w(t) is the window function, and U is defined as

    .. math::
    U = 1 / M \\Sum w^2(n)

    The periodogram is the average of :math:`P^i(f)`

    :param x: Input array
    :param nperseg: Number of points in each window
    :param noverlap: Number of points overlap between segments (default: nperseg // 2)
    :param window: Name of window function to use
    :param axis: Axis of the array to use for the periodogram
    :returns: Periodogram with the same axes as input
    """
    # pylint: disable=import-outside-toplevel
    from scipy.signal import get_window

    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap < 0 or noverlap > nperseg:
        raise ValueError(f"noverlap = {noverlap}")

    window_fn = get_window(window, nperseg)
    win_norm = np.sum(window_fn**2) / nperseg
    # Update axis to account for the new segment axis which will be added
    axis = axis if axis >= 0 else len(x.shape) - axis
    axis += 1
    # Reshape window for broadcasting with segmented array
    window_fn = np.moveaxis(window_fn.reshape((1,) * len(x.shape) + (-1,)), -1, axis)

    n_hop = nperseg - noverlap
    n_block = int(np.floor((len(x) - nperseg) / n_hop)) + 1
    blocks = [x[i * n_hop : i * n_hop + nperseg] for i in range(n_block)]
    blocks = np.stack(blocks)

    block_ft = np.abs(np.fft.rfft(window_fn * blocks, axis=axis)) ** 2
    avg_ft = np.mean(block_ft, axis=0)
    return avg_ft / win_norm / nperseg


def get_freq_correlation_wk(
    velocity: np.ndarray,
    mass: Optional[np.ndarray] = None,
    *,
    welch_params: Optional[Mapping[str, Any]] = None,
    norm_nat: bool = False,
) -> np.ndarray:
    """Calculate the frequency domain autocorrelation.

    Uses the Wiener Khinchin theorem. The frequency domain velocity autocorrelation is
    the norm square of the fourier transform of the velocity.

    :param velocity: matrix of velocities, in a.u./fs; shape [n_time, n_atom, 3]
    :param mass: vector of element masses; shape [n_atom,]
    :param welch_params: Parameters to calculate power spectrum with Welch's method
    :param norm_nat: Normalize VCF by the number of atoms in the sample
    :returns: frequency domain auto-correlation function
    """
    logger = logging.getLogger(__name__)
    if welch_params is None:
        logger.debug("Calculating periodogram with full width data")
        ft_velo_sq = np.abs(np.fft.rfft(velocity, axis=0)) ** 2
        # Apply normalization from fourier transform
        ft_velo_sq /= len(velocity)
    else:
        logger.debug("Calculating periodogram using Welch's method")
        logger.debug(
            "Using parameters: %s",
            ", ".join(f"{k}={v}" for k, v in welch_params.items()),
        )
        ft_velo_sq = welch(velocity, **welch_params)
    if mass is not None:
        logger.info("Weighting trajectory by element mass")
        ft_velo_sq = mass.reshape((-1, 1)) * ft_velo_sq
    # Average over atoms and sum on cartesian directions
    freq_corr = np.sum(ft_velo_sq, axis=(1, 2))
    if norm_nat:
        freq_corr /= velocity.shape[1]
    # Note: Factor of two corrects for different trajectory lengths and brings result
    # in line with explicit correlation function method
    freq_corr /= np.sqrt(2)
    return freq_corr


def get_time_correlation_wk(freq_correlation: np.ndarray) -> np.ndarray:
    """Calculate time correlation from FFT of the frequency spectrum."""
    # This is really an inverse transform, normalize appropriately
    # Factor of two cancels factor added in the VCF
    return (
        np.sqrt(2)
        * np.fft.hfft(freq_correlation, norm="forward")[: len(freq_correlation)]
    )
