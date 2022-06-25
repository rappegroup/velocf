"""Calculate time and frequency domain auto-correlation functions."""

import logging
from typing import Optional

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
    t_max = len(velocity) - lag
    n_at = velocity.shape[1]
    # Reduce over both time and atom axis
    corr = np.sum(velocity[:t_max] * velocity[lag:])
    # Normalize by time steps and atoms
    corr /= n_at * t_max
    return float(corr)


def get_time_correlation(
    velocity: np.ndarray, max_lag: Optional[int] = None
) -> np.ndarray:
    """Calculate time domain auto-correlation function.

    :param velocity: matrix of velocities, in a.u./fs; shape [n_time, n_atom, 3]
    :param max_lag: maximum time lag to include in correlation function
    :returns: time domain auto-correlation function
    """
    logger = logging.getLogger(__name__)
    if max_lag is None:
        max_lag = len(velocity) // 2
    if max_lag == -1:
        max_lag = len(velocity) - 1
    if max_lag < 0 or max_lag >= len(velocity):
        raise ValueError(f"Invalid correlation time lag {max_lag}")

    logger.debug(
        "Calculating time correlation for %d atoms with max lag of %d steps",
        velocity.shape[1],
        max_lag,
    )

    # Allocate space for time domain correlation
    corr = np.zeros(max_lag)

    # Calculate correlation for each lag
    # Could be done in parallel
    for lag in range(max_lag):
        corr[lag] = _lagged_correlation(velocity, lag)

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
