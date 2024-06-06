# velocf

[![DOI](https://zenodo.org/badge/507150770.svg)](https://zenodo.org/doi/10.5281/zenodo.11497902)

> Calculate velocity auto-correlation functions from SIESTA trajectories.

## Description

This program calculates the vibrational density of states from molecular dynamics trajectories.
The method using the velocity autocorrelation function directly is based on the implementation
by [Andrei Postnikov](https://www.home.uni-osnabrueck.de/apostnik/download.html).


## Installation

The package can be installed from a package with `pip install <package>`.
The package fileman either be downloaded from the repository or built with `python -m bulid .`.


## Usage

It is preferred to install the package and use the installed `velocf` script.

Complete usage is available by calling `velocf --help`. The program is called as
`velocf [options] <trajectory file> <time step>`.
The time step of the MD trajectory should be provided in fs. The trajectory may be either
an `MDCAR` file or a `.xsf` file.
Note that since `MDCAR` files do not record species labels, the input `.fdf` should be
provided using the `--fdf` option if mass-weighting is desired.

The output time-domain correlation functions are written to `prefix.VCT` and frequency-domain
functions are written to `prefix.VCF`.
The prefix is either specified  with `--out-pref` or inferred from input.
Additional species-resolved functions are written if requested.

By default, time-domain correlation functions are calculated directly, while frequency-domain
correlations can be calculated first if `--wk-corr` is specified (see below). It is recommended
to use the WK method, as it is less sensitive to the finite sampling window.
The WK method can also be applied using a windowed Fourier transform, which can

at the expense of frequency resolution.
The window length must be given as `--welch N` for a window of `N` samples, or `--welch N fs`
for a window in units of time (ps and fs are understood).


## Details

In the normal mode of operation, the time correlation function is first calculated as

$$C(t) = \sum_i m_i \left\langle v_i(t+\tau) \cdot v_i(\tau) \right\rangle_{\tau}$$

where the average is performed over all starting times $\tau$ and the time lag $t$ is truncated
based on user input. The frequency correlation $C(\omega)$ is then obtained by Fourier
transforming $C(t)$. Alternatively, by the Wiener-Khinchin theorem, we can obtain the frequency
correlation by

$$C(\omega) = \sum_i m_i | \mathcal{F}[ v_i(t) ]|^2$$

The time correlation is then obtained by Fourier transforming $C(\omega)$.
Velocities are calculated using a central finite difference.

This sum naturally decomposes the spectrum between different atoms.
By default, partial spectra for different atom species will be computed.
The spectrum can also be limited to a portion of the entire trajectory by specifying
a subset of atoms with the `--select` flag. Only the atoms in the provided structure will
be included in the correlation function.
