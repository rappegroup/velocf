import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from velocf.cell import Species, Trajectory, calc_velocity, normalize_positions
from velocf.corr import (
    get_freq_correlation,
    get_freq_grid,
    get_time_correlation,
    get_time_grid,
)
from velocf.mdcar import get_fdf_species, read_md_car


def _load_trajectory(
    prefix: str, workdir: Path, fdf_path: Optional[Path] = None
) -> Trajectory:
    """Load trajectory data from several provided files."""
    if fdf_path is not None:
        with open(fdf_path, encoding="utf8") as fdf_f:
            species = get_fdf_species(fdf_f)
    else:
        species = None
    md_car_path = workdir.joinpath(f"{prefix}.MD_CAR")
    with open(md_car_path, encoding="utf8") as mdcar_f:
        traj = read_md_car(prefix, mdcar_f.readlines(), species)
    return normalize_positions(traj)


def _calculate_correlation(
    velocity: np.ndarray, lag: int, time_step: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate correlation functions for velocity data."""
    logger = logging.getLogger(__name__)

    # Calculate time correlation function
    time_corr = get_time_correlation(velocity, max_lag=lag)
    logger.info("Calculated time correlation function")
    time_grid = get_time_grid(len(time_corr), time_step)
    time_data = np.stack([time_grid, time_corr]).T

    # Calculate frequency correlation
    freq_corr = get_freq_correlation(time_corr)
    logger.info("Calculated frequency correlation function")
    freq_grid = get_freq_grid(len(time_corr), time_step)
    freq_data = np.vstack([freq_grid, np.abs(freq_corr) ** 2, np.angle(freq_corr)]).T

    return time_data, freq_data


def _write_correlation(
    time_corr: np.ndarray, freq_corr: np.ndarray, out_dir: Path, out_prefix: str
) -> None:
    logger = logging.getLogger(__name__)

    logger.info("Writing correlation functions to file")

    # Write time correlation to file
    corr_path = out_dir.joinpath(out_prefix + ".VCT")
    logger.debug("Writing time correlation to %s", str(corr_path))
    header = "time(fs)   vel. autocorr(a.u./fs)^2"
    np.savetxt(corr_path, time_corr, header=header)

    # Write velocity correlation to file
    corr_path = out_dir.joinpath(out_prefix + ".VCF")
    logger.debug("Writing frequency correlation to %s", str(corr_path))
    header = "freq(THz)   freq(cm-1)    |G(w)|^2    arg[G(w)]"
    np.savetxt(corr_path, freq_corr, header=header)


def _mask_velocity(
    velocity: np.ndarray, species: Species, target_species: str
) -> np.ndarray:
    """Select only velocities for atoms of target_species."""
    assert len(species) == velocity.shape[1]
    mask = tuple(sp == target_species for sp in species)
    return velocity[:, mask]


def parse_args(args: Sequence[str]) -> Namespace:
    parser = ArgumentParser(prog="velocf")
    parser.add_argument("prefix", help="prefix for siesta output files")
    parser.add_argument("dt", type=float, help="MD timestep in fs")
    parser.add_argument(
        "--lag", type=int, help="Max lag for time correlation. -1 for full trajectory"
    )
    parser.add_argument(
        "--skip", type=int, help="Discard initial steps from the MD trajectory"
    )
    parser.add_argument(
        "--max-len", type=int, help="Take only this many trajectory steps"
    )
    parser.add_argument(
        "--fdf", type=Path, help="Path to .fdf file to read species from"
    )
    parser.add_argument(
        "--workdir", type=Path, default=Path.cwd(), help="Directory with input files"
    )
    parser.add_argument("--out-pref", dest="out_prefix", help="Prefix for output files")
    parser.add_argument("--outdir", type=Path, help="Directory to put output files")
    parser.add_argument("-v", action="count", dest="verbosity", default=0)
    parsed = parser.parse_args(args)
    # Set default args
    if parsed.out_prefix is None:
        parsed.out_prefix = parsed.prefix
    if parsed.outdir is None:
        parsed.outdir = parsed.workdir
    return parsed


def logging_init() -> None:
    """Set up root logger."""
    logger = logging.getLogger("velocf")
    ch = logging.StreamHandler()
    fmt = logging.Formatter(
        "{asctime} [{name}:{funcName}] {levelname}: {message}",
        datefmt="%H:%M:%S",
        style="{",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def set_verbosity(verbosity: int) -> None:
    """Set verbosity level on the root logger."""
    logger = logging.getLogger("velocf")
    if verbosity > 1:
        logger.setLevel(logging.DEBUG)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


def velocf(cli_args: Sequence[str]) -> None:
    """Run CLI for main program."""
    logging_init()
    args = parse_args(cli_args)
    set_verbosity(args.verbosity)
    logger = logging.getLogger(__name__)

    # Read trajectory from file
    traj = _load_trajectory(args.prefix, args.workdir, args.fdf)
    logger.info("Loaded %d trajectory steps", len(traj))
    if args.skip is not None:
        logger.info("Discarding %d initial steps", args.skip)
        traj = traj[args.skip :]
    if args.max_len is not None:
        if args.max_len < len(traj):
            traj = traj[: args.max_len]
        logger.info("Truncating trajectory to %d steps", len(traj))

    # Calculate velocity
    velocity = calc_velocity(traj, args.dt)
    logger.debug("Calculated velocity for trajectory")

    # Do correlation for all atoms
    time_corr, freq_corr = _calculate_correlation(velocity, args.lag, args.dt)
    _write_correlation(time_corr, freq_corr, args.outdir, args.out_prefix)

    # Do correlation by species
    if traj.species is not None:
        logger.info("Calculating correlation functions by species")
        for target_spec in set(traj.species):
            logger.info("Doing correlation for %s", target_spec)
            # Filter velocity
            masked_vel = _mask_velocity(velocity, traj.species, target_spec)
            time_corr, freq_corr = _calculate_correlation(masked_vel, args.lag, args.dt)
            _write_correlation(
                time_corr, freq_corr, args.outdir, f"{args.out_prefix}.{target_spec}"
            )
