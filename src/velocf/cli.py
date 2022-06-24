import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Sequence

from velocf.cell import Trajectory, calc_velocity, normalize_positions
from velocf.corr import (
    get_freq_correlation,
    get_time_correlation,
    write_freq_correlation,
    write_time_correlation,
)
from velocf.mdcar import get_fdf_species, read_md_car

# todo: mask time corr
# TODO: misc logging


def load_trajectory(
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


def parse_args(args: Sequence[str]) -> Namespace:
    parser = ArgumentParser(prog="velocf")
    parser.add_argument("prefix")
    parser.add_argument("dt", type=float, help="MD timestep in fs")
    parser.add_argument(
        "--lag", type=int, help="Max lag for time correlation. -1 for full trajectory"
    )
    parser.add_argument("--skip", type=int, help="Discard initial steps from the MD trajectory")
    parser.add_argument("--fdf", type=Path)
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Directory with input files")
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
    logging_init()
    args = parse_args(cli_args)
    set_verbosity(args.verbosity)
    logger = logging.getLogger(__name__)

    # Read trajectory from file
    traj = load_trajectory(args.prefix, args.workdir, args.fdf)
    logger.info("Loaded %d trajectory steps", len(traj))
    if args.skip is not None:
        logger.info("Discarding %d initial steps", args.skip)
        traj = traj[args.skip:]
    # Calculate velocity
    velocity = calc_velocity(traj, args.dt)
    logger.debug("Calculated velocity for trajectory")

    # Calculate time correlation function
    time_corr = get_time_correlation(velocity, max_lag=args.lag)
    logger.info("Calculated time correlation function")
    # Calculate frequency correlation
    freq_corr = get_freq_correlation(time_corr)
    logger.info("Calculated frequency correlation function")

    logger.info("Writing correlation functions to file")
    # Write time correlation to file
    corr_path = args.outdir.joinpath(args.out_prefix + ".VCT")
    logger.debug("Writing time correlation to %s", str(corr_path))
    with open(corr_path, "w", encoding="utf8") as f_out:
        write_time_correlation(f_out, time_corr, args.dt)
    # Write velocity correlation to file
    corr_path = args.outdir.joinpath(args.out_prefix + ".VCF")
    logger.debug("Writing frequency correlation to %s", str(corr_path))
    with open(corr_path, "w", encoding="utf8") as f_out:
        write_freq_correlation(f_out, freq_corr, len(time_corr), args.dt)
