import argparse
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import mendeleev
import numpy as np

from velocf import __version__
from velocf.cell import (
    Species,
    Trajectory,
    calc_velocity,
    convert_positions,
    min_distance,
    normalize_positions,
)
from velocf.corr import (
    get_freq_correlation,
    get_freq_correlation_wk,
    get_freq_grid,
    get_time_correlation,
    get_time_correlation_wk,
    get_time_grid,
)
from velocf.mdcar import get_fdf_species, read_md_car
from velocf.xsf import read_axsf, read_xsf


def _parse_input_path(
    prefix: str, workdir: Optional[Path]
) -> Tuple[str, Path, Path, bool]:
    """Returns prefix, in_path, workdir, is_axsf."""
    is_axsf = False
    if prefix.endswith(".axsf"):
        in_path = Path(prefix)
        is_axsf = True
    elif prefix.endswith(".MD_CAR"):
        in_path = Path(prefix)
    else:
        # Prefix only for siesta output
        in_path = Path(f"{prefix}.MD_CAR")

    prefix = in_path.stem
    if workdir is None:
        workdir = in_path.parent.resolve()
    return prefix, in_path, workdir, is_axsf


def _load_mdc_trajectory(
    prefix: str, mdc_path: Path, fdf_path: Optional[Path] = None
) -> Trajectory:
    if fdf_path is not None:
        with open(fdf_path, encoding="utf8") as fdf_f:
            species = get_fdf_species(fdf_f)
    else:
        species = None
    with open(mdc_path, encoding="utf8") as mdcar_f:
        return read_md_car(prefix, mdcar_f.readlines(), species)


def _load_axsf_trajectory(axsf_path: Path) -> Trajectory:
    # pylint: disable=import-outside-toplevel
    from velocf.cell import convert_trajectory

    with open(axsf_path, encoding="utf8") as axsf_f:
        traj = read_axsf(axsf_f)
    return convert_trajectory(traj, "crystal")


def _load_trajectory(
    prefix: str,
    in_path: Path,
    is_axsf: bool,
    fdf_path: Optional[Path] = None,
) -> Trajectory:
    """Load trajectory data from several provided files."""

    if is_axsf:
        traj = _load_axsf_trajectory(in_path)
    else:
        traj = _load_mdc_trajectory(prefix, in_path, fdf_path)
    return normalize_positions(traj)


def _restrict_trajectory(traj: Trajectory, select_path: Path) -> Trajectory:
    with open(select_path, encoding="utf8") as f_select:
        select_struct = read_xsf(f_select)

    select_struct = convert_positions(select_struct, "crystal")
    init_struct = convert_positions(traj[0], "crystal")
    select_idx: List[int] = []

    for i, pos in enumerate(select_struct.positions):
        idx_min = int(np.argmin(min_distance(pos, init_struct)))
        if init_struct.species is not None and select_struct.species is not None:
            assert select_struct.species[i] == init_struct.species[idx_min]
            select_idx.append(idx_min)

    select_idx = sorted(select_idx)
    # Assert no duplicates
    assert len(np.unique(select_idx)) == len(select_idx)

    return Trajectory(
        traj.basis,
        tuple(traj.species[i] for i in select_idx),
        traj.positions[:, select_idx],
        traj.coord_type,
        var_cell=traj.var_cell,
    )


def _calculate_correlation(
    velocity: np.ndarray,
    lag: int,
    time_step: float,
    species: Optional[Species] = None,
    use_wk=False,
    welch_params=None,
    norm_nat: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate correlation functions for velocity data."""
    logger = logging.getLogger(__name__)

    # Calculate element masses
    if species is not None:
        mass = np.array(tuple(map(lambda el: mendeleev.element(el).mass, species)))
    else:
        mass = None

    if not use_wk:
        # Calculate time correlation function
        time_corr = get_time_correlation(
            velocity, max_lag=lag, mass=mass, norm_nat=norm_nat
        )
        logger.info("Calculated time correlation function")
        # Calculate frequency correlation
        freq_grid = get_freq_grid(len(time_corr), time_step)
        freq_corr = get_freq_correlation(time_corr)
        logger.info("Calculated frequency correlation function")
    else:
        logger.info("Calculating correlation using Wiener Khinchin theorem")
        if not welch_params:
            freq_grid = get_freq_grid(len(velocity), time_step)
        else:
            freq_grid = get_freq_grid(welch_params["nperseg"], time_step)
        freq_corr = get_freq_correlation_wk(
            velocity, mass=mass, welch_params=welch_params, norm_nat=norm_nat
        )
        logger.info("Calculated frequency correlation function")
        time_corr = get_time_correlation_wk(freq_corr)
        logger.info("Calculated time correlation function")

    # Assemble data for output
    time_grid = get_time_grid(len(time_corr), time_step)
    time_data = np.stack([time_grid, time_corr]).T
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
    parser.add_argument("input", help="Trajectory input file")
    parser.add_argument("dt", type=float, help="MD timestep in fs")
    parser.add_argument("--prefix", help="prefix for siesta output files")

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
        "--wk-corr",
        action="store_true",
        help="Calculate correlation using Wiener Khinchin theorem",
    )
    parser.add_argument("--welch", metavar="WINDOW")
    parser.add_argument(
        "--norm-nat", action="store_true", help="Normalize by number of atoms in cell"
    )

    parser.add_argument(
        "--fdf", type=Path, help="Path to .fdf file to read species from"
    )
    parser.add_argument(
        "--select",
        type=Path,
        metavar="XSF",
        help="Use only atoms in the structure for correlation",
    )
    parser.add_argument(
        "--mass-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mass-weight the total correlation function",
    )
    parser.add_argument(
        "--mask-species",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calculate species-resolved correlation functions",
    )

    parser.add_argument("--workdir", type=Path, help="Directory with input files")
    parser.add_argument("--out-pref", dest="out_prefix", help="Prefix for output files")
    parser.add_argument("--outdir", type=Path, help="Directory to put output files")
    parser.add_argument("-v", action="count", dest="verbosity", default=0)
    parser.add_argument(
        "--version", "-V", action="version", version=f"%(prog)s {__version__}"
    )

    parsed = parser.parse_args(args)

    # Parse the input paths
    prefix, in_path, workdir, parsed.is_axsf = _parse_input_path(
        parsed.input, parsed.workdir
    )

    del parsed.input
    del parsed.workdir
    if parsed.prefix is None:
        parsed.prefix = prefix

    def _resolve_path(path: Optional[Path]) -> Optional[Path]:
        if path is None:
            return None
        if not path.is_absolute() and not path.exists():
            # Relative path that is NOT relative to cwd
            path = workdir / path
        # Ensure path exists
        if not path.exists():
            raise ValueError(f"{path} does not exist")
        return path

    parsed.in_path = _resolve_path(in_path)
    parsed.fdf = _resolve_path(parsed.fdf)
    parsed.select = _resolve_path(parsed.select)

    # Set default args
    if parsed.out_prefix is None:
        parsed.out_prefix = parsed.prefix
    if parsed.outdir is None:
        parsed.outdir = workdir
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
    traj = _load_trajectory(args.prefix, args.in_path, args.is_axsf, fdf_path=args.fdf)
    logger.info("Loaded %d trajectory steps", len(traj))

    # Pick only certain atoms
    if args.select is not None:
        logger.info("Restricting atoms based on provided structure")
        traj = _restrict_trajectory(traj, args.select)
        logger.debug("New trajectory contains %d atoms", len(traj.species))

    # Trim the trajectory length
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
    species = traj.species if args.mass_weight else None

    def _welch_params() -> Mapping[str, Any]:
        import re

        if match := re.match(r"^(\d+(?:\.\d+)?)([a-zA-Z]+)?", args.welch):
            if match.group(2) is None:
                _nperseg = int(match.group(1))
            else:
                _val = float(match.group(1))
                _unit = match.group(2)
                if _unit == "fs":
                    pass
                elif _unit == "ps":
                    _val *= 1e3
                else:
                    raise ValueError(f"Unknown unit {_unit}")
                _nperseg = int(np.floor(_val))
        else:
            raise ValueError(f"Could not parse {args.welch}")
        return {"nperseg": _nperseg}

    welch = None if args.welch is None else _welch_params()
    time_corr, freq_corr = _calculate_correlation(
        velocity,
        args.lag,
        args.dt,
        species=species,
        use_wk=args.wk_corr,
        welch_params=welch,
        norm_nat=args.norm_nat,
    )

    _write_correlation(time_corr, freq_corr, args.outdir, args.out_prefix)

    # Do correlation by species
    if traj.species is not None and args.mask_species:
        logger.info("Calculating correlation functions by species")
        for target_spec in set(traj.species):
            logger.info("Doing correlation for %s", target_spec)
            # Filter velocity
            masked_vel = _mask_velocity(velocity, traj.species, target_spec)
            masked_species = (
                masked_vel.shape[1] * (target_spec,) if species is not None else None
            )
            time_corr, freq_corr = _calculate_correlation(
                masked_vel,
                args.lag,
                args.dt,
                species=masked_species,
                use_wk=args.wk_corr,
                welch_params=welch,
                norm_nat=args.norm_nat,
            )
            _write_correlation(
                time_corr, freq_corr, args.outdir, f"{args.out_prefix}.{target_spec}"
            )
