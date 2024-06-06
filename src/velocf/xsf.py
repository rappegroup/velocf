"""Read for .xsf file format."""

from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

from velocf.cell import Basis, Position, Species, Structure, Trajectory


def _get_next_line(lines: Iterator[str]) -> str:
    """Consume items from `lines` until a non-comment, non-blank line is reached."""
    while True:
        line = next(lines).strip()
        # Blank line
        if line == "":
            continue
        # Comment line
        if line[0] == "#":
            continue
        return line.strip()


def _read_axsf_header(_lines: Iterator[str]) -> int:
    header_line = _get_next_line(_lines)
    if not header_line.startswith("ANIMSTEPS"):
        raise RuntimeError(".axsf file expected")
    n_steps = int(header_line.split()[1])
    header_line = _get_next_line(_lines)
    assert header_line.startswith("CRYSTAL") or header_line.startswith("POLYMER")
    return n_steps


def _read_primvec_header(header_line: str) -> Optional[int]:
    if not header_line.startswith("PRIMVEC"):
        raise ValueError("PRIMVEC not found")
    step = None
    if len(header_line.split()) > 1:
        step = int(header_line.split()[1])
    return step


def _read_primvec(lines: Iterator[str]) -> Tuple[Basis, Optional[int]]:
    # Parse the header
    step = _read_primvec_header(_get_next_line(lines))
    # Parse the content
    basis_lines = []
    for _ in range(3):
        lat_vec = tuple(float(x) for x in _get_next_line(lines).split())
        assert len(lat_vec) == 3
        basis_lines.append(lat_vec)
    basis = np.array(basis_lines)
    return basis, step


def _read_primcoord_header(header_line: str) -> Optional[int]:
    if not header_line.startswith("PRIMCOORD"):
        raise ValueError("PRIMCOORD not found")
    step = None
    if len(header_line.split()) > 1:
        step = int(header_line.split()[1])
    return step


def _read_primcoord(lines: Iterator[str]) -> Tuple[Species, Position, Optional[int]]:
    # Parse the header
    step = _read_primcoord_header(_get_next_line(lines))
    _nat, _flag = _get_next_line(lines).split()
    assert _flag == "1"
    nat = int(_nat)

    # Parse coordinates and species
    species = []
    positions = []
    for _ in range(nat):
        line = _get_next_line(lines).split()
        species.append(line[0])
        positions.append(tuple(float(x) for x in line[1:4]))

    return tuple(species), np.array(positions), step


def read_xsf(lines: Iterable[str]) -> Structure:
    """Read a *.xsf file with one structure."""
    lines_iter = iter(lines)
    header_line = _get_next_line(lines_iter)
    assert header_line.startswith("CRYSTAL") or header_line.startswith("POLYMER")
    basis, step = _read_primvec(lines_iter)
    assert step is None
    species, pos, step = _read_primcoord(lines_iter)
    assert step is None
    return Structure(basis, species, pos, "angstrom")


def read_axsf(
    lines: Iterable[str],
) -> Trajectory:
    """Read a *.axsf file with a sequence of structures."""
    lines_iter = iter(lines)
    n_steps = _read_axsf_header(lines_iter)

    # Read the first step
    step_basis, step = _read_primvec(lines_iter)
    animate_cell = step is not None
    species, step_pos, step = _read_primcoord(lines_iter)
    assert step == 1

    # Initialize accumulators
    if animate_cell:
        basis = [step_basis]
    else:
        basis = step_basis
    positions = [step_pos]

    # Read the remaining steps
    for i in range(2, n_steps + 1):
        if animate_cell:
            step_basis, step = _read_primvec(lines_iter)
            assert step == i
            basis.append(step_basis)
        step_spec, step_pos, step = _read_primcoord(lines_iter)
        assert step == i
        assert species == step_spec
        positions.append(step_pos)

    return Trajectory(
        basis, species, np.stack(positions), "angstrom", var_cell=animate_cell
    )


_T = TypeVar("_T")


def columns(
    matrix: Sequence[Sequence[_T]],
    *,
    min_space: int = 3,
    left_pad: Optional[int] = None,
    convert_fn: Callable[[_T], str] = str,
    sep: str = " ",
    align: str = "left",
) -> List[str]:
    """Format `matrix` into a string with aligned columns.
    Args:
        matrix: 2d array of values to print; may not be ragged
        min_space: Minimum number of separators between columns
        left_pad: Minimum number of leading separators (default `min_space`)
        convert_fn: Conversion function applied to each element of `matrix` before
            aligning (default :py:func:`str`)
        sep: String used as a separator between columns (default space)
        align: Direction to align elements if elements in a column are not the same
            width. Either 'left' or 'right' (default 'left')
    Returns:
        Formatted matrix as a list of formatted lines
    """
    if align not in ("left", "right"):
        raise ValueError(f"Incorrect alignment {align!r}")
    if any(len(line) != len(matrix[0]) for line in matrix):
        raise ValueError("Matrix may not be ragged")
    if left_pad is None:
        left_pad = min_space

    field_strings = [list(map(convert_fn, line)) for line in matrix]
    col_widths = [max(map(len, col)) for col in zip(*field_strings)]

    complete_lines = []
    for row in field_strings:
        line = []
        for width, field in zip(col_widths, row):
            if align == "left":
                line.append(field.ljust(width, sep))
            else:
                line.append(field.rjust(width, sep))
        complete_lines.append(
            left_pad * sep + (min_space * sep).join(line).rstrip() + "\n"
        )
    return complete_lines


def _gen_primvec(basis: Basis, *, step: int = None) -> Iterator[str]:
    if step is None:
        yield "PRIMVEC\n"
    else:
        yield f"PRIMVEC {step:d}\n"
    yield from columns(basis, convert_fn="{:.8f}".format)


def _gen_primcoord(
    species: Species, positions: Position, *, step: int = None
) -> Iterator[str]:
    if step is None:
        yield "PRIMCOORD\n"
    else:
        yield f"PRIMCOORD {step:d}\n"
    yield f"{len(species)} 1\n"
    lines = [
        (spec,) + tuple(map("{:.8f}".format, pos))
        for spec, pos in zip(species, positions)
    ]
    yield from columns(lines, left_pad=0)


def gen_xsf(struct: Structure) -> Iterator[str]:
    """Generate lines for a single structure .xsf file."""
    # pylint: disable=import-outside-toplevel
    from velocf.cell import convert_positions

    struct = convert_positions(struct, "angstrom")
    yield "CRYSTAL\n"
    yield from _gen_primvec(struct.basis)
    yield from _gen_primcoord(struct.species, struct.positions)


def gen_axsf(trajectory: Trajectory) -> Iterator[str]:
    """Generate lines for an animated .axsf trajectory."""
    # pylint: disable=import-outside-toplevel
    from velocf.cell import convert_trajectory

    trajectory = convert_trajectory(trajectory, "angstrom")
    yield f"ANIMSTEPS {len(trajectory)}\n"
    yield "CRYSTAL\n"

    if not trajectory.var_cell:
        yield from _gen_primvec(trajectory.basis)

    for i, struct in enumerate(trajectory):
        if trajectory.var_cell:
            yield from _gen_primvec(struct.basis, step=i)
        yield from _gen_primcoord(struct.species, struct.positions, step=i)
