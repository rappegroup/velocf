"""Read for .xsf file format."""

from typing import Iterable, Iterator, Optional, Tuple

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

    return Trajectory(basis, species, positions, "angstrom", var_cell=animate_cell)
