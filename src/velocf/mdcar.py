"""Parser for siesta .MD_CAR output files."""

from typing import Iterable, Optional, Tuple

import numpy as np

from velocf.cell import Basis, Position, Species, Trajectory


def get_fdf_species(fdf_lines: Iterable[str]) -> Species:
    """Extract species order from an .fdf input."""
    species = []
    fdf_iter = iter(fdf_lines)
    for line in fdf_iter:
        if "AtomicCoordinatesAndAtomicSpecies" in line:
            for coord_line in fdf_iter:
                if "%endblock" in coord_line:
                    break
                # Gather the species
                label = coord_line.split()[-1]
                species.append(label)
            else:
                raise RuntimeError("AtomicCoordinates block unfinished")
            return species
    raise RuntimeError("AtomicCoordinates block not found")


def _read_mdcar_block(
    lines: Iterable[str],
) -> Tuple[Basis, Tuple[int, ...], Position, str]:
    lines = iter(lines)
    scale = float(next(lines))
    basis = []
    for _ in range(3):
        lat_vec = next(lines).split()
        assert len(lat_vec) == 3
        basis.append(tuple(float(x) for x in lat_vec))
    basis = np.array(basis)
    basis *= scale
    species = tuple(map(int, next(lines).split()))
    coord_line = next(lines).strip().lower()
    if coord_line.startswith("d"):
        coord_type = "crystal"
    elif coord_line.startswith(("c", "k")):
        coord_type = "angstrom"
    else:
        raise ValueError(coord_line)
    pos = []
    for line in lines:
        pos_line = line.split()
        assert len(pos_line) == 3
        pos.append(tuple(map(float, pos_line)))
    assert len(pos) == sum(species)
    pos = np.array(pos)
    if coord_type == "angstrom":
        pos *= scale
    return basis, species, pos, coord_type


def read_md_car(
    tag: str, mdc_lines: Iterable[str], species: Optional[Species]
) -> Trajectory:
    """Read trajectory from an .MD_CAR file.

    Args:
        tag: calculation prefix used in the file
        mdc_lines: lines of the file
        species: species for the structure (must be read from elsewhere)
    """
    line_buf = []
    basis_buf = []
    ref_spec_order: Optional[Tuple[int, ...]] = None
    pos_buf = []
    coord_type: Optional[str] = None

    def _proc_line_buf(_lines: Iterable[str]) -> None:
        nonlocal ref_spec_order, coord_type
        if not _lines:
            return
        b, s, t, ct = _read_mdcar_block(_lines)
        if ref_spec_order is None:
            ref_spec_order = s
        if coord_type is None:
            coord_type = ct
        assert s == ref_spec_order
        assert ct == coord_type
        basis_buf.append(b)
        pos_buf.append(t)

    for line in mdc_lines:
        if f"---{tag}---" in line:
            _proc_line_buf(line_buf)
            line_buf = []
            continue
        line_buf.append(line)

    _proc_line_buf(line_buf)
    assert len(basis_buf) == len(pos_buf)
    return Trajectory(basis_buf, species, pos_buf, coord_type, var_cell=True)
