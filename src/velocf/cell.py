"""Data structures for unit cell and trajectory."""

from typing import NamedTuple, Optional, Sequence, Union, overload

import numpy as np
import scipy.constants

Basis = np.ndarray
Species = Sequence[str]
Position = np.ndarray

# Allowed output coordinate types
_POSITION_COORDINATES = ("angstrom", "bohr", "crystal")


class Structure(NamedTuple):
    """Periodic crystal structure."""

    basis: Basis
    species: Optional[Species]
    positions: Position
    coord_type: str


class Trajectory(Sequence[Structure]):
    """Time series of crystal structures."""

    def __init__(
        self,
        basis: Union[Basis, Sequence[Basis]],
        species: Optional[Species],
        positions: Sequence[Position],
        coord_type: str,
        *,
        var_cell: bool = False,
    ) -> None:
        self.basis = basis
        self.species = species
        self.positions = positions
        self.coord_type = coord_type
        self.var_cell = var_cell

    @overload
    def __getitem__(self, i: int) -> Structure:
        ...

    @overload
    def __getitem__(self, s: slice) -> "Trajectory":
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[Structure, "Trajectory"]:
        basis_slice = self.basis[idx] if self.var_cell else self.basis
        if isinstance(idx, int):
            return Structure(
                basis_slice, self.species, self.positions[idx], self.coord_type
            )
        return self.__class__(
            basis_slice,
            self.species,
            self.positions[idx],
            self.coord_type,
            var_cell=self.var_cell,
        )

    def __len__(self) -> int:
        return len(self.positions)

    def __repr__(self) -> str:
        return f"Trajectory(<{len(self)} steps>)"


def convert_positions(struct: Structure, out_type: str) -> Structure:
    """Convert the coordinate type of atomic positions in a structure.

    :parameter struct: structure to be converted
    :parameter out_type: Coordinate type to return
    :returns: structure in converted units
    """
    if out_type not in _POSITION_COORDINATES:
        raise ValueError(f"Invalid position coordinate type {out_type}.")
    # Short-circuit conversion
    if struct.coord_type == out_type:
        return struct
    # Otherwise convert first to crystal
    positions = to_crystal(struct.positions, struct.basis, struct.coord_type)
    # Then to desired type
    positions = from_crystal(positions, struct.basis, out_type)
    return Structure(struct.basis, struct.species, positions, out_type)


def to_crystal(positions: Position, basis: Basis, in_type: str) -> Position:
    """Convert from arbitrary coordinates to crystal."""
    bohr_to_ang = scipy.constants.value("Bohr radius") / scipy.constants.angstrom
    if in_type == "crystal":
        return positions
    elif in_type == "bohr":
        return bohr_to_ang * positions @ np.linalg.inv(basis)
    elif in_type == "angstrom":
        return positions @ np.linalg.inv(basis)
    raise ValueError(f"Invalid position coordinate type {in_type}")


def from_crystal(positions: Position, basis: Basis, out_type: str) -> Position:
    """Convert from crystal coordinates to arbitrary coordinate units."""
    ang_to_bohr = scipy.constants.angstrom / scipy.constants.value("Bohr radius")
    # lattice vectors are rows of the basis
    if out_type == "crystal":
        return positions
    elif out_type == "bohr":
        return ang_to_bohr * positions @ basis
    elif out_type == "angstrom":
        return positions @ basis
    raise ValueError(f"Invalid position coordinate type {out_type}")


def convert_trajectory(traj: Trajectory, out_type: str) -> Trajectory:
    """Convert the coordinate type of a trajectory.

    :parameter traj: trajectory to be converted
    :parameter out_type: Coordinate type to return
    :returns: trajectory in converted units
    """
    if out_type not in _POSITION_COORDINATES:
        raise ValueError(f"Invalid position coordinate type {out_type}.")
    # Short-circuit conversion
    if traj.coord_type == out_type:
        return traj
    # Otherwise convert first to crystal
    positions = _to_crystal_traj(traj.positions, traj.basis, traj.coord_type)
    # Then to desired type
    positions = _from_crystal_traj(positions, traj.basis, out_type)
    return Trajectory(
        traj.basis, traj.species, positions, out_type, var_cell=traj.var_cell
    )


def _to_crystal_traj(
    positions: Sequence[Position], basis: Union[Basis, Sequence[Basis]], in_type: str
) -> Sequence[Position]:
    """Convert trajectory positions to crystal coordinates."""
    bohr_to_ang = scipy.constants.value("Bohr radius") / scipy.constants.angstrom
    if in_type == "crystal":
        return positions
    else:
        # This is handled correctly for both shape [3, 3] and shape [N, 3, 3]
        inv_basis = np.linalg.inv(np.array(basis))
        positions = np.array(positions)
        if in_type == "bohr":
            return bohr_to_ang * positions @ inv_basis
        elif in_type == "angstrom":
            return positions @ inv_basis
    raise ValueError(f"Invalid position coordinate type {in_type}")


def _from_crystal_traj(
    positions: Sequence[Position], basis: Union[Basis, Sequence[Basis]], out_type: str
) -> Sequence[Position]:
    """Convert trajectory positions from crystal coordinates to new units."""
    ang_to_bohr = scipy.constants.angstrom / scipy.constants.value("Bohr radius")
    # lattice vectors are rows of the basis
    if out_type == "crystal":
        return positions
    elif out_type == "bohr":
        return ang_to_bohr * np.array(positions) @ basis
    elif out_type == "angstrom":
        return np.array(positions) @ basis
    raise ValueError(f"Invalid position coordinate type {out_type}")


def min_distance(ref_pos: Position, struct: Structure) -> Sequence[float]:
    """Minimum distance to each atom in struct, in angstrom.

    Args:
        ref_pos: Position to measure against, in crystal coordinates
        struct: Structure with positions to measure
    """
    struct = convert_positions(struct, "crystal")
    # In crystal coord
    delta_pos = struct.positions - ref_pos
    delta_pos[delta_pos > 0.5] -= 1.0
    delta_pos[delta_pos < -0.5] += 1.0
    # To cartesian coordinates
    np.matmul(delta_pos, struct.basis, out=delta_pos)
    return np.linalg.norm(delta_pos, axis=1)


def normalize_positions(traj: Trajectory) -> Trajectory:
    """Normalize trajectory to remove jumps across the periodic boundary.

    :parameter traj: input trajectory. Must be in crystal coordinates
    :returns: normalized trajectory
    """
    if traj.coord_type != "crystal":
        raise ValueError("Only positions in crystal coordinates can be normalized")
    pos = np.array(traj.positions)
    for t_idx in range(1, len(pos)):
        delta_pos = pos[t_idx] - pos[t_idx - 1]
        step = np.rint(delta_pos)
        pos[t_idx] -= step
    return Trajectory(traj.basis, traj.species, pos, "crystal", var_cell=traj.var_cell)


def calc_velocity(trajectory: Trajectory, time_step: float) -> np.ndarray:
    """Compute velocities for a trajectory using finite difference.

    :param trajectory: trajectory to compute velocities for
    :param time_step: trajectory time step, in fs
    :returns: velocity matrix in units of a.u./fs; shape: [n_time, n_atom, 3]
    """
    trajectory = convert_trajectory(trajectory, "bohr")
    pos = np.array(trajectory.positions).copy()
    delta_pos = pos[2:] - pos[:-2]
    return delta_pos / (2 * time_step)
