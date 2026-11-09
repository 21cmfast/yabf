"""Module to keep track of MPI things.

Most functions copied from cobaya.mpi.

MPI is initialised (by importing mpi4py) the first time one of this module's
attributes is used, not when yabf is imported: initialising it is slow, and a process
that has initialised MPI cannot launch MPI jobs of its own.
"""

from __future__ import annotations

import importlib
from typing import Any

_ATTRIBUTES = (
    "MPI",
    "mpi_comm",
    "mpi_size",
    "mpi_rank",
    "am_single_or_primary_process",
    "more_than_one_process",
)


def _initialise() -> dict[str, Any]:
    """Import mpi4py, if it is installed, and set this module's attributes."""
    try:
        mpi = importlib.import_module("mpi4py.MPI")
    except ImportError:
        mpi = None
    comm = getattr(mpi, "COMM_WORLD", None)
    size = getattr(comm, "Get_size", lambda: 0)()
    rank = getattr(comm, "Get_rank", lambda: None)()
    values = {
        "MPI": mpi,
        "mpi_comm": comm,
        "mpi_size": size,
        "mpi_rank": rank,
        "am_single_or_primary_process": not bool(rank),
        "more_than_one_process": size > 1,
    }
    globals().update(values)
    return values


def __getattr__(name: str) -> object:
    """Initialise MPI when one of its attributes is first used."""
    if name in _ATTRIBUTES:
        return _initialise()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def sync_processes() -> None:
    """Sync MPI processes."""
    values = _initialise() if "mpi_comm" not in globals() else globals()
    if values["more_than_one_process"]:
        values["mpi_comm"].barrier()
