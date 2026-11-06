"""A nautilus sampler whose checkpoints are written atomically.

nautilus 1.0.6 rewrites its HDF5 checkpoint in place: ``Sampler.write`` replaces the
file, and ``Sampler.write_shell_update`` edits it. A kill during either can leave no
usable checkpoint, and the run is lost; in tests of the 21cmMC v2 plan (M0.3,
21cmfast/21CMMC#96), 3 of 13 kills did. Here both write to a temporary file, which is
flushed to disk and then renamed over the checkpoint. A rename is atomic on POSIX, so
a kill at any moment leaves either the previous checkpoint or the new one.

Remove this once nautilus writes its checkpoints atomically.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import nautilus


def _temporary(path: Path) -> Path:
    """The temporary file for ``path``, keeping its suffix, which nautilus checks."""
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _commit(tmp: Path, path: Path) -> None:
    """Flush ``tmp`` to disk, and rename it over ``path`` in one step."""
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class AtomicSampler(nautilus.Sampler):
    """A :class:`nautilus.Sampler` that writes its checkpoint atomically."""

    def write(self, filepath: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the whole run to ``filepath``, replacing it in one step."""
        path = Path(filepath)
        if path.exists() and not overwrite:
            raise RuntimeError(f"File {path} already exists.")
        tmp = _temporary(path)
        super().write(tmp, overwrite=True)
        _commit(tmp, path)

    def write_shell_update(self, filepath: str | os.PathLike, shell: int) -> None:
        """Update one shell in ``filepath``, by writing a copy and replacing it."""
        path = Path(filepath)
        tmp = _temporary(path)
        shutil.copyfile(path, tmp)
        super().write_shell_update(tmp, shell)
        _commit(tmp, path)
