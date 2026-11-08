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

import contextlib
import os
import shutil
from collections.abc import Callable, Iterator
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


def _write(
    sampler: nautilus.Sampler,
    filepath: str | os.PathLike,
    overwrite: bool,
    original: Callable[..., None],
) -> None:
    """Write the whole run with ``original``, replacing ``filepath`` in one step."""
    path = Path(filepath)
    if path.exists() and not overwrite:
        raise RuntimeError(f"File {path} already exists.")
    tmp = _temporary(path)
    original(sampler, tmp, overwrite=True)
    _commit(tmp, path)


def _write_shell_update(
    sampler: nautilus.Sampler,
    filepath: str | os.PathLike,
    shell: int,
    original: Callable[..., None],
) -> None:
    """Update one shell with ``original``, by writing a copy and replacing it."""
    path = Path(filepath)
    tmp = _temporary(path)
    shutil.copyfile(path, tmp)
    original(sampler, tmp, shell)
    _commit(tmp, path)


class AtomicSampler(nautilus.Sampler):
    """A :class:`nautilus.Sampler` that writes its checkpoint atomically."""

    def write(self, filepath: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the whole run to ``filepath``, replacing it in one step."""
        _write(self, filepath, overwrite, nautilus.Sampler.write)

    def write_shell_update(self, filepath: str | os.PathLike, shell: int) -> None:
        """Update one shell in ``filepath``, by writing a copy and replacing it."""
        _write_shell_update(self, filepath, shell, nautilus.Sampler.write_shell_update)


@contextlib.contextmanager
def atomic_checkpoints() -> Iterator[None]:
    """Make every :class:`nautilus.Sampler` write its checkpoint atomically, for now.

    This is for samplers that code outside yabf creates, such as CosmoSIS's: while
    the context is open, ``nautilus.Sampler`` writes as :class:`AtomicSampler` does.
    The original methods are restored when it closes.
    """
    write, shell_update = nautilus.Sampler.write, nautilus.Sampler.write_shell_update

    def atomic_write(
        self: nautilus.Sampler, filepath: str | os.PathLike, overwrite: bool = False
    ) -> None:
        """Write the whole run, replacing ``filepath`` in one step."""
        _write(self, filepath, overwrite, write)

    def atomic_shell_update(
        self: nautilus.Sampler, filepath: str | os.PathLike, shell: int
    ) -> None:
        """Update one shell, by writing a copy and replacing ``filepath``."""
        _write_shell_update(self, filepath, shell, shell_update)

    nautilus.Sampler.write = atomic_write
    nautilus.Sampler.write_shell_update = atomic_shell_update
    try:
        yield
    finally:
        nautilus.Sampler.write = write
        nautilus.Sampler.write_shell_update = shell_update
