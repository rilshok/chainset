"""Storage backend for datasets."""

__all__ = [
    "DatasetStorage",
    "LikeStorage",
    "StorageBackend",
    "dataset_storage",
]

from pathlib import Path
from typing import Any, BinaryIO, TypeVar

from iokit import (
    BufferedState,
    FormatState,
    MemoryStorage,
    Storage,
    StreamLocalStorage,
    StreamMemoryStorage,
)

StorageBackend = Storage[BinaryIO]

S = TypeVar("S", bound=FormatState[Any])


class DatasetStorage:
    """Store dataset items in a binary storage backend."""

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize the dataset storage.

        Args:
            backend: Binary storage used to persist the dataset.

        """
        self._backend = backend

    def _key(self, uid: str, origin: str, key: str) -> str:
        return f"{uid}/{origin}/{key}"

    def push(self, uid: str, origin: str, state: FormatState[Any], *, force: bool = False) -> None:
        """Write a state to the storage."""
        key = self._key(uid, origin, state.path)
        self._backend.push(key, state.buffer, force=force)

    def pull(self, uid: str, origin: str, key: str, state_t: type[S]) -> S:
        """Read a state from the storage."""
        key = self._key(uid, origin, key)
        buffer = self._backend.pull(key)
        state: BufferedState[Any] = BufferedState(buffer, path=key)
        return state_t.from_state(state)

    def remove(self, uid: str, origin: str, key: str) -> None:
        """Delete a state from the storage."""
        key = self._key(uid, origin, key)
        return self._backend.remove(key)

    def exists(self, uid: str, origin: str, key: str) -> bool:
        """Check whether a state is present in the storage."""
        key = self._key(uid, origin, key)
        return self._backend.exists(key)

    def size(self, uid: str, origin: str, key: str) -> int:
        """Report the stored size of a state."""
        key = self._key(uid, origin, key)
        return self._backend.size(key)


LikeStorage = None | dict[str, bytes] | MemoryStorage | str | Path | StorageBackend | DatasetStorage


def dataset_storage(storage: LikeStorage) -> DatasetStorage:
    """Coerce a storage-like value into a dataset storage.

    Args:
        storage: Existing storage, a backend, a mapping or path to build one from,
            or `None` for an in-memory storage.

    Returns:
        The dataset storage backed by `storage`.

    """
    if isinstance(storage, DatasetStorage):
        return storage
    if storage is None:
        storage = {}
    if isinstance(storage, dict):
        storage = MemoryStorage(storage)
    if isinstance(storage, MemoryStorage):
        storage = StreamMemoryStorage(storage)
    if isinstance(storage, str):
        storage = Path(storage).resolve()
    if isinstance(storage, Path):
        storage = StreamLocalStorage(storage)
    return DatasetStorage(storage)
