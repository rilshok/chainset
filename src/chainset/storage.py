"""Storage backend for datasets."""

from typing import Any, BinaryIO, TypeVar

from iokit import BufferedState, Storage

from .record import Record

StorageBackend = Storage[BinaryIO]

R = TypeVar("R", bound=Record[Any])


class DatasetStorage:
    """Store dataset items in a binary storage backend."""

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize the dataset storage.

        Args:
            backend: Binary storage used to persist the dataset.

        """
        self._backend = backend

    def _key(self, uid: str, origin: str, path: str) -> str:
        return f"{uid}/{origin}/{path}"

    def push(self, uid: str, origin: str, record: Record[Any], *, force: bool = False) -> None:
        """Write a record to the backend.

        Args:
            uid: Identifier of the dataset item.
            origin: Namespace the record belongs to.
            record: Record whose state is persisted.
            force: Whether to overwrite an existing entry.

        """
        state = record.state
        key = self._key(uid, origin, state.path)
        self._backend.push(key, state.buffer, force=force)

    def pull(self, uid: str, origin: str, path: str, record_of: type[R]) -> R:
        """Read a record from the backend.

        Args:
            uid: Identifier of the dataset item.
            origin: Namespace the record belongs to.
            path: Path of the stored state.
            record_of: Record type used to rebuild the value.

        Returns:
            The record reconstructed from the stored state.

        """
        key = self._key(uid, origin, path)
        buffer = self._backend.pull(key)
        state: BufferedState[Any] = BufferedState(buffer, path=path)
        return record_of.from_state(state)

    def remove(self, uid: str, origin: str, path: str) -> None:
        """Delete a record from the backend.

        Args:
            uid: Identifier of the dataset item.
            origin: Namespace the record belongs to.
            path: Path of the stored state.

        """
        key = self._key(uid, origin, path)
        return self._backend.remove(key)

    def exists(self, uid: str, origin: str, path: str) -> bool:
        """Check whether a record is present in the backend.

        Args:
            uid: Identifier of the dataset item.
            origin: Namespace the record belongs to.
            path: Path of the stored state.

        Returns:
            ``True`` if the record exists, ``False`` otherwise.

        """
        key = self._key(uid, origin, path)
        return self._backend.exists(key)

    def size(self, uid: str, origin: str, path: str) -> int:
        """Report the stored size of a record.

        Args:
            uid: Identifier of the dataset item.
            origin: Namespace the record belongs to.
            path: Path of the stored state.

        Returns:
            The size of the stored record in bytes.

        """
        key = self._key(uid, origin, path)
        return self._backend.size(key)
