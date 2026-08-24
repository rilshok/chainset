"""Storage backend for datasets."""

from typing import BinaryIO

from iokit import Storage

StorageBackend = Storage[BinaryIO]


class DatasetStorage:
    """Store dataset items in a binary storage backend."""

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize the dataset storage.

        Args:
            backend: Binary storage used to persist the dataset.

        """
        self._backend = backend
