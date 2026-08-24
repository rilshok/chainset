"""Chain of dataset processing steps."""

from .provider import Provider
from .storage import DatasetStorage, StorageBackend


class Chain:
    """Process a dataset by combining a storage backend with a data provider."""

    def __init__(self, storage: StorageBackend, provider: Provider) -> None:
        """Initialize the chain.

        Args:
            storage: Backend used to persist the dataset.
            provider: Source of the dataset items.

        """
        self._storage = DatasetStorage(storage)
        self._provider = provider
