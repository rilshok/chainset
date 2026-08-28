"""Chain of dataset processing steps."""

from collections.abc import Callable
from types import FunctionType
from typing import Generic, TypeVar, overload

from iokit import FormatState, State
from iokit.utils.time import Timestamp
from typing_extensions import Self

from .storage import DatasetStorage, StorageBackend


class Chain:
    """Process a dataset by combining a storage backend with a data provider."""

    def __init__(self, storage: StorageBackend, origin: str) -> None:
        """Initialize the chain.

        Args:
            storage: Backend used to persist the dataset.
            origin: Origin of chain.

        """
        self._storage = DatasetStorage(storage)
        self._origin = origin

    @property
    def storage(self) -> DatasetStorage:
        """Backend used to persist the dataset."""
        return self._storage

    @property
    def origin(self) -> str:
        """Origin of chain."""
        return self._origin


T = TypeVar("T", bound=object)


class StateCodec(Generic[T]):
    """Convert records to and from states of a single storage format."""

    def __init__(self, state_t: type[FormatState[T]], **config: object) -> None:
        """Initialize the codec.

        Args:
            state_t: State type used to encode and decode the records.
            config: Options forwarded to `state_t` on encoding and decoding.

        """
        self.state_t = state_t
        self.config = config

    def encode(self, data: T, key: str) -> FormatState[T]:
        """Encode `data` into a state stored under `key`.

        Args:
            data: Record produced by a chain method.
            key: Name of the record within the storage.

        Returns:
            The state holding the encoded `data`.

        """
        return self.state_t(
            data,
            stem=None,
            path=key + self.state_t.extension(),
            timestamp=Timestamp.now(),
            **self.config,
        )

    def decode(self, state: State[T]) -> T:
        """Decode a record from `state`.

        Args:
            state: State previously produced by `encode`.

        Returns:
            The record held by `state`.

        """
        return state.load(**self.config)


C = TypeVar("C", bound=Chain)


class BoundStoredMethod(Generic[C, T]):
    """Stored method bound to a chain instance."""

    def __init__(
        self,
        obj: C,
        func: Callable[[C, str], T],
        codec: StateCodec[T],
    ) -> None:
        """Initialize the bound stored method.

        Args:
            obj: Chain instance owning the records.
            func: Chain method computing a record from an item identifier.
            codec: Codec used to encode and decode the records of `func`.

        """
        self._obj = obj
        self._func = func
        self._codec = codec
        self._origin = obj.origin
        self._key = self._func.__name__

    def __call__(self, uid: str) -> T:
        """Return the record of the item `uid`, computing and storing it if needed.

        Args:
            uid: Identifier of the dataset item.

        Returns:
            The stored record of the item `uid`.

        """
        if self.exists(uid):
            return self.pull(uid)
        record = self._func(self._obj, uid)
        self.push(uid, record=record, force=False)
        return record

    def pull(self, uid: str) -> T:
        """Read the record of the item `uid` from the storage.

        Args:
            uid: Identifier of the dataset item.

        Returns:
            The stored record of the item `uid`.

        """
        state = self._obj.storage.pull(
            uid=uid,
            origin=self._origin,
            key=self._key,
            state_t=self._codec.state_t,
        )
        return self._codec.decode(state)

    def push(self, uid: str, record: T, *, force: bool = False) -> None:
        """Write the `record` of the item `uid` to the storage.

        Args:
            uid: Identifier of the dataset item.
            record: Record to store.
            force: Whether to overwrite an already stored record.

        """
        state = self._codec.encode(record, key=self._key)
        self._obj.storage.push(uid=uid, origin=self._origin, state=state, force=force)

    def remove(self, uid: str) -> None:
        """Delete the record of the item `uid` from the storage.

        Args:
            uid: Identifier of the dataset item.

        """
        self._obj.storage.remove(uid=uid, origin=self._origin, key=self._key)

    def exists(self, uid: str) -> bool:
        """Check whether the record of the item `uid` is stored.

        Args:
            uid: Identifier of the dataset item.

        Returns:
            True if the record is present in the storage.

        """
        return self._obj.storage.exists(uid, origin=self._origin, key=self._key)


class StoredMethod(Generic[C, T]):
    """Descriptor persisting the results of a chain method in the chain storage."""

    def __init__(
        self,
        func: Callable[[C, str], T],
        codec: StateCodec[T],
    ) -> None:
        """Initialize the stored method.

        Args:
            func: Chain method computing a record from an item identifier.
            codec: Codec used to encode and decode the records of `func`.

        Raises:
            TypeError: If `func` is not a plain function.

        """
        if not isinstance(func, FunctionType):
            msg = f"Stored methods must be plain functions, got {type(func).__name__!r}"
            raise TypeError(msg)
        self._func = func
        self._codec = codec

    @overload
    def __get__(self, obj: None, objtype: type[C] | None = None) -> Self: ...

    @overload
    def __get__(self, obj: C, objtype: type[C] | None = None) -> BoundStoredMethod[C, T]: ...

    def __get__(
        self,
        obj: C | None,
        objtype: type[C] | None = None,
    ) -> Self | BoundStoredMethod[C, T]:
        """Bind the stored method to a chain instance.

        Args:
            obj: Chain instance the method is accessed on, or None on class access.
            objtype: Chain class the method is accessed through.

        Returns:
            The descriptor itself on class access, otherwise the bound method.

        """
        if obj is None:
            return self
        return BoundStoredMethod(obj=obj, func=self._func, codec=self._codec)


def store_as(
    state_t: type[FormatState[T]],
    **config: object,
) -> Callable[[Callable[[C, str], T]], StoredMethod[C, T]]:
    """Store the results of the decorated chain method in the given state format.

    Args:
        state_t: State type used to encode and decode the records.
        config: Options forwarded to `state_t` on encoding and decoding.

    Returns:
        A decorator turning a chain method into a `StoredMethod`.

    """
    codec = StateCodec(state_t, **config)

    def decorator(func: Callable[[C, str], T]) -> StoredMethod[C, T]:
        return StoredMethod(func=func, codec=codec)

    return decorator
