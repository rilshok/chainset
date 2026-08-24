"""Records exchanged by dataset processing steps."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from iokit import State
from typing_extensions import Self

T = TypeVar("T", bound=State[Any])


class Record(ABC, Generic[T]):
    """Value that can be converted to and from a serializable state."""

    @property
    @abstractmethod
    def state(self) -> T:
        """Serializable state of the record.

        Returns:
            The state the record is persisted as.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_state(cls, state: T) -> Self:
        """Rebuild a record from its serialized state.

        Args:
            state: State previously produced by `state`.

        Returns:
            The record reconstructed from `state`.

        """
        raise NotImplementedError
