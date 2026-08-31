"""Tests for chain method result serialization and caching."""

import hashlib
import json

from iokit import Data, Json, StreamMemoryStorage

from chainset import Chain, store_as
from chainset.storage import StorageBackend


class TextData:
    """Wrapper around a plain text ``content`` string, for use with ``store_as``."""

    def __init__(self, content: str) -> None:
        """Store the content string."""
        self.content = content


class TextJson(Json[TextData]):
    """JSON (de)serializer for ``TextData``."""

    def dump(self, data: TextData) -> dict[str, str]:
        """Serialize ``data`` to a JSON-compatible dict."""
        return {"content": data.content}

    def parse(self, data: dict[str, str]) -> TextData:
        """Deserialize ``data`` back into a ``TextData``."""
        return TextData(data["content"])


class HashChain(Chain):
    """Chain that hashes UIDs and counts how many times it actually ran."""

    counter: int
    alg: str

    def __init__(self, storage: StorageBackend, alg: str) -> None:
        """Initialize the chain.

        Args:
            storage: Backend used to persist the dataset.
            alg: Hashing algorithm, also used as the chain origin.

        """
        super().__init__(storage)
        self.alg = alg
        self.origin = alg

    @store_as(TextJson)
    def id_digest(self, uid: str) -> TextData:
        """Hash ``uid`` with the chain's origin, counting the call."""
        self.counter += 1
        return TextData(Data.from_ascii(uid).digest(self.alg).hex())


def _hexdigest(value: str, algorithm: str) -> str:
    """Compute the reference hex digest of ``value`` the same way ``HashChain`` does."""
    return hashlib.new(algorithm, value.encode("ascii")).hexdigest()


def test_de_serialization() -> None:
    """Results are cached per origin and (de)serialized to the storage."""
    storage = StreamMemoryStorage()
    chain = HashChain(storage=storage, alg="sha256")
    chain.counter = 0
    assert chain.id_digest("123").content == _hexdigest("123", "sha256")
    assert chain.id_digest("234").content == _hexdigest("234", "sha256")
    assert chain.id_digest("123").content == _hexdigest("123", "sha256")
    assert chain.counter == 2

    chain = HashChain(storage=storage, alg="md5")
    chain.counter = 0
    assert chain.id_digest("123").content == _hexdigest("123", "md5")

    expected = {
        "123/sha256/id_digest.json": _hexdigest("123", "sha256"),
        "234/sha256/id_digest.json": _hexdigest("234", "sha256"),
        "123/md5/id_digest.json": _hexdigest("123", "md5"),
    }

    assert list(storage.index()) == list(expected)

    for key, content in expected.items():
        assert json.loads(storage.pull(key).read())["content"] == content
