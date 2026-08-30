"""Chainable dataset processing."""

__all__: list[str] = [
    "Chain",
    "store_as",
]

from importlib.metadata import version

__version__ = version("chainset")

from .chain import Chain, store_as
