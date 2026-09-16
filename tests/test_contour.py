"""Tests for the self-intersection check of closed contours."""

import pytest

from chainset.utils.contour_intersection import is_self_intersecting

Contour = list[tuple[float, float]]

CONTOURS: list[tuple[Contour, bool]] = [
    # unit box square
    ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], True),
    # bowtie
    ([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)], False),
    # triangle with a doubled corner
    ([(0.0, 0.0), (1.0, 0.5), (1.0, 0.5), (0.0, 1.0)], True),
    # box of zero height
    ([(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)], False),
    # square patch with p4 on p2
    ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.0)], False),
    # triangle with a whisker
    ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)], True),
    # bowtie touching in the center
    ([(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.0, 1.0)], True),
    # bowtie with a corridor between the centers
    ([(0.0, 0.0), (0.5, 0.4), (1.0, 0.0), (1.0, 1.0), (0.5, 0.6), (0.0, 1.0)], True),
    # bowtie with the centers passing each other
    ([(0.0, 0.0), (0.5, 0.6), (1.0, 0.0), (1.0, 1.0), (0.5, 0.4), (0.0, 1.0)], False),
]


@pytest.mark.parametrize(("contour", "valid"), CONTOURS)
def test_is_self_intersecting(contour: Contour, *, valid: bool) -> None:
    """A contour is valid exactly when it does not intersect itself."""
    assert is_self_intersecting(contour) is not valid
