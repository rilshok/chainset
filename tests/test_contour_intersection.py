"""Tests for the self-intersection check of closed contours."""

from chainset.dtype.points import Box2D
from chainset.utils.contour_intersection import is_self_intersecting


def test_box_square_is_not_self_intersecting() -> None:
    """The square traced by the corners of a unit box does not meet itself."""
    square = Box2D(((0.0, 0.0), (1.0, 1.0))).patch.points
    assert not is_self_intersecting(square)
