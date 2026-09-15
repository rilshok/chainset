"""Tests for the self-intersection check of closed contours."""

from chainset.dtype.points import Box2D
from chainset.utils.contour_intersection import is_self_intersecting


def test_box_square_is_not_self_intersecting() -> None:
    """The square traced by the corners of a unit box does not meet itself."""
    square = Box2D(((0.0, 0.0), (1.0, 1.0))).patch.points
    assert not is_self_intersecting(square)


def test_bowtie_is_self_intersecting() -> None:
    """A bowtie crosses itself where its two diagonal edges meet."""
    bowtie = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    assert is_self_intersecting(bowtie)


def test_coincident_consecutive_corners_are_not_self_intersecting() -> None:
    """Two corners in a row at the same point collapse into a single vertex."""
    contour = [(0.0, 0.0), (1.0, 0.5), (1.0, 0.5), (0.0, 1.0)]
    assert not is_self_intersecting(contour)


def test_zero_height_box_is_self_intersecting() -> None:
    """A box of zero height traces one segment there and back again."""
    flat = Box2D(((0.0, 0.0), (1.0, 0.0))).patch.points
    assert is_self_intersecting(flat)


def test_square_patch_with_p4_on_p2_is_self_intersecting() -> None:
    """Moving `p4` onto `p2` makes the contour run back along its own edges."""
    contour = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.0)]
    assert is_self_intersecting(contour)


def test_patch_with_p4_on_p2_closed_through_extra_corner_is_not_self_intersecting() -> None:
    """Closing the degenerate patch through an extra corner leaves a triangle."""
    contour = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    assert not is_self_intersecting(contour)


def test_bowtie_through_center_is_not_self_intersecting() -> None:
    """A bowtie that visits the center twice only touches itself there."""
    contour = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.0, 1.0)]
    assert not is_self_intersecting(contour)
