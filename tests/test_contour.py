"""Tests for the self-intersection check of closed contours."""

import pytest

from chainset.utils.contour_intersection import is_self_intersecting

Contour = list[tuple[float, float]]

# The unit square, the plainest contour there is.
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

# A triangle whose apex is listed twice in a row.
DOUBLED_CORNER = [(0.0, 0.0), (1.0, 0.5), (1.0, 0.5), (0.0, 1.0)]

# Two loops meeting at the center, which the contour passes through twice.
CENTER_TOUCH = [
    (0.0, 0.0),
    (0.5, 0.5),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.5, 0.5),
    (0.0, 1.0),
]

# The same two loops, pulled apart into a corridor.
CENTER_CORRIDOR = [
    (0.0, 0.0),
    (0.5, 0.4),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.5, 0.6),
    (0.0, 1.0),
]

# A square wound clockwise and started from another corner.
CLOCKWISE_SQUARE = [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)]

# A triangle with an extra vertex sitting on a straight edge.
MIDEDGE_VERTEX = [(-2.0, 0.0), (0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]

# An L shape, to cover a concave contour.
L_SHAPE = [
    (-1.0, -1.0),
    (1.0, -1.0),
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 2.0),
    (-1.0, 2.0),
]

# A rectangle closed by hand: the last point repeats the first.
CLOSED_SQUARE = [
    (-3.0, -1.0),
    (-1.0, -1.0),
    (-1.0, 1.0),
    (-3.0, 1.0),
    (-3.0, -1.0),
]

# A corridor a thousand times narrower than the contour around it.
NARROW_CORRIDOR = [
    (-100.0, -100.0),
    (0.0, -0.001),
    (100.0, -100.0),
    (100.0, 100.0),
    (0.0, 0.001),
    (-100.0, 100.0),
]

# Two loops that meet where a vertex lands in the middle of an edge.
VERTEX_ON_EDGE = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 0.0), (0.0, 2.0)]

# A hole wound against the outline, touching it at a single point.
TOUCHING_HOLE = [
    (-2.0, -2.0),
    (2.0, -2.0),
    (2.0, 2.0),
    (0.0, 2.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 2.0),
    (-2.0, 2.0),
]

# A hole reached through a corridor of its own, never touching the outline.
KEYHOLE = [
    (0.0, 1.6),
    (0.0, 0.0),
    (4.0, 0.0),
    (4.0, 4.0),
    (0.0, 4.0),
    (0.0, 2.4),
    (1.0, 2.4),
    (1.0, 3.0),
    (3.0, 3.0),
    (3.0, 1.0),
    (1.0, 1.0),
    (1.0, 1.6),
]

# The two diagonals of a square, crossing in the middle.
BOWTIE = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]

# A box of zero height: one segment walked there and back.
FLAT_BOX = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]

# A square patch with its fourth corner moved onto the second.
PATCH_P4_ON_P2 = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.0)]

# A triangle with an edge that runs out and back along itself.
WHISKER = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]

# Two loops whose tips overshoot each other instead of meeting.
CROSSED_CENTERS = [
    (0.0, 0.0),
    (0.5, 0.6),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.5, 0.4),
    (0.0, 1.0),
]

# No points at all.
EMPTY: Contour = []

# A contour of one point.
SINGLE_POINT = [(-0.5, 2.0)]

# A contour of two points, with no area between them.
TWO_POINTS = [(-1.0, -1.0), (2.0, 1.0)]

# Three points, all in the same place.
COINCIDENT_POINTS = [(-2.0, 3.0), (-2.0, 3.0), (-2.0, 3.0)]

# Three distinct points that lie on one line.
COLLINEAR_POINTS = [(-1.0, -1.0), (0.5, 0.5), (2.0, 2.0)]

# Two squares joined at a corner the contour visits twice.
SHARED_VERTEX_LOOPS = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (0.0, 0.0),
    (0.0, -1.0),
    (-1.0, -1.0),
    (-1.0, 0.0),
]

# The same square traced twice in a row.
DOUBLED_SQUARE = [
    (1.0, 1.0),
    (3.0, 1.0),
    (3.0, 3.0),
    (1.0, 3.0),
    (1.0, 1.0),
    (3.0, 1.0),
    (3.0, 3.0),
    (1.0, 3.0),
]

# A five pointed star, every edge crossing two others.
PENTAGRAM = [
    (0.0, 2.0),
    (-1.18, -1.62),
    (1.9, 0.62),
    (-1.9, 0.62),
    (1.18, -1.62),
]

# A contour whose closing edge cuts across an earlier one.
CROSSED_CLOSING_EDGE = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]

# An inner loop wound like the outline, touching it at a single point.
SAME_WINDING_LOOP = [
    (-2.0, -2.0),
    (2.0, -2.0),
    (2.0, 2.0),
    (0.0, 2.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 2.0),
    (-2.0, 2.0),
]

# A hole reached by a bridge of zero width, walked along both sides.
ZERO_WIDTH_BRIDGE = [
    (-1.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
    (-0.5, -0.5),
    (-0.5, 0.5),
    (0.5, 0.5),
    (0.5, -0.5),
    (-0.5, -0.5),
]


# A valid contour is a closed curve that bounds an object of non-zero area. Every
# portion of it with non-zero length separates the object from the background, and the
# object always lies on the same side of the direction of traversal. A point belongs to
# the object if a ray cast from it crosses the contour once more in one direction than
# in the other, and to the background if the crossings in the two directions are equal
# in number.

VALID_CONTOURS: list[Contour] = [
    SQUARE,
    DOUBLED_CORNER,
    CENTER_TOUCH,  # has problem
    CENTER_CORRIDOR,
    CLOCKWISE_SQUARE,
    MIDEDGE_VERTEX,
    L_SHAPE,
    CLOSED_SQUARE,
    NARROW_CORRIDOR,
    VERTEX_ON_EDGE,  # has problem
    TOUCHING_HOLE,  # has problem
    KEYHOLE,
]

INVALID_CONTOURS: list[Contour] = [
    BOWTIE,
    FLAT_BOX,
    PATCH_P4_ON_P2,
    WHISKER,
    CROSSED_CENTERS,
    EMPTY,  # has problem
    SINGLE_POINT,  # has problem
    TWO_POINTS,
    COINCIDENT_POINTS,  # has problem
    COLLINEAR_POINTS,
    SHARED_VERTEX_LOOPS,
    DOUBLED_SQUARE,
    PENTAGRAM,
    CROSSED_CLOSING_EDGE,
    SAME_WINDING_LOOP,
    ZERO_WIDTH_BRIDGE,
]


@pytest.mark.parametrize("contour", VALID_CONTOURS)
def test_valid_contour_does_not_intersect_itself(contour: Contour) -> None:
    """A valid contour bounds a shape without meeting itself."""
    assert not is_self_intersecting(contour)


@pytest.mark.parametrize("contour", INVALID_CONTOURS)
def test_invalid_contour_intersects_itself(contour: Contour) -> None:
    """An invalid contour crosses or touches itself somewhere."""
    assert is_self_intersecting(contour)
