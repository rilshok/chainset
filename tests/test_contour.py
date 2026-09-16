"""Tests for the validity check of closed contours."""

import pytest

from chainset.dtype.points import Polygon2D
from chainset.utils.contour_validity import is_valid_contour
from chainset.utils.polygon_area import signed_area

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

# A vertex that passes through an edge instead of stopping on it.
VERTEX_CROSSING = [
    (0.0, 0.0),
    (4.0, 0.0),
    (4.0, 2.0),
    (2.0, 0.0),
    (2.0, -2.0),
    (0.0, -2.0),
]

# A slot of zero width, cut into an edge and walked along both of its sides.
PARTIAL_OVERLAP = [
    (0.0, 0.0),
    (3.0, 0.0),
    (3.0, 1.0),
    (2.0, 1.0),
    (2.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
]

# Three edges through one point, none of them changing the side the object is on.
TRIPLE_POINT = [
    (2.0, 3.0),
    (3.0, 2.0),
    (0.0, 1.0),
    (0.0, 2.0),
    (3.0, 1.0),
    (1.0, 0.0),
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
    CENTER_TOUCH,
    CENTER_CORRIDOR,
    CLOCKWISE_SQUARE,
    MIDEDGE_VERTEX,
    L_SHAPE,
    CLOSED_SQUARE,
    NARROW_CORRIDOR,
    VERTEX_ON_EDGE,
    TOUCHING_HOLE,
    KEYHOLE,
    TRIPLE_POINT,
]

INVALID_CONTOURS: list[Contour] = [
    BOWTIE,
    FLAT_BOX,
    PATCH_P4_ON_P2,
    WHISKER,
    CROSSED_CENTERS,
    EMPTY,
    SINGLE_POINT,
    TWO_POINTS,
    COINCIDENT_POINTS,
    COLLINEAR_POINTS,
    SHARED_VERTEX_LOOPS,
    DOUBLED_SQUARE,
    PENTAGRAM,
    CROSSED_CLOSING_EDGE,
    SAME_WINDING_LOOP,
    ZERO_WIDTH_BRIDGE,
    VERTEX_CROSSING,
    PARTIAL_OVERLAP,
]


@pytest.mark.parametrize("contour", VALID_CONTOURS)
def test_valid_contour(contour: Contour) -> None:
    """A valid contour bounds an object of non-zero area."""
    assert is_valid_contour(contour)


@pytest.mark.parametrize("contour", INVALID_CONTOURS)
def test_invalid_contour(contour: Contour) -> None:
    """An invalid contour bounds nothing, or bounds it from both sides."""
    assert not is_valid_contour(contour)


# A polygon repairs what it is given before judging it, so an invalid contour
# whose only flaw encloses nothing still becomes a polygon.
REPAIRABLE_CONTOURS: list[Contour] = [WHISKER]


@pytest.mark.parametrize("contour", VALID_CONTOURS + REPAIRABLE_CONTOURS)
def test_valid_contour_builds_a_polygon(contour: Contour) -> None:
    """A valid contour is taken as it is."""
    assert Polygon2D(contour).area > 0.0


BEYOND_REPAIR: list[Contour] = [
    contour for contour in INVALID_CONTOURS if contour not in REPAIRABLE_CONTOURS
]


@pytest.mark.parametrize("contour", BEYOND_REPAIR)
def test_contour_beyond_repair_is_rejected(contour: Contour) -> None:
    """No repair turns a contour that crosses itself into a polygon."""
    with pytest.raises(ValueError, match="Polygon2D requires"):
        Polygon2D(contour)


# The signed area every contour above encloses, worked out by hand. A contour that
# winds clockwise encloses a negative area, and one that encloses nothing, or encloses
# as much one way round as the other, encloses zero.

AREAS: list[tuple[Contour, float]] = [
    (SQUARE, 1.0),
    (DOUBLED_CORNER, 0.5),
    (CENTER_TOUCH, 0.5),
    (CENTER_CORRIDOR, 0.6),
    (CLOCKWISE_SQUARE, -4.0),
    (MIDEDGE_VERTEX, 4.0),
    (L_SHAPE, 4.0),
    (CLOSED_SQUARE, 4.0),
    (NARROW_CORRIDOR, 20000.2),
    (VERTEX_ON_EDGE, 4.0),
    (TOUCHING_HOLE, 14.0),
    (KEYHOLE, 11.2),
    (BOWTIE, 0.0),
    (FLAT_BOX, 0.0),
    (PATCH_P4_ON_P2, 0.0),
    (WHISKER, 0.5),
    (CROSSED_CENTERS, 0.4),
    (EMPTY, 0.0),
    (SINGLE_POINT, 0.0),
    (TWO_POINTS, 0.0),
    (COINCIDENT_POINTS, 0.0),
    (COLLINEAR_POINTS, 0.0),
    (SHARED_VERTEX_LOOPS, 0.0),
    (DOUBLED_SQUARE, 8.0),
    (PENTAGRAM, 5.8844),
    (CROSSED_CLOSING_EDGE, 0.0),
    (SAME_WINDING_LOOP, 18.0),
    (ZERO_WIDTH_BRIDGE, 3.0),
    (VERTEX_CROSSING, -2.0),
    (PARTIAL_OVERLAP, 2.0),
    (TRIPLE_POINT, -3.0),
]


@pytest.mark.parametrize(("contour", "expected"), AREAS)
def test_signed_area(contour: Contour, expected: float) -> None:
    """The shoelace sum reproduces the area worked out by hand."""
    assert signed_area(contour) == pytest.approx(expected, abs=1e-9)
