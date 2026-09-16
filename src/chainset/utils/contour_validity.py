"""Validity test for closed planar contours.

A valid contour bounds an object of non-zero area: every part of it of non-zero
length keeps the object on one side and the background on the other, always the
same side along the direction of traversal. In winding numbers, that is zero
around the background and the same one, either way round, around the object.

The test is local. Neighbouring faces differ by one winding, so a face winding
twice, or the wrong way, is only reachable through a place where the contour
meets itself, and around such a place the winding numbers follow from the
strands alone: counter-clockwise, a strand leaving adds one and a strand
arriving takes one away. The contour is valid when every such place sees at most
two winding numbers, no two edges share a segment, and the area is not zero.

A sweep along the less crowded axis finds the places. Predicates are exact:
floating point first, fractions whenever its answer cannot be trusted, so three
edges through one point make one place and not three.
"""

__all__ = [
    "is_valid_contour",
]

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from fractions import Fraction
from functools import cmp_to_key

from chainset.utils.contour_normalization import contour_loop
from chainset.utils.polygon_area import signed_area

Point = tuple[float, float]

# An edge, from one vertex to the next.
Edge = tuple[Point, Point]

# A place the contour meets itself: a vertex, or a crossing of two edges.
Spot = tuple[Fraction | float, Fraction | float]

# Where a strand through a place comes from and goes to.
Strand = tuple[Point, Point]

# Direction from a place towards a neighbour.
Ray = tuple[Fraction, Fraction]

# Bounds of an edge, along the swept axis first and across it second.
Box = tuple[float, float, float, float]

# Relative error bound of a determinant of products of two floats.
_ROUNDING = 2.0**-50


def is_valid_contour(points: Iterable[Iterable[float]]) -> bool:
    """Check whether `points` trace a contour that bounds an object.

    The contour is a loop: the last point connects back to the first. Repeated
    points in a row and an explicit closing point are ignored. Meeting itself is
    allowed while the object stays on one side: loops may meet at a point, a
    hole may touch its outline, and edges may even cross where a third one runs
    through the same crossing.

    Args:
        points: Vertices of the contour in order, each an `(x, y)` pair of any
            kind: tuples, lists or rows of an array.

    Returns:
        `True` if the contour bounds an object, `False` otherwise.

    Raises:
        ValueError: If a point does not hold exactly two coordinates.

    """
    loop = contour_loop(points)
    if len(loop) < 3 or signed_area(loop) == 0.0:
        return False

    size = len(loop)
    edges: list[Edge] = [(vertex, loop[(index + 1) % size]) for index, vertex in enumerate(loop)]

    through: set[tuple[Spot, int]] = set()
    for this, other in _candidates(_boxes(edges)):
        (a, b), (c, d) = edges[this], edges[other]
        first, second = _orientation(a, b, c), _orientation(a, b, d)
        if first * second > 0:
            continue
        third, fourth = _orientation(c, d, a), _orientation(c, d, b)
        if third * fourth > 0:
            continue
        if first == 0 and second == 0:
            # Collinear edges either overlap or abut at an end, which is a vertex.
            if _overlaps_along(a, b, c, d):
                return False
            continue
        if first and second and third and fourth:
            crossing = _crossing_spot(a, b, c, d)
            through.update(((crossing, this), (crossing, other)))
            continue
        ends = (
            (a, third, c, d, other),
            (b, fourth, c, d, other),
            (c, first, a, b, this),
            (d, second, a, b, this),
        )
        through.update(
            (vertex, index)
            for vertex, side, p, q, index in ends
            if side == 0 and _strictly_inside(vertex, p, q)
        )

    return _places_agree(loop, edges, through)


def _boxes(edges: Sequence[Edge]) -> list[Box]:
    """Bounds of the edges, arranged so that the sweep runs the cheaper way."""
    xs = [(min(a[0], b[0]), max(a[0], b[0])) for a, b in edges]
    ys = [(min(a[1], b[1]), max(a[1], b[1])) for a, b in edges]
    if _crowding(xs) <= _crowding(ys):
        return [(lo, hi, across, beyond) for (lo, hi), (across, beyond) in zip(xs, ys, strict=True)]
    return [(lo, hi, across, beyond) for (lo, hi), (across, beyond) in zip(ys, xs, strict=True)]


def _crowding(bounds: Sequence[tuple[float, float]]) -> float:
    """How many edges a sweep along this axis carries at a time, on average."""
    spread = max(hi for _, hi in bounds) - min(lo for lo, _ in bounds)
    return math.fsum(hi - lo for lo, hi in bounds) / spread if spread > 0.0 else math.inf


def _candidates(boxes: Sequence[Box]) -> Iterator[tuple[int, int]]:
    """Yield, by index, the pairs of edges whose boxes overlap, by a sweep.

    Edges join the sweep ordered by the near end of their bounds and drop out of
    it once the sweep has passed their far end.
    """
    carried: list[int] = []
    for index in sorted(range(len(boxes)), key=lambda edge: boxes[edge][0]):
        near, _, across, beyond = boxes[index]
        kept: list[int] = []
        for other in carried:
            _, other_far, other_across, other_beyond = boxes[other]
            if other_far < near:
                continue
            kept.append(other)
            if across <= other_beyond and other_across <= beyond:
                yield index, other
        kept.append(index)
        carried = kept


def _places_agree(
    loop: Sequence[Point],
    edges: Sequence[Edge],
    through: set[tuple[Spot, int]],
) -> bool:
    """Check the places the contour meets itself, if it meets itself at all.

    A place is a crossing, a vertex an edge runs through, or a vertex the
    contour visits twice; its strands are those edges and the contour itself.
    """
    repeated = {vertex for vertex, times in Counter(loop).items() if times > 1}
    if not through and not repeated:
        return True

    meetings: defaultdict[Spot, list[Strand]] = defaultdict(list)
    for spot, index in through:
        meetings[spot].append(edges[index])

    size = len(loop)
    places = repeated.union(spot for spot, _ in through)
    for index, vertex in enumerate(loop):
        if vertex in places:
            meetings[vertex].append((loop[index - 1], loop[(index + 1) % size]))

    return all(
        _sectors_agree(spot, strands) for spot, strands in meetings.items() if len(strands) > 1
    )


def _sectors_agree(spot: Spot, strands: Sequence[Strand]) -> bool:
    """Check that the sectors around `spot` show at most two winding numbers.

    Counter-clockwise, a strand leaving adds one and a strand arriving takes one
    away, which gives every sector its winding up to a common offset.
    """
    center = (Fraction(spot[0]), Fraction(spot[1]))
    rays = [
        (_ray(center, point), step)
        for arrives, leaves in strands
        for point, step in ((arrives, -1), (leaves, 1))
    ]
    rays.sort(key=cmp_to_key(_by_angle))

    winding = 0
    seen = {winding}
    for _, step in rays:
        winding += step
        seen.add(winding)
    return max(seen) - min(seen) <= 1


def _ray(center: Ray, point: Point) -> Ray:
    """Direction from `center` towards `point`, exactly."""
    return Fraction(point[0]) - center[0], Fraction(point[1]) - center[1]


def _by_angle(left: tuple[Ray, int], right: tuple[Ray, int]) -> int:
    """Order two rays counter-clockwise from the positive x direction, exactly."""
    u, v = left[0], right[0]
    if (half := _half(u)) != (other := _half(v)):
        return -1 if half < other else 1
    cross = u[0] * v[1] - u[1] * v[0]
    return -1 if cross > 0 else int(cross < 0)


def _half(ray: Ray) -> int:
    """Tell the upper half plane, the positive x axis included, from the lower one."""
    return 0 if ray[1] > 0 or (ray[1] == 0 and ray[0] > 0) else 1


def _crossing_spot(a: Point, b: Point, c: Point, d: Point) -> Spot:
    """Exact place where the segments `a -> b` and `c -> d` cross."""
    ax, ay = Fraction(a[0]), Fraction(a[1])
    rx, ry = Fraction(b[0]) - ax, Fraction(b[1]) - ay
    cx, cy = Fraction(c[0]), Fraction(c[1])
    sx, sy = Fraction(d[0]) - cx, Fraction(d[1]) - cy
    step = ((cx - ax) * sy - (cy - ay) * sx) / (rx * sy - ry * sx)
    return ax + step * rx, ay + step * ry


def _overlaps_along(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Check whether two collinear segments share more than one point.

    They are compared along the axis they run more steeply through, so that
    neither of them shrinks to a point.
    """
    axis = 0 if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else 1
    first = sorted((a[axis], b[axis]))
    second = sorted((c[axis], d[axis]))
    return min(first[1], second[1]) > max(first[0], second[0])


def _strictly_inside(vertex: Point, a: Point, b: Point) -> bool:
    """Check whether `vertex`, known to be on the line `a -> b`, lies between them."""
    if vertex in (a, b):
        return False
    inside = min(a[0], b[0]) <= vertex[0] <= max(a[0], b[0])
    return inside and min(a[1], b[1]) <= vertex[1] <= max(a[1], b[1])


def _orientation(o: Point, a: Point, b: Point) -> int:
    """Side of the line `o -> a` that `b` lies on, in floats and then in fractions.

    Returns:
        `1` for a left turn, `-1` for a right turn, `0` when collinear.

    """
    left = (a[0] - o[0]) * (b[1] - o[1])
    right = (a[1] - o[1]) * (b[0] - o[0])
    if abs(left - right) > _ROUNDING * (abs(left) + abs(right)):
        return 1 if left > right else -1
    ax, ay = Fraction(a[0]) - Fraction(o[0]), Fraction(a[1]) - Fraction(o[1])
    bx, by = Fraction(b[0]) - Fraction(o[0]), Fraction(b[1]) - Fraction(o[1])
    exact = ax * by - ay * bx
    return (exact > 0) - (exact < 0)
