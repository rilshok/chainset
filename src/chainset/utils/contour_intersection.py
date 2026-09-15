"""Self-intersection test for closed planar contours.

`is_self_intersecting` compares every pair of edges of the loop. Neighbouring
edges are allowed to meet at their shared vertex and nowhere else; any other
contact, from a proper crossing down to a vertex grazing an edge, counts.
"""

__all__ = [
    "is_self_intersecting",
]

from collections.abc import Iterable

Point = tuple[float, ...]

_EPS = 1e-12


def is_self_intersecting(points: Iterable[Iterable[float]]) -> bool:
    """Check whether the closed contour through `points` meets itself.

    The contour is a loop: the last point connects back to the first. Any
    contact between edges other than the vertex shared by neighbouring edges
    counts, so besides proper crossings (a figure eight, a "bowtie") this also
    detects a vertex lying on another edge, a vertex visited twice, and edges
    that overlap collinearly, including a spike that doubles back on itself.
    Consecutive duplicate points and an explicit closing point are ignored.

    Every pair of edges is tested, which takes `O(n^2)` time but stays exact
    in all degenerate cases, unlike a sweep line on floating point input.

    Args:
        points: Vertices of the contour in order, each an `(x, y)` pair of any
            kind: tuples, lists or rows of an array.

    Returns:
        `True` if the contour touches or crosses itself, `False` otherwise.

    """
    loop: list[Point] = []
    for vertex in (tuple(map(float, point)) for point in points):
        if not loop or vertex != loop[-1]:
            loop.append(vertex)
    while len(loop) > 1 and loop[0] == loop[-1]:
        loop.pop()
    n = len(loop)
    if n < 2:
        return False

    edges = [(loop[i], loop[(i + 1) % n]) for i in range(n)]
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, n):
            c, d = edges[j]
            if j == i + 1:
                touch = _folds_back(b, a, d)
            elif i == 0 and j == n - 1:
                touch = _folds_back(a, b, c)
            else:
                touch = _segments_touch(a, b, c, d)
            if touch:
                return True
    return False


def _orientation(o: Point, a: Point, b: Point) -> int:
    """Side of the line `o -> a` that `b` lies on.

    Collinearity is decided relative to the magnitude of the cross product
    terms, so the answer does not depend on the scale of the coordinates.

    Returns:
        `1` for a left turn, `-1` for a right turn, `0` when collinear.

    """
    left = (a[0] - o[0]) * (b[1] - o[1])
    right = (a[1] - o[1]) * (b[0] - o[0])
    if abs(left - right) <= _EPS * (abs(left) + abs(right)):
        return 0
    return 1 if left > right else -1


def _within_box(p: Point, a: Point, b: Point) -> bool:
    """Check that `p` lies in the bounding box of the segment `a -> b`."""
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _folds_back(shared: Point, u: Point, v: Point) -> bool:
    """Check whether edges `shared -> u` and `shared -> v` overlap.

    Neighbouring edges always meet at `shared`; they overlap beyond it only
    when both run along the same ray from that vertex.
    """
    if _orientation(shared, u, v) != 0:
        return False
    return (u[0] - shared[0]) * (v[0] - shared[0]) + (u[1] - shared[1]) * (v[1] - shared[1]) > 0.0


def _segments_touch(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Check whether the closed segments `a -> b` and `c -> d` share a point."""
    d1 = _orientation(c, d, a)
    d2 = _orientation(c, d, b)
    d3 = _orientation(a, b, c)
    d4 = _orientation(a, b, d)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return (
        (d1 == 0 and _within_box(a, c, d))
        or (d2 == 0 and _within_box(b, c, d))
        or (d3 == 0 and _within_box(c, a, b))
        or (d4 == 0 and _within_box(d, a, b))
    )
