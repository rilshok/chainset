"""Area of planar polygons and of their overlap.

`signed_area` is the shoelace formula, evaluated so that neither the distance
from the origin nor the number of vertices costs precision. `quad_coverage`
builds on it: it clips a polygon against a quadrilateral and compares the
area that survives with the area of the whole polygon.
"""

__all__ = [
    "quad_coverage",
    "signed_area",
]

import math
from collections.abc import Sequence
from itertools import pairwise

Point = tuple[float, ...]

_EPS = 1e-12


def signed_area(polygon: Sequence[Point]) -> float:
    """Signed polygon area, positive when the vertices wind counter-clockwise.

    The shoelace formula is exact for simple polygons, convex or not; for a
    self-intersecting loop, regions wound in opposite directions cancel out.
    Coordinates are taken relative to the first vertex, so a polygon far from
    the origin does not lose precision to large cancelling cross products, and
    the terms are summed with `math.fsum` to avoid accumulated rounding error.
    Terms involving the first vertex vanish, which closes the loop for free.

    Args:
        polygon: Vertices of a closed polygon in order.

    Returns:
        The signed area, or `0.0` for fewer than three vertices.

    """
    if len(polygon) < 3:
        return 0.0
    ox, oy = polygon[0]
    rel = [(x - ox, y - oy) for x, y in polygon[1:]]
    return math.fsum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in pairwise(rel)) / 2.0


def quad_coverage(polygon: Sequence[Point], quad: Sequence[Point]) -> float:
    """Fraction of the area of `polygon` that lies inside the quadrilateral `quad`.

    Clipping needs a convex window, so `quad` is fanned into two triangles from
    its first corner. The fan is exact for a non-convex `quad` too, as long as
    each triangle contributes with the sign of its winding: the part of the fan
    that falls outside `quad` is then cancelled out.

    Args:
        polygon: Vertices of the covered polygon in order.
        quad: The four corners of the covering quadrilateral, in loop order.

    Returns:
        A value in `[0, 1]`; `0.0` when `polygon` is degenerate.

    """
    area = abs(signed_area(polygon))
    if area <= _EPS:
        return 0.0

    a, b, c, d = quad
    covered = 0.0
    for triangle in ((a, b, c), (a, c, d)):
        if abs(signed := signed_area(triangle)) <= _EPS:
            continue
        window = triangle if signed > 0.0 else triangle[::-1]
        clipped: Sequence[Point] = polygon
        for p, q in pairwise([*window, window[0]]):
            clipped = _clip(clipped, p, q)
            if not clipped:
                break
        covered += math.copysign(abs(signed_area(clipped)), signed)
    return min(abs(covered) / area, 1.0)


def _clip(polygon: Sequence[Point], p: Point, q: Point) -> list[Point]:
    """Clip a non-empty `polygon` to the left half-plane of the line `p -> q`.

    One step of Sutherland-Hodgman clipping.

    Args:
        polygon: Vertices of the subject polygon in order.
        p: Origin of the directed line.
        q: Target of the directed line.

    Returns:
        The clipped polygon, empty when nothing survives.

    """
    (px, py), (qx, qy) = p, q
    ex, ey = qx - px, qy - py

    result: list[Point] = []
    ax, ay = polygon[-1]
    before = ex * (ay - py) - ey * (ax - px)
    for point in polygon:
        bx, by = point
        side = ex * (by - py) - ey * (bx - px)
        if (side >= 0.0) != (before >= 0.0):
            step = before / (before - side)
            result.append((ax + (bx - ax) * step, ay + (by - ay) * step))
        if side >= 0.0:
            result.append(point)
        ax, ay, before = bx, by, side
    return result
