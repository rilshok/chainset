"""Safe repairs of a closed contour, to be made before it is judged.

A contour often arrives with flaws that say nothing about the region it bounds:
a vertex listed twice in a row, an explicit closing vertex, or an excursion
that leaves a vertex and comes straight back along itself. None of them moves
the boundary or changes the enclosed area, so all of them can be dropped.

Flaws that do move the boundary are left alone. An edge walked twice with a
detour in between, for one, is how a contour reaches a hole through a bridge of
zero width, and dropping it would join the hole to the background.
"""

__all__ = [
    "contour_loop",
    "normalize_contour",
]

from collections import deque
from collections.abc import Iterable

Point = tuple[float, float]


def contour_loop(points: Iterable[Iterable[float]]) -> list[Point]:
    """Read `points` as a loop of vertices.

    Args:
        points: Vertices of the contour in order, each an `(x, y)` pair of any
            kind: tuples, lists or rows of an array.

    Returns:
        The vertices as floats, without repeats in a row and without a closing
        vertex that repeats the first one.

    Raises:
        ValueError: If a point does not hold exactly two coordinates.

    """
    loop: list[Point] = []
    for index, point in enumerate(points):
        coordinates = tuple(point)
        if len(coordinates) != 2:
            msg = f"Point {index} has {len(coordinates)} coordinates, expected 2."
            raise ValueError(msg)
        vertex = (float(coordinates[0]), float(coordinates[1]))
        if not loop or vertex != loop[-1]:
            loop.append(vertex)
    while len(loop) > 1 and loop[0] == loop[-1]:
        loop.pop()
    return loop


def normalize_contour(points: Iterable[Iterable[float]]) -> list[Point]:
    """Read `points` as a loop and drop the excursions that enclose nothing.

    An excursion out to a vertex and straight back turns at a vertex whose
    neighbours coincide; dropping that vertex leaves its neighbours next to
    each other, which may expose the excursion that hid behind it, so the
    vertices are folded onto a stack and the seam between the last and the
    first is settled afterwards.

    Two vertices or fewer are left alone: such a loop encloses nothing, which
    is for the caller to reject.

    Args:
        points: Vertices of the contour in order, each an `(x, y)` pair of any
            kind: tuples, lists or rows of an array.

    Returns:
        The vertices of a loop of the same enclosed area, free of repeats and
        of zero-width excursions.

    Raises:
        ValueError: If a point does not hold exactly two coordinates.

    """
    loop: deque[Point] = deque()
    for vertex in contour_loop(points):
        while len(loop) > 1 and loop[-2] == vertex:
            loop.pop()
        if loop and loop[-1] == vertex:
            continue
        loop.append(vertex)

    while len(loop) > 2:
        if loop[0] == loop[-1]:
            loop.pop()
        elif loop[1] == loop[-1]:
            # The first vertex turns an excursion, and its neighbours then meet.
            loop.popleft()
            loop.pop()
        elif loop[0] == loop[-2]:
            # So does the last one.
            loop.pop()
            loop.pop()
        else:
            break
    return list(loop)
