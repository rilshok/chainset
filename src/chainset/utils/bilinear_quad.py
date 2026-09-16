"""Bilinear map between the unit square and a quadrilateral.

A quadrilateral with corners `p1`, `p2`, `p3`, `p4` in loop order is the image
of the unit square under the bilinear map that sends `(0, 0)`, `(1, 0)`,
`(1, 1)` and `(0, 1)` to those corners. `quad_point_at` evaluates the map,
`quad_point_of` inverts it by solving a quadratic, and `quad_meshgrid` samples
it on a regular grid of cell centers.
"""

__all__ = [
    "quad_meshgrid",
    "quad_point_at",
    "quad_point_of",
]

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Point = tuple[float, ...]

_EPS = 1e-12


def quad_point_at(quad: Sequence[Point], point: Point) -> tuple[float, float]:
    """Map a unit-square `point` into `quad` via bilinear interpolation.

    Args:
        quad: The four corners of the quadrilateral, in loop order.
        point: Normalized coordinates in `[0, 1] x [0, 1]`.

    Returns:
        The interpolated point inside `quad`.

    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = quad
    i, j = point
    a, b = 1.0 - i, 1.0 - j

    x = a * b * x1 + i * b * x2 + i * j * x3 + a * j * x4
    y = a * b * y1 + i * b * y2 + i * j * y3 + a * j * y4

    return x, y


def quad_point_of(quad: Sequence[Point], point: Point) -> tuple[float, float]:
    """Map a `point` of the frame of `quad` back to the unit square.

    Inverse of `quad_point_at`. Eliminating the first normalized coordinate
    leaves a quadratic in the second one, which degenerates into a linear
    equation when `quad` is a parallelogram.

    Args:
        quad: The four corners of the quadrilateral, in loop order.
        point: Coordinates in the same frame as `quad`.

    Returns:
        Normalized coordinates in `[0, 1] x [0, 1]` for points inside `quad`,
        extrapolated outside otherwise.

    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = quad
    x, y = point

    bx, by = x2 - x1, y2 - y1
    cx, cy = x4 - x1, y4 - y1
    dx = x1 - x2 + x3 - x4
    dy = y1 - y2 + y3 - y4
    qx, qy = x - x1, y - y1

    j = _solve_bilinear_root(
        a=cy * dx - cx * dy,
        b=qx * dy - qy * dx + bx * cy - by * cx,
        c=qx * by - qy * bx,
    )

    den_x, den_y = bx + j * dx, by + j * dy
    if abs(den_x) >= abs(den_y):
        i = (qx - j * cx) / den_x if abs(den_x) > _EPS else 0.0
    else:
        i = (qy - j * cy) / den_y if abs(den_y) > _EPS else 0.0

    return i, j


def quad_meshgrid(
    quad: Sequence[Point],
    width: int,
    height: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Sample a grid of `height` by `width` cell centers spanning `quad`.

    The grid follows the edges of `quad`, so a warped quadrilateral yields a
    warped grid. Samples sit at cell centers, half a step away from the edges,
    which keeps them symmetric with respect to `quad`.

    Args:
        quad: The four corners of the quadrilateral, in loop order.
        width: Number of samples along the `p1` to `p2` direction.
        height: Number of samples along the `p1` to `p4` direction.

    Returns:
        The x and y coordinates of the samples as `float32`, each of shape
        `(height, width)`.

    """
    p1, p2, p3, p4 = quad
    xy = _linspace(_linspace(p1, p2, width), _linspace(p4, p3, width), height)
    return xy[..., 0], xy[..., 1]


def _linspace(
    start: Point | NDArray[np.float32],
    stop: Point | NDArray[np.float32],
    length: int,
) -> NDArray[np.float32]:
    """Split `start -> stop` into `length` cells and return their centers."""
    points, step = np.linspace(start, stop, length, endpoint=False, retstep=True, dtype=np.float32)
    # `retstep` hands back a float64 step, which would upcast the sum.
    return points + np.asarray(step / 2, dtype=np.float32)


def _unit_interval_distance(value: float) -> float:
    """Measure how far `value` falls outside the unit interval.

    Args:
        value: The value to measure.

    Returns:
        The distance to `[0, 1]`, or `0.0` for a value inside it.

    """
    return max(0.0, -value, value - 1.0)


def _solve_bilinear_root(a: float, b: float, c: float) -> float:
    """Solve `a * t^2 + b * t + c = 0`, preferring a root inside `[0, 1]`.

    Uses the cancellation-free form of the quadratic formula, which matters
    because a nearly affine quadrilateral makes `a` vanishingly small.

    Args:
        a: Quadratic coefficient; zero for an affine (non-warped) quadrilateral.
        b: Linear coefficient.
        c: Constant coefficient.

    Returns:
        The root closest to the unit interval, or `0.0` if the equation degenerates.

    """
    if abs(b) < _EPS and abs(a) < _EPS:
        return 0.0
    if abs(a) < _EPS:
        return -c / b
    disc = math.sqrt(max(b * b - 4.0 * a * c, 0.0))
    q = -0.5 * (b + math.copysign(disc, b))
    roots = (q / a, c / q) if abs(q) > _EPS else (0.0, -b / a)
    return min(roots, key=lambda t: (_unit_interval_distance(t), abs(t - 0.5)))
