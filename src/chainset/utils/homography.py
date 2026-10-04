"""Homography between the unit square and a convex quadrilateral."""

__all__ = [
    "homography_jacobian",
    "homography_meshgrid",
    "homography_point_at",
    "homography_point_of",
    "quad_aspect_ratio",
    "quad_homography",
    "quad_lossless_height",
]

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Point = tuple[float, ...]


def quad_homography(quad: Sequence[Sequence[float]]) -> NDArray[np.float64]:
    """Find the homography mapping the unit square onto `quad`.

    Args:
        quad: Four corners of a strictly convex quadrilateral, in loop order.

    Returns:
        A 3x3 matrix with the last entry equal to one.

    """
    p1, p2, p3, p4 = (np.array([x, y, 1.0]) for x, y in quad)
    depth2, depth4, _ = np.linalg.solve(np.column_stack([p2, p4, -p3]), p1)
    return np.column_stack([depth2 * p2 - p1, depth4 * p4 - p1, p1])


def homography_point_at(quad: Sequence[Sequence[float]], point: Point) -> tuple[float, float]:
    """Map a unit-square `point` into `quad`.

    Args:
        quad: Four corners of a strictly convex quadrilateral, in loop order.
        point: Normalized coordinates.

    Returns:
        The mapped point.

    """
    x, y, weight = quad_homography(quad) @ (*point, 1.0)
    return float(x / weight), float(y / weight)


def homography_meshgrid(
    quad: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Map a grid of `height` by `width` cell centers of the unit square into `quad`.

    Args:
        quad: Four corners of a strictly convex quadrilateral, in loop order.
        width: Number of samples along `p1 -> p2`.
        height: Number of samples along `p1 -> p4`.

    Returns:
        The x and y coordinates of the samples as `float32`, of shape `(height, width)`.

    """
    s, t = np.meshgrid((np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height)
    x, y, weight = np.tensordot(quad_homography(quad), np.stack([s, t, np.ones_like(s)]), 1)
    return (x / weight).astype(np.float32), (y / weight).astype(np.float32)


def homography_point_of(quad: Sequence[Sequence[float]], point: Point) -> tuple[float, float]:
    """Map a `point` of `quad` back to the unit square.

    Args:
        quad: Four corners of a strictly convex quadrilateral, in loop order.
        point: Coordinates in the frame of `quad`.

    Returns:
        The normalized coordinates.

    """
    s, t, weight = np.linalg.solve(quad_homography(quad), (*point, 1.0))
    return float(s / weight), float(t / weight)


def quad_aspect_ratio(quad: Sequence[Sequence[float]], width: float, height: float) -> float:
    """Estimate the width-to-height ratio of the rectangle seen as `quad`.

    Uses Zhang and He (2007): a pinhole camera centered on the image. When the
    focal length cannot be estimated, it is taken equal to the image diagonal.

    Args:
        quad: Four corners of a strictly convex quadrilateral, in pixels.
        width: Width of the image, in pixels.
        height: Height of the image, in pixels.

    Returns:
        The ratio of side `p1 -> p2` to side `p1 -> p4`.

    """
    homography = quad_homography(quad)
    tilts = homography[2, :2]
    sides = homography[:2, :2] - np.outer((width / 2.0, height / 2.0), tilts)
    diagonal = math.hypot(width, height)
    focal_squared = diagonal * diagonal
    # Vanishing points beyond a thousand diagonals count as infinite.
    if np.all(np.abs(tilts) * 1e3 * diagonal > np.hypot(*sides)):
        estimate = -float(sides[:, 0] @ sides[:, 1]) / float(tilts[0] * tilts[1])
        if estimate > 0.0:
            focal_squared = estimate
    across, down = (sides * sides).sum(axis=0) + focal_squared * tilts * tilts
    return math.sqrt(across / down)


def quad_lossless_height(quad: Sequence[Sequence[float]], ratio: float) -> float:
    """Give the least height at which a cut of `quad` downsamples none of it.

    Args:
        quad: Four corners of a strictly convex quadrilateral, in pixels.
        ratio: Width-to-height ratio of the cut.

    Returns:
        The height, in pixels.

    """
    # The map from a cut one pixel high onto `quad`.
    homography = quad_homography(quad) @ np.diag([1.0 / ratio, 1.0, 1.0])
    u, v = np.meshgrid(np.linspace(0.0, ratio, 17), np.linspace(0.0, 1.0, 17))
    xu, yu, xv, yv = homography_jacobian(homography, u, v)
    # Source pixels under one output pixel along its widest direction.
    squares = xu * xu + yu * yu + xv * xv + yv * yv
    det = xu * yv - xv * yu
    return float(np.sqrt((squares + np.sqrt(np.maximum(squares**2 - 4 * det**2, 0))) / 2).max())


def homography_jacobian(
    homography: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Differentiate the map `homography` at the points `(u, v)`.

    Args:
        homography: A 3x3 projective matrix.
        u: First coordinates of the points.
        v: Second coordinates of the points.

    Returns:
        The derivatives of `x` and `y` along `u`, then of `x` and `y` along `v`.

    """
    (a, b, c), (d, e, f), (g, h, i) = homography
    w = g * u + h * v + i
    x, y = (a * u + b * v + c) / w, (d * u + e * v + f) / w
    return (a - g * x) / w, (d - g * y) / w, (b - h * x) / w, (e - h * y) / w
