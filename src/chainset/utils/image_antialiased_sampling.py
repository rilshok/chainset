"""Antialiased cutting of a quadrilateral out of an image, undoing perspective."""

__all__ = [
    "sample_quad_antialiased_uint8",
]

import math
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from chainset.utils.homography import homography_jacobian, quad_homography
from chainset.utils.image_patch_sampling import (
    WHITE,
    Corners,
    FillValue,
    sample_quad_projective_uint8,
    sample_quad_uint8,
)


def sample_quad_antialiased_uint8(
    image: NDArray[np.uint8],
    corners: Corners,
    shape: tuple[int, int],
    *,
    fill: FillValue = WHITE,
) -> NDArray[np.uint8]:
    """Cut the quadrilateral `corners` out of `image`, undoing its perspective.

    The source is box-reduced to the scale of the cut, then sampled bilinearly.
    Where the scale varies too much for one reduction, the cut goes in strips,
    each reduced on its own.

    Args:
        image: Source of shape `(height, width, 3)`.
        corners: Four corners of a strictly convex quadrilateral, in pixels.
        shape: Height and width of the result.
        fill: Value outside the image, one level or one per channel.

    Returns:
        The cut, of shape `(*shape, 3)`.

    Raises:
        ValueError: If `corners` or `fill` is unusable.

    """
    height, width = shape
    if not (height and width):
        return sample_quad_uint8(image, corners, shape, fill=fill)
    p1, p2, p3, p4 = np.asarray(corners, dtype=np.float64)
    # On a parallelogram the homography is the bilinear map, sampled faster.
    parallel = np.abs(p1 + p3 - p2 - p4).max() < 1e-6
    sample = sample_quad_uint8 if parallel else sample_quad_projective_uint8
    homography = quad_homography(corners) @ np.diag([1.0 / width, 1.0 / height, 1.0])
    least, most = _reduction(homography, (0, 0, width, height))
    if least == most:
        return sample(*_reduced(image, corners, least), shape, fill=fill)

    out = np.empty((height, width, image.shape[2]), dtype=np.uint8)
    # Four strips across the direction in which the scale changes, each reduced on its own.
    g, h, _ = homography[2]
    down = abs(g) * width <= abs(h) * height
    for start, stop in pairwise(np.linspace(0, height if down else width, 5).astype(int)):
        if start == stop:
            continue
        left, top, right, bottom = (0, start, width, stop) if down else (start, 0, stop, height)
        x, y, w = homography @ [[left, right, right, left], [top, top, bottom, bottom], [1] * 4]
        strip = list(zip(x / w, y / w, strict=True))
        reduction = _reduction(homography, (left, top, right, bottom))[0]
        cut = sample(*_reduced(image, strip, reduction), (bottom - top, right - left), fill=fill)
        out[top:bottom, left:right] = cut
    return out


def _reduction(
    homography: NDArray[np.float64],
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Give the least and most box reduction of the source over `box` of the result."""
    left, top, right, bottom = box
    u, v = np.meshgrid(np.linspace(left, right, 5), np.linspace(top, bottom, 5))
    xu, yu, xv, yv = homography_jacobian(homography, u, v)
    # Source pixels per output pixel along each source axis.
    det = np.abs(xu * yv - xv * yu)
    across = np.maximum(det / np.hypot(yu, yv), 1.0).astype(int)
    down = np.maximum(det / np.hypot(xu, xv), 1.0).astype(int)
    return (int(across.min()), int(down.min())), (int(across.max()), int(down.max()))


def _reduced(
    image: NDArray[np.uint8],
    corners: Corners,
    reduction: tuple[int, int],
) -> tuple[NDArray[np.uint8], Corners]:
    """Box-reduce the part of `image` under `corners`, moving the corners along."""
    if reduction == (1, 1):
        return image, corners
    rx, ry = reduction
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    left = max(0, math.floor(min(xs)) - 2 * rx)
    top = max(0, math.floor(min(ys)) - 2 * ry)
    right = max(left, min(image.shape[1], math.ceil(max(xs)) + 2 * rx))
    bottom = max(top, min(image.shape[0], math.ceil(max(ys)) + 2 * ry))
    level = np.ascontiguousarray(image[top:bottom, left:right])
    if level.size:
        level = np.asarray(Image.fromarray(level).reduce(reduction))
    return level, [((x - left) / rx, (y - top) / ry) for x, y in corners]
