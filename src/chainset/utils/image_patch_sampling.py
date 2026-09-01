"""Bilinear sampling of a 2-D image.

`sample_quad_uint8` cuts the quadrilateral named by four corners, which is
what warping a patch amounts to. Everything outside the image reads a
constant that defaults to white.

The quadrilateral maps onto the source bilinearly, so a source coordinate is
an affine ramp along every output row: a band of output rows derives its
coordinates from the corners in two operations, without building a grid. When
the corners describe an upright rectangle the two axes separate, and a band is
resampled one axis at a time, so every source row is blended across once and
then shared by the output rows that draw on it.
"""

__all__ = [
    "WHITE",
    "Corners",
    "FillValue",
    "sample_quad_uint8",
]

import math
from collections.abc import Iterator, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

SampleArray = NDArray[np.uint8] | NDArray[np.float32]
DifferenceDtype = np.dtype[np.int16] | np.dtype[np.float32]
Axis = tuple[NDArray[np.int32], NDArray[np.float32]]
Band = tuple[NDArray[np.generic], NDArray[np.float32]]
Coordinates = Iterator[tuple[int, int, NDArray[np.float32], NDArray[np.float32]]]

# The four corners of a quadrilateral in source pixels, in patch order.
Corners = Sequence[Sequence[float]]

# What to read outside the image: one level for every channel, or one each.
FillValue = float | Sequence[float]

# The default fill: a patch reaching past the image is padded with white.
WHITE = 255.0

_HALF = np.float32(0.5)
_CORNERS = 4
_GRID_NDIM = 3
_GRID_COORDS = 2
_UINT8_MIN = 0
_UINT8_MAX = 255

# Output cells per band: enough to amortize NumPy's per-call overhead, few
# enough to keep a band in cache.
_CELLS_PER_BAND = 16384


def _rows_per_band(out_width: int) -> int:
    """Choose how many output rows to handle at once."""
    return max(1, _CELLS_PER_BAND // max(out_width, 1))


def _fill_levels(
    fill: FillValue,
    channels: int,
    dtype: np.dtype[np.generic],
) -> NDArray[np.float64]:
    """Spread `fill` over the channels, at the precision the image can hold."""
    levels = np.atleast_1d(np.asarray(fill, dtype=np.float64))
    if levels.ndim != 1 or levels.size not in (1, channels):
        msg = "fill must be a single level or one level per channel"
        raise ValueError(msg)
    if dtype == np.uint8:
        if levels.min() < _UINT8_MIN or levels.max() > _UINT8_MAX:
            msg = "fill levels must lie between 0 and 255 for a uint8 image"
            raise ValueError(msg)
        levels = np.floor(levels + _HALF)
    return np.broadcast_to(levels, (channels,))


def _bordered_planes(image: SampleArray, levels: NDArray[np.float64]) -> SampleArray:
    """Copy `image` into planes ringed by the fill, one plane per channel.

    The ring is one pixel on the top and left and two on the bottom and right,
    so a clamped coordinate can address all four of its neighbors unchecked.
    """
    height, width, channels = image.shape
    bordered = np.empty((height + 3, width + 3, channels), dtype=image.dtype)
    bordered[0] = levels
    bordered[height + 1 :] = levels
    bordered[1 : height + 1, 0] = levels
    bordered[1 : height + 1, width + 1 :] = levels
    bordered[1 : height + 1, 1 : width + 1] = image
    planes: SampleArray = np.ascontiguousarray(bordered.transpose(2, 0, 1))
    return planes


def _locate(pixels: NDArray[np.float32], size: int) -> Axis:
    """Bracket pixel coordinates on a `size`-long axis of a bordered plane.

    Overwrites `pixels`. Clamping to one pixel outside the image leaves a
    coordinate that is further out on the border with a zero fraction, so it
    reads the fill and nothing else.
    """
    np.clip(pixels, -1.0, size, out=pixels)
    floor = np.floor(pixels)
    fraction = pixels - floor
    index = floor.astype(np.int32)
    index += 1
    return index, fraction


def _lerp(
    start: NDArray[np.generic],
    end: NDArray[np.generic],
    fraction: NDArray[np.float32],
    difference: DifferenceDtype,
) -> NDArray[np.float32]:
    """Walk `fraction` of the way from `start` to `end`.

    `start + (end - start) * fraction` rounds once less than a weighted sum,
    returns the ends of the interval untouched, and for a `uint8` image takes
    the difference in exact integer arithmetic.
    """
    walked: NDArray[np.float32] = np.subtract(end, start, dtype=difference) * fraction
    walked += cast("NDArray[np.float32]", start)
    return walked


def _ramp(length: int) -> NDArray[np.float64]:
    """Give the cell centers of a `length`-long axis, as fractions of it."""
    return (np.arange(length, dtype=np.float64) + 0.5) / length


def _quad_coordinates(corners: Corners, shape: tuple[int, int], rows: int) -> Coordinates:
    """Yield the source coordinates of each band of output rows of a quadrilateral.

    A corner pair fixes the source point at each end of an output row, so the
    coordinates along a row are an affine ramp and a whole band costs one
    multiply and one add per axis.
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = ((float(x), float(y)) for x, y in corners)
    out_height, out_width = shape
    across = _ramp(out_width)
    down = _ramp(out_height).astype(np.float32)

    starts = []
    slopes = []
    for near, along, down_near, opposite in ((x1, x2, x4, x3), (y1, y2, y4, y3)):
        starts.append((near + (along - near) * across).astype(np.float32))
        slope = (down_near - near) + ((opposite - down_near) - (along - near)) * across
        slopes.append(slope.astype(np.float32))

    for first in range(0, out_height, rows):
        stop = min(first + rows, out_height)
        band = down[first:stop, None]
        yield first, stop, starts[0] + band * slopes[0], starts[1] + band * slopes[1]


def _scattered_bands(
    planes: SampleArray,
    coordinates: Coordinates,
    out: NDArray[np.generic],
    difference: DifferenceDtype,
) -> Iterator[Band]:
    """Blend the four neighbors of every output cell, a band of rows at a time."""
    channels, bordered_height, stride = planes.shape
    flat = planes.reshape(channels, -1)
    last = flat.shape[1] - stride - 2
    for first, stop, x, y in coordinates:
        column, dx = _locate(x, stride - 3)
        row, dy = _locate(y, bordered_height - 3)
        row *= np.int32(stride)
        row += column
        np.clip(row, 0, last, out=row)

        north = _lerp(np.take(flat, row, axis=1), np.take(flat, row + 1, axis=1), dx, difference)
        row += stride
        south = _lerp(np.take(flat, row, axis=1), np.take(flat, row + 1, axis=1), dx, difference)
        south -= north
        south *= dy
        south += north
        yield out[first:stop], south


def _upright_bands(
    planes: SampleArray,
    columns: Axis,
    rows: Axis,
    out: NDArray[np.generic],
    difference: DifferenceDtype,
) -> Iterator[Band]:
    """Resample an upright cut one axis at a time, a band of output rows at a time."""
    column, dx = columns
    row, dy = rows
    for first in range(0, out.shape[0], _rows_per_band(out.shape[1])):
        stop = min(first + _rows_per_band(out.shape[1]), out.shape[0])
        band_rows = row[first:stop]
        top = int(band_rows.min())
        band = planes[:, top : int(band_rows.max()) + 2, :]
        middle = _lerp(band[..., column], band[..., column + 1], dx, difference)
        near = band_rows - top
        sampled = _lerp(
            np.take(middle, near, axis=1),
            np.take(middle, near + 1, axis=1),
            dy[first:stop, None],
            np.dtype(np.float32),
        )
        yield out[first:stop], sampled


def _collect(
    bands: Iterator[Band],
    out: NDArray[np.generic],
    source_dtype: np.dtype[np.generic],
) -> NDArray[np.generic]:
    """Quantize each band and scatter its planes into the interleaved output."""
    quantizing = out.dtype == np.uint8
    # A blend cannot leave the range of the source pixels, so only a float
    # source, whose own values may sit outside it, needs clipping.
    clipping = quantizing and source_dtype != np.uint8
    for target, band in bands:
        if clipping:
            np.clip(band, _UINT8_MIN, _UINT8_MAX, out=band)
        if quantizing:
            np.add(band, _HALF, out=band)
        # Scattering plane by plane beats transposing the band, because a
        # three-byte interleaved copy is a slow path in NumPy.
        for channel in range(band.shape[0]):
            target[..., channel] = band[channel]
    return out


def _prepare(
    image: SampleArray,
    shape: tuple[int, int],
    fill: FillValue,
    dtype: np.dtype[np.generic],
) -> tuple[NDArray[np.generic], SampleArray | None, DifferenceDtype]:
    """Allocate the output and the bordered planes, or say the output is all fill."""
    if image.ndim != _GRID_NDIM:
        msg = "image must have shape (height, width, channels)"
        raise ValueError(msg)
    if image.dtype not in (np.uint8, np.float32):
        msg = "image dtype must be uint8 or float32"
        raise ValueError(msg)
    channels = image.shape[2]
    levels = _fill_levels(fill, channels, image.dtype)
    difference: DifferenceDtype = (
        np.dtype(np.int16) if image.dtype == np.uint8 else np.dtype(np.float32)
    )
    if image.shape[0] == 0 or image.shape[1] == 0 or shape[0] == 0 or shape[1] == 0:
        return np.full((*shape, channels), levels, dtype=dtype), None, difference
    out = np.empty((*shape, channels), dtype=dtype)
    return out, _bordered_planes(image, levels), difference


def _upright(corners: Corners) -> bool:
    """Tell whether the quadrilateral is an upright rectangle, possibly flipped."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corners
    return x1 == x4 and x2 == x3 and y1 == y2 and y3 == y4


def _turned(corners: Corners) -> bool:
    """Tell whether the quadrilateral is upright once the two axes are swapped."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corners
    return x1 == x2 and x3 == x4 and y1 == y4 and y2 == y3


def _crop(image: SampleArray, corners: Corners) -> tuple[SampleArray, Corners]:
    """Narrow the source to the pixels the quadrilateral can reach.

    `corners` holds corner coordinates: `(0, 0)` is the top left of the image,
    so the center of the first pixel lies at `(0.5, 0.5)`. The returned corners
    are moved into the crop and onto pixel centers, which is what the sampler
    reads.
    """
    height, width = image.shape[0], image.shape[1]
    xs = [float(corner[0]) for corner in corners]
    ys = [float(corner[1]) for corner in corners]
    left = max(0, math.floor(min(xs)) - 1)
    right = max(left, min(width, math.ceil(max(xs)) + 2))
    top = max(0, math.floor(min(ys)) - 1)
    bottom = max(top, min(height, math.ceil(max(ys)) + 2))
    moved = [(x - left - 0.5, y - top - 0.5) for x, y in zip(xs, ys, strict=True)]
    return image[top:bottom, left:right], moved


def sample_quad_uint8(
    image: SampleArray,
    corners: Corners,
    shape: tuple[int, int],
    *,
    fill: FillValue = WHITE,
) -> NDArray[np.uint8]:
    """Cut the quadrilateral named by `corners` out of `image`, warped upright.

    The corners run around the quadrilateral, `corners[0]` landing on the top
    left of the result and `corners[1]` on its top right. Output cells sample
    the bilinear map between them, so an upright rectangle is a plain resize
    and a turned or skewed one is warped back into a rectangle.

    Args:
        image: Source image of shape `(height, width, channels)`, `uint8` or
            `float32`, possibly a non-contiguous view.
        corners: The four corners in patch order, in source pixels measured
            from the top left, so the first pixel is centered on `(0.5, 0.5)`.
        shape: Height and width of the result.
        fill: What to read outside the image, one level or one per channel.

    Returns:
        The cut as `uint8`, of shape `(*shape, channels)`.

    Raises:
        ValueError: If a shape, the image dtype, `corners` or `fill` is unusable.

    """
    if len(corners) != _CORNERS or any(len(corner) != _GRID_COORDS for corner in corners):
        msg = "corners must be four (x, y) points"
        raise ValueError(msg)

    if _turned(corners) and not _upright(corners):
        first, second, third, fourth = corners
        turned = sample_quad_uint8(
            image,
            (first, fourth, third, second),
            (shape[1], shape[0]),
            fill=fill,
        )
        return np.ascontiguousarray(turned.transpose(1, 0, 2))

    cropped, moved = _crop(image, corners)
    out, planes, difference = _prepare(cropped, shape, fill, np.dtype(np.uint8))
    if planes is None:
        return cast("NDArray[np.uint8]", out)

    if _upright(moved):
        (x1, y1), (x2, _), _, (_, y4) = moved
        columns = _locate((x1 + (x2 - x1) * _ramp(shape[1])).astype(np.float32), cropped.shape[1])
        rows = _locate((y1 + (y4 - y1) * _ramp(shape[0])).astype(np.float32), cropped.shape[0])
        bands = _upright_bands(planes, columns, rows, out, difference)
    else:
        coordinates = _quad_coordinates(moved, shape, _rows_per_band(shape[1]))
        bands = _scattered_bands(planes, coordinates, out, difference)
    return cast("NDArray[np.uint8]", _collect(bands, out, cropped.dtype))
