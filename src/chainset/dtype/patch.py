"""Quadrilateral patches of a 2D plane."""

__all__ = [
    "Patch2D",
    "Patch2DDat",
]
import math
import struct
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from typing import SupportsIndex

import numpy as np
from iokit import Dat
from numpy.typing import NDArray
from typing_extensions import Self

Point = list[float]


_EPS = 1e-12


class Patch2D:
    """Region of a plane bounded by four corners.

    The corners are stored in order and treated as a closed loop, so `p1`/`p2`
    and `p4`/`p3` span the width of the patch while `p2`/`p3` and `p1`/`p4`
    span its height. Coordinates are normally kept relative to the image the
    patch was taken from, which makes a patch independent of the image size.
    """

    __slots__ = ("x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4")

    def __init__(self, points: Iterable[Iterable[float]]) -> None:
        """Store the four corners, rounded to five decimal places.

        Args:
            points: Exactly four `(x, y)` corners, in order.

        Raises:
            ValueError: If `points` does not hold four pairs of coordinates.

        """
        try:
            p1, p2, p3, p4 = points
            (
                self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.x3,
                self.y3,
                self.x4,
                self.y4,
            ) = (round(v, 5) for point in (p1, p2, p3, p4) for v in point)
        except ValueError as exc:
            msg = "Patch2D requires exactly 4 points of 2 coordinates each"
            raise ValueError(msg) from exc

    @classmethod
    def from_xyxy(cls, xmin: float, ymin: float, xmax: float, ymax: float) -> Self:
        """Build an axis-aligned patch from the bounds of a rectangle.

        Args:
            xmin: Lower bound along the x axis, shared by `p1` and `p4`.
            ymin: Lower bound along the y axis, shared by `p1` and `p2`.
            xmax: Upper bound along the x axis, shared by `p2` and `p3`.
            ymax: Upper bound along the y axis, shared by `p3` and `p4`.

        Returns:
            A patch with corners `(xmin, ymin)`, `(xmax, ymin)`, `(xmax, ymax)`,
            `(xmin, ymax)`.

        """
        return cls(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))

    @classmethod
    def from_pixels(
        cls,
        points: Iterable[Iterable[float]],
        width: int,
        height: int,
    ) -> Self:
        """Build a patch from pixel coordinates, relative to the image size.

        Args:
            points: Exactly four `(x, y)` corners in pixels, in order.
            width: Width of the image the corners were measured in.
            height: Height of the image the corners were measured in.

        Returns:
            The same patch with corners in `[0, 1]` for corners inside the image.

        """
        patch = cls(points)
        return cls(
            (
                (patch.x1 / width, patch.y1 / height),
                (patch.x2 / width, patch.y2 / height),
                (patch.x3 / width, patch.y3 / height),
                (patch.x4 / width, patch.y4 / height),
            ),
        )

    def to_pixels(self, width: int, height: int) -> Self:
        """Scale the relative corners back to pixel coordinates.

        Args:
            width: Width of the image to measure the corners in.
            height: Height of the image to measure the corners in.

        Returns:
            The same patch with corners expressed in pixels.

        """
        return type(self)(
            (
                (self.x1 * width, self.y1 * height),
                (self.x2 * width, self.y2 * height),
                (self.x3 * width, self.y3 * height),
                (self.x4 * width, self.y4 * height),
            ),
        )

    @property
    def p1(self) -> Point:
        """First corner, as `[x, y]`."""
        return [self.x1, self.y1]

    @property
    def p2(self) -> Point:
        """Second corner, across the width from `p1`."""
        return [self.x2, self.y2]

    @property
    def p3(self) -> Point:
        """Third corner, opposite `p1`."""
        return [self.x3, self.y3]

    @property
    def p4(self) -> Point:
        """Fourth corner, across the height from `p1`."""
        return [self.x4, self.y4]

    @property
    def points(self) -> list[Point]:
        """The four corners in order, from `p1` to `p4`."""
        return [self.p1, self.p2, self.p3, self.p4]

    def __repr__(self) -> str:
        """Represent Patch."""
        return f"Patch2D({self.points})"

    @property
    def width_max(self) -> float:
        """Length of the longer of the two edges spanning the width."""
        return max(math.dist(self.p1, self.p2), math.dist(self.p3, self.p4))

    @property
    def height_max(self) -> float:
        """Length of the longer of the two edges spanning the height."""
        return max(math.dist(self.p2, self.p3), math.dist(self.p4, self.p1))

    @property
    def shape_max(self) -> tuple[float, float]:
        """The `width_max` and `height_max` of the patch, as a pair."""
        return self.width_max, self.height_max

    @property
    def box(self) -> Self:
        """Axis-aligned bounding box of the patch, as a patch of its own."""
        xs = (self.x1, self.x2, self.x3, self.x4)
        ys = (self.y1, self.y2, self.y3, self.y4)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return type(self)(
            (
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            ),
        )

    def covered_by(self, other: "Patch2D") -> float:
        """Fraction of this patch's area that lies inside `other`.

        Both patches are arbitrary quadrilaterals, not bounding boxes. Clipping
        needs a convex window, so `other` is fanned into two triangles from its
        first corner. The fan is exact for a non-convex `other` too, as long as
        each triangle contributes with the sign of its winding: the part of the
        fan that falls outside `other` is then cancelled out.

        Args:
            other: The covering patch.

        Returns:
            A value in `[0, 1]`; `0.0` when this patch is degenerate.

        """
        polygon = self.points
        area = abs(_shoelace(polygon))
        if area <= _EPS:
            return 0.0

        a, b, c, d = other.points
        covered = 0.0
        for triangle in ((a, b, c), (a, c, d)):
            if abs(signed := _shoelace(triangle)) <= _EPS:
                continue
            window = triangle if signed > 0.0 else triangle[::-1]
            clipped = polygon
            for p, q in pairwise([*window, window[0]]):
                clipped = _clip(clipped, p, q)
                if not clipped:
                    break
            covered += math.copysign(abs(_shoelace(clipped)), signed)
        return min(abs(covered) / area, 1.0)

    def _apply(self, fn: Callable[[float], float]) -> Self:
        return type(self)(
            (
                (fn(self.x1), fn(self.y1)),
                (fn(self.x2), fn(self.y2)),
                (fn(self.x3), fn(self.y3)),
                (fn(self.x4), fn(self.y4)),
            ),
        )

    def round(self, ndigits: SupportsIndex | None = None) -> Self:
        """Round every coordinate of the patch.

        Args:
            ndigits: Decimal places to keep; `None` rounds to whole numbers.

        Returns:
            A new patch with all eight coordinates rounded.

        """
        return self._apply(lambda x: round(x, ndigits))

    def meshgrid(
        self,
        width: int,
        height: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Sample a grid of `height` by `width` cell centers spanning the patch.

        The grid follows the edges of the patch, so a warped patch yields a
        warped grid. Samples sit at cell centers, half a step away from the
        edges, which keeps them symmetric with respect to the patch.

        Args:
            width: Number of samples along the `p1` to `p2` direction.
            height: Number of samples along the `p1` to `p4` direction.

        Returns:
            The x and y coordinates of the samples as `float32`, each of shape
            `(height, width)`.

        """

        def _linspace(
            p1: Point | NDArray[np.float32],
            p2: Point | NDArray[np.float32],
            length: int,
        ) -> NDArray[np.float32]:
            points, step = np.linspace(
                p1,
                p2,
                length,
                endpoint=False,
                retstep=True,
                dtype=np.float32,
            )
            # `retstep` hands back a float64 step, which would upcast the sum.
            return points + np.asarray(step / 2, dtype=np.float32)

        xy = _linspace(
            _linspace(self.p1, self.p2, width),
            _linspace(self.p4, self.p3, width),
            height,
        )
        return xy[..., 0], xy[..., 1]

    def point_at(self, point: Point) -> Point:
        """Map a unit-square `point` into the `Patch2D` via bilinear interpolation.

        Args:
            point: Normalized coordinates in `[0, 1] x [0, 1]`.

        Returns:
            The interpolated point inside the `Patch2D`.

        """
        i, j = point
        a, b = 1.0 - i, 1.0 - j

        x = a * b * self.x1 + i * b * self.x2 + i * j * self.x3 + a * j * self.x4
        y = a * b * self.y1 + i * b * self.y2 + i * j * self.y3 + a * j * self.y4

        return [x, y]

    def project_into(self, glob: "Patch2D") -> Self:
        """Project this patch's corners into `glob` as normalized coordinates.

        Args:
            glob: Parent `Patch2D` whose frame the corners are mapped into.

        Returns:
            This patch expressed in `glob`'s coordinate frame.

        """
        return type(self)(
            [
                glob.point_at(self.p1),
                glob.point_at(self.p2),
                glob.point_at(self.p3),
                glob.point_at(self.p4),
            ],
        )

    def point_of(self, point: Point) -> Point:
        """Map a `point` of the `Patch2D` frame back to the unit square.

        Inverse of `point_at`: solves the bilinear map for the normalized
        coordinates that `point_at` would send to `point`.

        Args:
            point: Coordinates in the same frame as this `Patch2D`.

        Returns:
            Normalized coordinates in `[0, 1] x [0, 1]` for points inside
            the `Patch2D`, extrapolated outside otherwise.

        """
        x, y = point

        bx, by = self.x2 - self.x1, self.y2 - self.y1
        cx, cy = self.x4 - self.x1, self.y4 - self.y1
        dx = self.x1 - self.x2 + self.x3 - self.x4
        dy = self.y1 - self.y2 + self.y3 - self.y4
        qx, qy = x - self.x1, y - self.y1

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

        return [i, j]

    def project_from(self, glob: "Patch2D") -> Self:
        """Express this patch in `glob`'s normalized frame.

        Inverse of `project_into`: both patches must share a coordinate frame
        and `glob` is expected to surround this patch. Cutting `glob` out of an
        image and then cutting the result with the returned patch yields the
        same crop as cutting the original image with this patch.

        Args:
            glob: Surrounding `Patch2D` whose frame this patch is expressed in.

        Returns:
            This patch as normalized coordinates inside `glob`.

        """
        return type(self)(
            [
                glob.point_of(self.p1),
                glob.point_of(self.p2),
                glob.point_of(self.p3),
                glob.point_of(self.p4),
            ],
        )

    def translate(self, x: float = 0.0, y: float = 0.0) -> Self:
        """Move the patch by a linear offset along both axes at once.

        The offset is expressed in the patch's own coordinate frame, so for a
        patch in normalized image coordinates `x` and `y` are fractions of the
        image width and height.

        Args:
            x: Offset added to every corner's x coordinate.
            y: Offset added to every corner's y coordinate.

        Returns:
            A new `Patch2D` with all four corners moved.

        """
        return type(self)(
            (
                (self.x1 + x, self.y1 + y),
                (self.x2 + x, self.y2 + y),
                (self.x3 + x, self.y3 + y),
                (self.x4 + x, self.y4 + y),
            ),
        )

    def shift(self, k: int) -> Self:
        """Rotate the corner order, so that the patch starts from another corner.

        Args:
            k: Number of positions to rotate by, taken modulo 4.

        Returns:
            The same quadrilateral, with `p1` taken from the corner `k` steps
            further along the loop.

        """
        k %= 4
        points = self.points
        return type(self)(points[k:] + points[:k])


def _shoelace(polygon: Sequence[Point]) -> float:
    """Signed polygon area, positive when the vertices wind counter-clockwise.

    Args:
        polygon: Vertices of a closed polygon in order.

    Returns:
        The signed area, or `0.0` for fewer than three vertices.

    """
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        total += x0 * y1 - x1 * y0
        x0, y0 = x1, y1
    return total / 2.0


def _clip(polygon: Sequence[Point], p: Point, q: Point) -> list[Point]:
    """Clip a non-empty `polygon` to the left half-plane of the line `p -> q`.

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
            result.append([ax + (bx - ax) * step, ay + (by - ay) * step])
        if side >= 0.0:
            result.append(point)
        ax, ay, before = bx, by, side
    return result


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
    because a nearly affine patch makes `a` vanishingly small.

    Args:
        a: Quadratic coefficient; zero for an affine (non-warped) patch.
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


class Patch2DDat(Dat[Patch2D]):
    """Binary state holding a `Patch2D` as eight little-endian floats.

    The coordinates are scaled by 100 before packing, so the five decimal
    places a patch keeps survive the round trip through `float32`.
    """

    def dump(self, data: Patch2D) -> bytes:
        """Pack the corners of a patch into bytes.

        Args:
            data: The patch to serialize.

        Returns:
            The eight scaled coordinates, from `p1` to `p4`, as packed floats.

        """
        return struct.pack("<8f", *(round(v * 100, 5) for p in data.points for v in p))

    def parse(self, data: bytes) -> Patch2D:
        """Rebuild a patch from its packed coordinates.

        Args:
            data: Bytes previously produced by `dump`.

        Returns:
            The patch the coordinates were taken from.

        """
        values = struct.unpack("<8f", data)
        return Patch2D((values[i] / 100, values[i + 1] / 100) for i in range(0, 8, 2))
