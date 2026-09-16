"""Shapes defined by points."""

__all__ = [
    "Patch2D",
    "Patch2DDat",
]
import math
import struct
from collections.abc import Callable, Iterable
from typing import SupportsIndex

import numpy as np
from iokit import Dat
from numpy.typing import NDArray
from typing_extensions import Self

from chainset.utils.bilinear_quad import quad_meshgrid, quad_point_at, quad_point_of
from chainset.utils.contour_validity import is_valid_contour
from chainset.utils.polygon_area import quad_coverage, signed_area

Point = tuple[float, ...]


def _round(value: float) -> float:
    value = struct.unpack(">1f", struct.pack(">1f", float(value)))[0]
    return round(value, 5)


class PointSequence:
    """Ordered sequence of points sharing the same number of coordinates."""

    __slots__ = ("points",)

    points: list[Point]

    def __init__(self, points: Iterable[Iterable[float]], dim: int) -> None:
        """Store the points with every coordinate passed through `_round`.

        Args:
            points: Points of any count, each holding `dim` coordinates.
            dim: Number of coordinates every point must have.

        Raises:
            ValueError: If a point does not hold exactly `dim` coordinates.

        """
        self.points = [tuple(map(_round, point)) for point in points]
        for index, point in enumerate(self.points):
            if len(point) != dim:
                msg = f"Point {index} has {len(point)} coordinates, expected {dim}."
                raise ValueError(msg)

    def __eq__(self, other: object) -> bool:
        """Compare shapes of the same type point by point."""
        if type(other) is not type(self):
            return NotImplemented
        return self.points == other.points

    def __hash__(self) -> int:
        """Hash the type and the points, consistently with `__eq__`."""
        return hash((type(self), tuple(self.points)))

    def __repr__(self) -> str:
        """Represent the shape by its class name and points."""
        return f"{type(self).__name__}({self.points})"


class PointSequence2D(PointSequence):
    __slots__ = ()

    def __init__(self, points: Iterable[Iterable[float]]) -> None:
        super().__init__(points=points, dim=2)

    @classmethod
    def from_pixels(
        cls,
        points: Iterable[Iterable[float]],
        width: int,
        height: int,
    ) -> Self:
        """Build a shape from pixel coordinates, relative to the image size.

        Args:
            points: `(x, y)` points in pixels, in order.
            width: Width of the image the points were measured in.
            height: Height of the image the points were measured in.

        Returns:
            The same shape with points in `[0, 1]` for points inside the image.

        """
        shape = cls(points)
        return cls((x / width, y / height) for x, y in shape.points)

    def to_pixels(self, width: int, height: int) -> Self:
        """Scale the relative points back to pixel coordinates.

        Args:
            width: Width of the image to measure the points in.
            height: Height of the image to measure the points in.

        Returns:
            The same shape with points expressed in pixels.

        """
        return type(self)((x * width, y * height) for x, y in self.points)

    def translate(self, x: float = 0.0, y: float = 0.0) -> Self:
        """Move the shape by a linear offset along both axes at once.

        The offset is expressed in the shape's own coordinate frame, so for a
        shape in normalized image coordinates `x` and `y` are fractions of the
        image width and height.

        Args:
            x: Offset added to every point's x coordinate.
            y: Offset added to every point's y coordinate.

        Returns:
            The same shape with all points moved.

        """
        return type(self)((px + x, py + y) for px, py in self.points)

    @property
    def box(self) -> "Box2D":
        """Axis-aligned bounding box."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return Box2D(((min(xs), min(ys)), (max(xs), max(ys))))


class Box2D(PointSequence2D):
    __slots__ = ()

    def __init__(self, points: Iterable[Iterable[float]]) -> None:
        super().__init__(points)
        if len(self.points) != 2:
            msg = "Box2D requires exactly 2 points."
            raise ValueError(msg)

    @property
    def xmin(self) -> float:
        return self.points[0][0]

    @property
    def ymin(self) -> float:
        return self.points[0][1]

    @property
    def xmax(self) -> float:
        return self.points[1][0]

    @property
    def ymax(self) -> float:
        return self.points[1][1]

    @property
    def patch(self) -> "Patch2D":
        return Patch2D.from_xyxy(self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def box(self) -> Self:
        return self


class Patch2D(PointSequence2D):
    """Region of a plane bounded by four corners.

    The corners are stored in order and treated as a closed loop, so `p1`/`p2`
    and `p4`/`p3` span the width of the patch while `p2`/`p3` and `p1`/`p4`
    span its height. Coordinates are normally kept relative to the image the
    patch was taken from, which makes a patch independent of the image size.
    """

    __slots__ = ()

    def __init__(self, points: Iterable[Iterable[float]]) -> None:
        """Store the four corners, rounded to five decimal places.

        The corners have to trace a valid contour: a quadrilateral of non-zero
        area whose sides do not cross, so that the patch covers a region of the
        plane exactly once.

        Args:
            points: Exactly four `(x, y)` corners, in order.

        Raises:
            ValueError: If `points` does not hold four pairs of coordinates, or
                the corners do not trace a valid contour.

        """
        super().__init__(points)
        if len(self.points) != 4:
            msg = "Patch2D requires exactly 4 points."
            raise ValueError(msg)
        if not is_valid_contour(self.points):
            msg = "Patch2D requires corners that bound an area without crossing."
            raise ValueError(msg)

    @property
    def x1(self) -> float:
        """X coordinate of the first corner."""
        return self.points[0][0]

    @property
    def y1(self) -> float:
        """Y coordinate of the first corner."""
        return self.points[0][1]

    @property
    def x2(self) -> float:
        """X coordinate of the second corner."""
        return self.points[1][0]

    @property
    def y2(self) -> float:
        """Y coordinate of the second corner."""
        return self.points[1][1]

    @property
    def x3(self) -> float:
        """X coordinate of the third corner."""
        return self.points[2][0]

    @property
    def y3(self) -> float:
        """Y coordinate of the third corner."""
        return self.points[2][1]

    @property
    def x4(self) -> float:
        """X coordinate of the fourth corner."""
        return self.points[3][0]

    @property
    def y4(self) -> float:
        """Y coordinate of the fourth corner."""
        return self.points[3][1]

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

    @property
    def p1(self) -> Point:
        """First corner, as `[x, y]`."""
        return self.x1, self.y1

    @property
    def p2(self) -> Point:
        """Second corner, across the width from `p1`."""
        return self.x2, self.y2

    @property
    def p3(self) -> Point:
        """Third corner, opposite `p1`."""
        return self.x3, self.y3

    @property
    def p4(self) -> Point:
        """Fourth corner, across the height from `p1`."""
        return self.x4, self.y4

    def to_polygon(self) -> "Polygon2D":
        """Convert the patch into a quadrilateral polygon.

        Returns:
            A polygon tracing the four corners, from `p1` to `p4`.

        """
        return Polygon2D(self.points)

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

    def covered_by(self, other: "Patch2D") -> float:
        """Fraction of this patch's area that lies inside `other`.

        Both patches are arbitrary quadrilaterals, not bounding boxes.

        Args:
            other: The covering patch.

        Returns:
            A value in `[0, 1]`; `0.0` when this patch is degenerate.

        """
        return quad_coverage(self.points, other.points)

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
        return quad_meshgrid(self.points, width, height)

    def point_at(self, point: Point) -> Point:
        """Map a unit-square `point` into the `Patch2D` via bilinear interpolation.

        Args:
            point: Normalized coordinates in `[0, 1] x [0, 1]`.

        Returns:
            The interpolated point inside the `Patch2D`.

        """
        return quad_point_at(self.points, point)

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

        Inverse of `point_at`: finds the normalized coordinates that `point_at`
        would send to `point`.

        Args:
            point: Coordinates in the same frame as this `Patch2D`.

        Returns:
            Normalized coordinates in `[0, 1] x [0, 1]` for points inside
            the `Patch2D`, extrapolated outside otherwise.

        """
        return quad_point_of(self.points, point)

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
        return struct.pack(">8f", *(round(v * 100, 5) for p in data.points for v in p))

    def parse(self, data: bytes) -> Patch2D:
        """Rebuild a patch from its packed coordinates.

        Args:
            data: Bytes previously produced by `dump`.

        Returns:
            The patch the coordinates were taken from.

        """
        values = struct.unpack(">8f", data)
        return Patch2D((values[i] / 100, values[i + 1] / 100) for i in range(0, 8, 2))


class Polygon2D(PointSequence2D):
    """Closed polygon bounded by an ordered loop of vertices."""

    __slots__ = ()

    def __init__(self, points: Iterable[Iterable[float]]) -> None:
        """Store the vertices, rounded to five decimal places.

        A trailing vertex equal to the first one is treated as an explicit
        closing of the loop and dropped. The loop has to bound an object of
        non-zero area, keeping it on the same side all the way round; loops
        that meet at a point are allowed, loops that cross are not.

        Args:
            points: At least three `(x, y)` vertices, in loop order.

        Raises:
            ValueError: If a point is not a pair, fewer than three vertices
                remain, or the vertices do not trace a valid contour.

        """
        super().__init__(points)
        if len(self.points) > 1 and self.points[0] == self.points[-1]:
            self.points.pop()
        if len(self.points) < 3:
            msg = "Polygon2D requires at least 3 vertices."
            raise ValueError(msg)
        if not is_valid_contour(self.points):
            msg = "Polygon2D requires a contour that bounds an area without crossing itself."
            raise ValueError(msg)

    @property
    def area(self) -> float:
        """Area enclosed by the polygon, regardless of vertex order."""
        return abs(signed_area(self.points))
