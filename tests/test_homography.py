"""Tests for the projective map between the unit square and a quadrilateral."""

import math

import pytest

from chainset.dtype.points import Patch2D, PatchMapping
from chainset.utils.bilinear_quad import quad_point_at
from chainset.utils.homography import (
    homography_point_at,
    homography_point_of,
    quad_aspect_ratio,
    quad_lossless_height,
)
from tests.camera import view_of_rectangle

Quad = list[tuple[float, float]]

# A quadrilateral seen at a steep angle, no two of its edges parallel.
TILTED: Quad = [(150.0, 120.0), (650.0, 60.0), (700.0, 560.0), (90.0, 420.0)]

# A parallelogram, on which the projective and the bilinear map agree.
PARALLELOGRAM: Quad = [(10.0, 20.0), (110.0, 40.0), (130.0, 140.0), (30.0, 120.0)]

UNIT_CORNERS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


@pytest.mark.parametrize("quad", [TILTED, PARALLELOGRAM])
def test_map_sends_corners_to_corners_and_back(quad: Quad) -> None:
    """The corners of the square land on the corners, and every point maps back."""
    for unit, corner in zip(UNIT_CORNERS, quad, strict=True):
        assert homography_point_at(quad, unit) == pytest.approx(corner)
    for point in [(0.3, 0.7), (-0.2, 1.4), (0.9, 0.05)]:
        there = homography_point_at(quad, point)
        assert homography_point_of(quad, there) == pytest.approx(point, abs=1e-12)


def test_parallelogram_maps_like_bilinear() -> None:
    """On a parallelogram the projective map is the bilinear one."""
    for point in [(0.3, 0.6), (0.0, 0.9), (0.75, 0.25)]:
        expected = quad_point_at(PARALLELOGRAM, point)
        assert homography_point_at(PARALLELOGRAM, point) == pytest.approx(expected)


def test_straight_lines_stay_straight() -> None:
    """Points along a diagonal of the square stay on one line, unlike bilinearly."""
    (x0, y0), *inner, (x1, y1) = [homography_point_at(TILTED, (t, t)) for t in (0, 0.3, 0.6, 1)]
    for x, y in inner:
        assert (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("size", "turn"),
    [((2.0, 1.0), (0.6, 0.3)), ((1.0, 1.414), (-0.9, 0.5)), ((3.0, 1.0), (0.2, -1.0))],
)
def test_aspect_ratio_of_a_view(
    size: tuple[float, float],
    turn: tuple[float, float],
) -> None:
    """The proportions of a photographed rectangle come back from its four corners."""
    quad = view_of_rectangle(size, turn, focal=800.0, frame=(1280, 720))
    assert quad_aspect_ratio(quad, 1280, 720) == pytest.approx(size[0] / size[1], rel=1e-9)


def test_aspect_ratio_of_a_parallelogram() -> None:
    """A parallelogram keeps the ratio of its edges."""
    expected = math.hypot(100.0, 20.0) / math.hypot(20.0, 100.0)
    assert quad_aspect_ratio(PARALLELOGRAM, 1280, 720) == pytest.approx(expected)


def test_aspect_ratio_with_a_normal_lens() -> None:
    """With one vanishing point at infinity the focal length is taken for the diagonal."""
    frame = (1280, 720)
    quad = view_of_rectangle((2.0, 1.0), (0.7, 0.0), focal=math.hypot(*frame), frame=frame)
    assert quad_aspect_ratio(quad, *frame) == pytest.approx(2.0, rel=1e-6)


def test_lossless_height() -> None:
    """An upright rectangle keeps every pixel at its own height, a tilted one needs more."""
    rectangle = [(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)]
    assert quad_lossless_height(rectangle, 640 / 480) == pytest.approx(480.0)
    trapezoid = [(200.0, 0.0), (440.0, 0.0), (640.0, 480.0), (0.0, 480.0)]
    assert quad_lossless_height(trapezoid, 640 / 480) > 480.0


@pytest.mark.parametrize("mode", list(PatchMapping))
def test_meshgrid_follows_the_mapping(mode: PatchMapping) -> None:
    """Every grid point is where `point_at` sends the cell center."""
    patch = Patch2D(TILTED)
    xs, ys = patch.meshgrid(4, 3, mode=mode)
    for row in range(3):
        for column in range(4):
            expected = patch.point_at(((column + 0.5) / 4, (row + 0.5) / 3), mode=mode)
            assert (xs[row, column], ys[row, column]) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("mode", list(PatchMapping))
def test_patch_projections_invert_each_other(mode: PatchMapping) -> None:
    """Projecting a patch into another and back returns it."""
    glob = Patch2D(TILTED)
    local = Patch2D([(0.1, 0.2), (0.8, 0.1), (0.9, 0.7), (0.2, 0.9)])
    back = local.project_into(glob, mode=mode).project_from(glob, mode=mode)
    for corner, expected in zip(back.points, local.points, strict=True):
        assert corner == pytest.approx(expected, abs=1e-4)


def test_patch_maps_in_perspective_by_default() -> None:
    """A patch maps the square projectively unless told otherwise."""
    patch = Patch2D(TILTED)
    assert patch.point_at((0.5, 0.5)) == homography_point_at(TILTED, (0.5, 0.5))
    assert patch.point_at((0.5, 0.5), mode=PatchMapping.BILINEAR) == quad_point_at(
        TILTED,
        (0.5, 0.5),
    )
