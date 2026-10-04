"""Tests for cutting patches out of images."""

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from chainset.dtype.image import LoadedRGBImage, PatchRGBImage
from chainset.dtype.points import Patch2D, PatchMapping
from chainset.utils.homography import quad_homography
from chainset.utils.image_antialiased_sampling import sample_quad_antialiased_uint8
from chainset.utils.image_patch_sampling import sample_quad_uint8
from chainset.utils.polygon_area import signed_area
from tests.camera import view_of_rectangle

FRAME = (800, 600)

# Width and height of the flat picture the camera photographs.
FLAT = (600, 300)

# Distance between the lines of the flat picture.
SPACING = 60

# Corners of the flat picture in the photograph: a 2:1 rectangle turned away.
VIEW = view_of_rectangle((2.0, 1.0), (0.7, 0.4), focal=700.0, frame=FRAME)


def _photograph(picture: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Render `picture` at `VIEW`, supersampled four times."""
    width, height = FLAT
    onto_photo = quad_homography(VIEW) @ np.diag([1.0 / width, 1.0 / height, 1.0])
    into_picture = np.linalg.inv(onto_photo) @ np.diag([0.25, 0.25, 1.0])
    coefficients = tuple((into_picture / into_picture[2, 2]).flat)[:8]
    channels = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(picture[..., channel]))
            .transform((3200, 2400), Image.Transform.PERSPECTIVE, coefficients, fillcolor=255.0)
            .resize(FRAME, Image.Resampling.LANCZOS),
        )
        for channel in range(3)
    ]
    return np.clip(np.stack(channels, axis=-1) + 0.5, 0.0, 255.0).astype(np.uint8)


@pytest.fixture(scope="module")
def picture() -> NDArray[np.float32]:
    """Give a light picture crossed by dark lines every `SPACING` pixels."""
    width, height = FLAT
    picture = np.full((height, width, 3), 230.0, dtype=np.float32)
    for offset in (0, 1):
        picture[:, offset::SPACING] = 20.0
        picture[offset::SPACING, :] = 20.0
    return picture


@pytest.fixture(scope="module")
def photo(picture: NDArray[np.float32]) -> LoadedRGBImage:
    """Give the picture photographed at an angle."""
    return LoadedRGBImage(_photograph(picture))


@pytest.fixture
def noise() -> NDArray[np.uint8]:
    """Give a small image of random pixels."""
    return np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)


def _psnr(result: NDArray[np.uint8], expected: NDArray[np.generic]) -> float:
    """Give the peak signal-to-noise ratio of `result`, in decibels."""
    error = np.mean((result.astype(np.float64) - expected.astype(np.float64)) ** 2)
    return float(10.0 * np.log10(255.0**2 / error))


def test_mode_defaults_to_perspective_and_accepts_values(noise: NDArray[np.uint8]) -> None:
    """A mode is a `PatchMapping` or its value, perspective by default."""
    image, patch = LoadedRGBImage(noise), Patch2D.from_xyxy(0.1, 0.1, 0.9, 0.9)
    assert image.cut(patch).mode is PatchMapping.PERSPECTIVE
    assert image.cut(patch, mode="bilinear").mode is PatchMapping.BILINEAR
    bilinear = patch.point_at((0.3, 0.6), mode=PatchMapping.BILINEAR)
    assert patch.point_at((0.3, 0.6), mode="bilinear") == bilinear
    with pytest.raises(ValueError, match="PatchMapping"):
        image.cut(patch, mode="affine")


def test_perspective_cut_restores_the_picture(
    photo: LoadedRGBImage,
    picture: NDArray[np.float32],
) -> None:
    """Lines evenly spaced on the plane come out evenly spaced, unlike bilinearly."""
    patch = Patch2D.from_pixels(VIEW, *FRAME)
    width, height = FLAT
    perspective = photo.cut(patch).array(width=width, height=height)
    bilinear = photo.cut(patch, mode=PatchMapping.BILINEAR).array(width=width, height=height)

    dark = np.flatnonzero(perspective[height // 2, :, 0] < 120)
    lines = [run.mean() for run in np.split(dark, np.flatnonzero(np.diff(dark) > 2) + 1)]
    assert lines == pytest.approx([SPACING * index + 0.5 for index in range(10)], abs=1.0)
    # The steep view keeps fewer rows than the picture has, which caps the match.
    assert _psnr(perspective, picture) > 22.0
    assert _psnr(perspective, picture) > _psnr(bilinear, picture) + 10.0


def test_perspective_cut_keeps_proportions(photo: LoadedRGBImage) -> None:
    """By default the cut keeps the proportions of the rectangle and every source pixel."""
    cut = photo.cut(Patch2D.from_pixels(VIEW, *FRAME))
    height, width = cut.array().shape[:2]
    assert (cut.height, cut.width) == (height, width)
    assert width / height == pytest.approx(2.0, abs=0.01)
    assert abs(signed_area(VIEW)) < width * height <= 4 * abs(signed_area(VIEW))
    assert cut.array(width=300).shape == (150, 300, 3)
    assert cut.array(height=100).shape == (100, 200, 3)


def test_perspective_cut_is_free_of_aliasing() -> None:
    """Shrinking a one-pixel checkerboard gives an even grey rather than moire."""
    checkerboard = (np.indices((800, 800)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = LoadedRGBImage(np.repeat(checkerboard[..., None], 3, axis=-1))
    corners = [(100.0, 100.0), (700.0, 150.0), (650.0, 700.0), (120.0, 600.0)]
    patch = Patch2D.from_pixels(corners, 800, 800)
    perspective = image.cut(patch).array(width=101, height=97)
    bilinear = image.cut(patch, mode=PatchMapping.BILINEAR).array(width=101, height=97)
    assert perspective.mean() == pytest.approx(127.5, abs=1.0)
    assert perspective.std() < 2.0
    assert bilinear.std() > 20.0


def test_shrinking_matches_a_plain_resize() -> None:
    """Shrinking a cut filters it as well as resizing the image does."""
    rows, columns = np.indices((300, 400))
    smooth = 127.5 + 60.0 * np.sin(columns / 7.0) * np.cos(rows / 11.0)
    pixels = np.repeat(smooth[..., None], 3, axis=-1).astype(np.uint8)
    result = LoadedRGBImage(pixels).cut(Patch2D.from_xyxy(0, 0, 1, 1)).array(width=40, height=30)
    expected = np.asarray(Image.fromarray(pixels).resize((40, 30), Image.Resampling.LANCZOS))
    assert _psnr(result, expected) > 35.0


@pytest.mark.parametrize("mode", list(PatchMapping))
@pytest.mark.parametrize("turns", [0, 1, 2, 3, 5, -1])
def test_rot90_is_exact(noise: NDArray[np.uint8], mode: PatchMapping, turns: int) -> None:
    """Quarter turns move pixels without resampling them."""
    image = LoadedRGBImage(noise)
    rotated = PatchRGBImage(image, Patch2D.from_xyxy(0, 0, 1, 1), mode=mode).rot90(turns)
    assert rotated.mode is mode
    assert np.array_equal(rotated.array(), np.rot90(noise, turns))
    assert np.array_equal(image.rot90(turns).array(), np.rot90(noise, turns))


@pytest.mark.parametrize("mode", list(PatchMapping))
def test_upright_cut_at_own_size_is_a_slice(noise: NDArray[np.uint8], mode: PatchMapping) -> None:
    """An upright rectangle on pixel boundaries cuts out exactly those pixels."""
    patch = Patch2D.from_pixels([(8, 4), (40, 4), (40, 36), (8, 36)], 64, 48)
    assert np.array_equal(LoadedRGBImage(noise).cut(patch, mode=mode).array(), noise[4:36, 8:40])


def test_perspective_cut_reads_fill_outside(noise: NDArray[np.uint8]) -> None:
    """Where the patch reaches past the image, the cut reads the fill."""
    image = LoadedRGBImage(noise)
    result = image.cut(Patch2D.from_xyxy(-0.5, -0.5, 0.5, 0.5), fill=(255, 0, 0)).array()
    assert np.array_equal(result[:20, :28], np.broadcast_to([255, 0, 0], (20, 28, 3)))
    assert np.array_equal(result[24:, 32:], noise[:24, :32])
    outside = Patch2D([(2.0, 2.0), (3.0, 2.2), (3.1, 3.0), (2.0, 2.9)])
    result = image.cut(outside, fill=(10, 20, 30)).array(width=7, height=5)
    assert np.array_equal(result, np.broadcast_to([10, 20, 30], (5, 7, 3)))


def test_bilinear_mode_samples_as_before(photo: LoadedRGBImage) -> None:
    """The bilinear mode keeps the previous sampling and size."""
    patch = Patch2D.from_pixels(VIEW, *FRAME)
    pixels = patch.to_pixels(*FRAME)
    cut = photo.cut(patch, mode=PatchMapping.BILINEAR)
    expected = sample_quad_uint8(photo.array(), pixels.points, (123, 245))
    assert np.array_equal(cut.array(width=245, height=123), expected)
    assert (cut.width, cut.height) == (round(pixels.width_max), round(pixels.height_max))


def test_cut_of_cut_matches_direct_cut(photo: LoadedRGBImage) -> None:
    """Cutting a patch out of a cut matches cutting it directly."""
    glob = Patch2D.from_pixels(VIEW, *FRAME)
    corners = [(380.0, 190.0), (600.0, 170.0), (610.0, 300.0), (400.0, 310.0)]
    local = Patch2D.from_pixels(corners, *FRAME)
    direct = photo.cut(local).array(width=200, height=120)
    nested = photo.cut(glob).loaded.cut(local.project_from(glob)).array(width=200, height=120)
    assert _psnr(nested, direct) > 25.0


def test_sampler_handles_an_empty_shape(noise: NDArray[np.uint8]) -> None:
    """An empty shape gives an empty cut."""
    corners = [(4.0, 4.0), (40.0, 6.0), (38.0, 30.0), (6.0, 28.0)]
    assert sample_quad_antialiased_uint8(noise, corners, (0, 5)).shape == (0, 5, 3)
