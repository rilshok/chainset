"""Images backed by an array, a file, a URL or a patch of another image."""

__all__ = [
    "FileRGBImage",
    "LoadedRGBImage",
    "PatchRGBImage",
    "RGBImage",
    "RGBImageJpeg",
    "RGBImagePng",
    "WebRGBImage",
]

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

import numpy as np
from iokit import Jpeg, Png, web
from iokit.dtype.extension import Extension
from iokit.state import Image as ImageFormatState
from numpy.typing import NDArray
from PIL import Image as PILImage

from chainset.dtype.points import Patch2D, PatchMapping
from chainset.utils.homography import quad_aspect_ratio, quad_lossless_height
from chainset.utils.image_antialiased_sampling import sample_quad_antialiased_uint8
from chainset.utils.image_patch_sampling import WHITE, FillValue, sample_quad_uint8
from chainset.utils.polygon_area import signed_area

RGBArray = NDArray[np.uint8]
PathLike = str | Path


# TODO(@rilshok): when touching the arr, calculate h and w and vice versa


def _assert_rgb_array(array: RGBArray) -> None:
    """Check that `array` holds RGB pixels in the layout the images expect.

    Args:
        array: The array to check.

    Raises:
        ValueError: If `array` is not `uint8` of shape `(height, width, 3)`.

    """
    if array.ndim != 3:
        msg = "RGB array must have 3 dimensions"
        raise ValueError(msg)
    if array.shape[2] != 3:
        msg = "RGB array must have 3 channels"
        raise ValueError(msg)
    if array.dtype != np.uint8:
        msg = "RGB array dtype must be uint8"
        raise ValueError(msg)


def _pil_to_rgb(image: PILImage.Image) -> RGBArray:
    """Convert a PIL image to an RGB array, flattening alpha onto white.

    Args:
        image: The image to convert, in any mode PIL can read.

    Returns:
        The pixels as `uint8` of shape `(height, width, 3)`.

    """
    if image.mode == "RGB":
        return np.array(image, dtype=np.uint8)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    background = PILImage.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    return np.array(background, dtype=np.uint8)


def _load_rgb_image_safe(buffer: PathLike | BinaryIO) -> RGBArray:
    """Load an image from a path or an open stream, as an RGB array.

    Args:
        buffer: Path to the file, or a binary stream positioned at its start.

    Returns:
        The pixels as `uint8` of shape `(height, width, 3)`, alpha flattened
        onto white.

    """
    with PILImage.open(buffer) as image:
        return _pil_to_rgb(image)


def _resize_array(array: RGBArray, width: int, height: int) -> RGBArray:
    """Resize an RGB array with Lanczos resampling.

    Args:
        array: The pixels to resize.
        width: Width of the result, in pixels.
        height: Height of the result, in pixels.

    Returns:
        The resized pixels, of shape `(height, width, 3)`.

    """
    image = PILImage.fromarray(array)
    image = image.resize((width, height), PILImage.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)


def _maybe_resize_array(
    array: RGBArray,
    width: int | None,
    height: int | None,
) -> RGBArray:
    """Resize an RGB array only if a size was asked for.

    Args:
        array: The pixels to resize.
        width: Width of the result, or `None` to derive it from `height`.
        height: Height of the result, or `None` to derive it from `width`.

    Returns:
        The pixels at the requested size, or `array` itself when neither side
        was given. Giving one side alone keeps the aspect ratio.

    Raises:
        SystemError: If both sides are `None` past the early return, which
            cannot happen.

    """
    if width is None and height is None:
        return array
    orig_height, orig_width = array.shape[:2]
    if width is None:
        if height is None:
            msg = "Both width and height are None after early return check"
            raise SystemError(msg)
        width = round(height * orig_width / orig_height)
    elif height is None:
        height = round(width * orig_height / orig_width)
    return _resize_array(array, width, height)


class RGBImage(ABC):
    """An image that can hand out RGB pixels, however it stores them.

    A subclass supplies `array` and the two size properties; everything else -
    encoding, cutting, rotating, displaying - is built on those.
    """

    @abstractmethod
    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        """Give the pixels of the image, resized if a size is asked for.

        Args:
            width: Width of the result, or `None` to derive it from `height`.
            height: Height of the result, or `None` to derive it from `width`.

        Returns:
            The pixels as `uint8` of shape `(height, width, 3)`. Leaving both
            sides out gives the image at its own size.

        Raises:
            NotImplementedError: If the subclass has not overridden this.

        """
        raise NotImplementedError

    @property
    def width(self) -> int:
        """Width of the image in pixels."""
        raise NotImplementedError

    @property
    def height(self) -> int:
        """Height of the image in pixels."""
        raise NotImplementedError

    @property
    def pil(self) -> PILImage.Image:
        """The image as a PIL image, at its own size."""
        return PILImage.fromarray(self.array())

    def to_jpeg(self, stem: str | None = None) -> Jpeg:
        """Encode the image as JPEG.

        Args:
            stem: File name without its extension, or `None` to leave it unset.

        Returns:
            The encoded image as an `iokit` state.

        """
        return Jpeg(self.pil, stem=stem)

    def to_png(self, stem: str | None = None) -> Png:
        """Encode the image as PNG.

        Args:
            stem: File name without its extension, or `None` to leave it unset.

        Returns:
            The encoded image as an `iokit` state.

        """
        return Png(self.pil, stem=stem)

    def _repr_html_(self) -> str:
        """Show the image inline in a notebook, as an embedded JPEG."""
        content = self.to_jpeg().data.base64
        return f'<img src="data:image/jpeg;base64,{content}" />'

    @property
    def loaded(self) -> "LoadedRGBImage":
        """The same image with its pixels realized and held in memory."""
        return LoadedRGBImage(self.array())

    def cut(
        self,
        patch: Patch2D,
        *,
        fill: FillValue = WHITE,
        mode: PatchMapping | str = PatchMapping.PERSPECTIVE,
    ) -> "PatchRGBImage":
        """Cut `patch` out of the image, warping it back to an upright rectangle.

        Args:
            patch: Region to cut, in coordinates relative to the image.
            fill: What to use where the patch reaches past the image: one
                level for all three channels, or an `(r, g, b)` color.
                Defaults to white.
            mode: The mapping, as a `PatchMapping` or its value.

        Returns:
            The patch as an image of its own, cut only once it is realized.

        Raises:
            ValueError: If `mode` is not a `PatchMapping`.

        """
        return PatchRGBImage(image=self, patch=patch, fill=fill, mode=mode)

    def rot90(self, k: int) -> "PatchRGBImage":
        """Rotate the image counter-clockwise by `k` quarter turns.

        Args:
            k: Number of quarter turns; taken modulo 4.

        Returns:
            A `PatchRGBImage` cut with the full frame patch whose corners are
            shifted by `k`.

        """
        return self.cut(Patch2D.from_xyxy(0.0, 0.0, 1.0, 1.0).shift(k))


class _RGBImageState(ImageFormatState[RGBImage]):
    """Base `iokit` state that carries an `RGBImage` in an image format."""

    def dump(self, data: RGBImage) -> PILImage.Image:
        """Hand the image to `iokit` for encoding.

        Args:
            data: The image to encode.

        Returns:
            The image as a PIL image.

        """
        return data.pil

    def parse(self, data: PILImage.Image) -> RGBImage:
        """Take a decoded image back from `iokit`.

        Args:
            data: The decoded image.

        Returns:
            The pixels as a `LoadedRGBImage`, alpha flattened onto white.

        """
        return LoadedRGBImage(_pil_to_rgb(data))


class RGBImageJpeg(_RGBImageState):
    """An `RGBImage` stored as JPEG."""

    __extension__ = Extension.JPEG


class RGBImagePng(_RGBImageState):
    """An `RGBImage` stored as PNG."""

    __extension__ = Extension.PNG


class LoadedRGBImage(RGBImage):
    """An image whose pixels are already in memory."""

    __slots__ = ("source",)

    def __init__(self, array: RGBArray) -> None:
        """Hold `array` as the pixels of the image, without copying it.

        Args:
            array: The pixels, `uint8` of shape `(height, width, 3)`.

        Raises:
            ValueError: If `array` is not in that layout.

        """
        _assert_rgb_array(array)
        self.source = array

    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        """Give a copy of the pixels, resized if a size is asked for.

        Args:
            width: Width of the result, or `None` to derive it from `height`.
            height: Height of the result, or `None` to derive it from `width`.

        Returns:
            The pixels as `uint8` of shape `(height, width, 3)`. The copy keeps
            a caller from writing through to the stored array.

        """
        if width is None and height is None:
            return self.source.copy()
        return _maybe_resize_array(
            array=self.source,
            width=width,
            height=height,
        )

    @property
    def width(self) -> int:
        """Width of the image in pixels."""
        return int(self.source.shape[1])

    @property
    def height(self) -> int:
        """Height of the image in pixels."""
        return int(self.source.shape[0])

    @property
    def loaded(self) -> "LoadedRGBImage":
        """The image itself, its pixels being loaded already."""
        return self


class FileRGBImage(RGBImage):
    """An image read from a file on demand, decoded afresh on every call."""

    __slots__ = ("_height", "_width", "source")

    def __init__(self, path: PathLike) -> None:
        """Point the image at a file, without opening it.

        Args:
            path: Path to the image file.

        """
        self.source = Path(path).as_posix()
        self._width: int | None = None
        self._height: int | None = None

    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        """Read the file and give its pixels, resized if a size is asked for.

        Args:
            width: Width of the result, or `None` to derive it from `height`.
            height: Height of the result, or `None` to derive it from `width`.

        Returns:
            The pixels as `uint8` of shape `(height, width, 3)`.

        """
        return _maybe_resize_array(
            array=_load_rgb_image_safe(self.source),
            width=width,
            height=height,
        )

    @property
    def width(self) -> int:
        """Width of the image in pixels, read from the file once and kept."""
        if self._width is None:
            shape = self.array().shape
            self._height = shape[0]
            self._width = shape[1]
        return self._width

    @property
    def height(self) -> int:
        """Height of the image in pixels, read from the file once and kept."""
        if self._height is None:
            shape = self.array().shape
            self._height = shape[0]
            self._width = shape[1]
        return self._height


class WebRGBImage(RGBImage):
    """An image fetched over the network on demand, refetched on every call."""

    __slots__ = ("_height", "_width", "source")

    def __init__(self, uri: str) -> None:
        """Point the image at a URI, without fetching it.

        Args:
            uri: Where the image lives, under the http, https or data scheme.

        Raises:
            ValueError: If `uri` uses another scheme.

        """
        if not uri.startswith(("http://", "https://", "data:")):
            msg = "WebImage uri must use the http, https or data scheme"
            raise ValueError(msg)
        self.source = uri
        self._width: int | None = None
        self._height: int | None = None

    def _fetch(self) -> RGBArray:
        """Fetch and decode the image, keeping its size for the size properties.

        Returns:
            The pixels as `uint8` of shape `(height, width, 3)`.

        """
        with web(self.source).buffer as buffer:
            array = _load_rgb_image_safe(buffer)
        self._height, self._width = array.shape[:2]
        return array

    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        """Fetch the image and give its pixels, resized if a size is asked for.

        Args:
            width: Width of the result, or `None` to derive it from `height`.
            height: Height of the result, or `None` to derive it from `width`.

        Returns:
            The pixels as `uint8` of shape `(height, width, 3)`.

        """
        return _maybe_resize_array(array=self._fetch(), width=width, height=height)

    @property
    def width(self) -> int:
        """Width of the image in pixels, fetched once and kept.

        Raises:
            SystemError: If the fetch left the size unknown.

        """
        if self._width is None:
            self._fetch()
        if self._width is None:
            msg = "Failed to determine image width after fetching"
            raise SystemError(msg)
        return self._width

    @property
    def height(self) -> int:
        """Height of the image in pixels, fetched once and kept.

        Raises:
            SystemError: If the fetch left the size unknown.

        """
        if self._height is None:
            self._fetch()
        if self._height is None:
            msg = "Failed to determine image height after fetching"
            raise SystemError(msg)
        return self._height


def _natural_size(
    patch: Patch2D,
    mode: PatchMapping,
    frame: tuple[int, int],
) -> tuple[float, float]:
    """Give the default width and height of a cut of `patch`, in pixels of `frame`."""
    match mode:
        case PatchMapping.PERSPECTIVE:
            # The proportions of the rectangle, fine enough to keep every source pixel,
            # but holding at most four times as many pixels as the patch covers.
            ratio = quad_aspect_ratio(patch.points, *frame)
            cap = math.sqrt(4 * abs(signed_area(patch.points)) / ratio)
            height = min(quad_lossless_height(patch.points, ratio), cap)
            return height * ratio, height
        case PatchMapping.BILINEAR:
            return patch.width_max, patch.height_max


def _patch_shape(
    natural: tuple[float, float],
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    """Pick the height and width of a cut, keeping the `natural` proportions."""
    across, down = natural
    if width is None:
        width = round(across if height is None else height * across / down)
        if height is None:
            height = round(down)
    elif height is None:
        height = round(width * down / across)
    return max(1, height), max(1, width)


class PatchRGBImage(RGBImage):
    """A region of another image, warped back to an upright rectangle.

    Nothing is cut until `array` is called, so a patch of a patch costs only
    the bookkeeping until one of them is realized.
    """

    __slots__ = ("fill", "mode", "patch", "source")

    def __init__(
        self,
        image: RGBImage,
        patch: Patch2D,
        fill: FillValue = WHITE,
        mode: PatchMapping | str = PatchMapping.PERSPECTIVE,
    ) -> None:
        """Name a region of `image` without cutting it.

        Args:
            image: The image to cut from.
            patch: Region to cut, in coordinates relative to `image`.
            fill: What to use where the patch reaches past the image: one level
                for all three channels, or an `(r, g, b)` color.
            mode: The mapping, as a `PatchMapping` or its value.

        Raises:
            ValueError: If `mode` is not a `PatchMapping`.

        """
        self.source = image
        self.patch = patch
        self.fill = fill
        self.mode = PatchMapping(mode)

    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        """Cut the patch out of the source image, warped back upright.

        Args:
            width: Width of the result; derived from `height` and the shape of
                the patch when left out.
            height: Height of the result; derived the same way from `width`.

        Returns:
            The patch as an RGB array.

        """
        source = self.source.loaded.source
        frame = (source.shape[1], source.shape[0])
        patch = self.patch.to_pixels(*frame)
        shape = _patch_shape(_natural_size(patch, self.mode, frame), width, height)
        match self.mode:
            case PatchMapping.PERSPECTIVE:
                return sample_quad_antialiased_uint8(source, patch.points, shape, fill=self.fill)
            case PatchMapping.BILINEAR:
                return sample_quad_uint8(source, patch.points, shape, fill=self.fill)

    @property
    def width(self) -> int:
        """Width of the cut in pixels, by default."""
        return self._shape()[1]

    @property
    def height(self) -> int:
        """Height of the cut in pixels, by default."""
        return self._shape()[0]

    def _shape(self) -> tuple[int, int]:
        """Give the default height and width of the cut."""
        frame = (self.source.width, self.source.height)
        patch = self.patch.to_pixels(*frame)
        return _patch_shape(_natural_size(patch, self.mode, frame), None, None)

    def rot90(self, k: int) -> "PatchRGBImage":
        """Rotate the cut counter-clockwise by `k` quarter turns.

        Args:
            k: Number of quarter turns; taken modulo 4.

        Returns:
            The same region of the same source, with its corner order shifted
            by `k`, rather than a patch stacked on top of this one.

        """
        return type(self)(
            image=self.source,
            patch=self.patch.shift(k),
            fill=self.fill,
            mode=self.mode,
        )
