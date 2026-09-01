"""Images backed by an array, a file, a URL or a patch of another image."""

import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

import numpy as np
from iokit import Data, Jpeg, Png
from iokit.dtype.extension import Extension
from iokit.state import Image as ImageFormatState
from numpy.typing import NDArray
from PIL import Image as PILImage

from chainset.dtype.patch import Patch2D
from chainset.utils.image_patch_sampling import WHITE, FillValue, sample_quad_uint8

RGBArray = NDArray[np.uint8]
PathLike = str | Path


# TODO(@rilshok): when touching the arr, calculate h and w and vice versa


def _normalize_extension(extension: str | Extension) -> Extension:
    """Read an image extension, with or without its leading dot.

    Args:
        extension: The extension to read, such as `"png"` or `".png"`.

    Returns:
        The matching `Extension`.

    Raises:
        NotImplementedError: If `extension` names anything but `.jpeg` or `.png`.

    """
    if isinstance(extension, str):
        extension = f".{extension.removeprefix('.')}"
        extension = Extension(extension)
    if extension not in {Extension.JPEG, Extension.PNG}:
        raise NotImplementedError
    return extension


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


def load_image_safe(buffer: PathLike | BinaryIO) -> RGBArray:
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

    def data(self, extension: str | Extension) -> Data:
        """Encode the image in the format `extension` names.

        Args:
            extension: The format to encode in, `.jpeg` or `.png`.

        Returns:
            The encoded bytes.

        Raises:
            NotImplementedError: If `extension` names another format.

        """
        match _normalize_extension(extension):
            case Extension.JPEG | Extension.JPG:
                return self.to_jpeg().data
            case Extension.PNG:
                return self.to_png().data
            case _:
                raise NotImplementedError

    def _repr_html_(self) -> str:
        """Show the image inline in a notebook, as an embedded JPEG."""
        content = self.data(Extension.JPEG).base64
        return f'<img src="data:image/jpeg;base64,{content}" />'

    @property
    def loaded(self) -> "LoadedRGBImage":
        """The same image with its pixels realized and held in memory."""
        return LoadedRGBImage(self.array())

    def cut(self, patch: Patch2D, *, fill: FillValue = WHITE) -> "PatchRGBImage":
        """Cut `patch` out of the image, warping it back to an upright rectangle.

        Args:
            patch: Region to cut, in coordinates relative to the image.
            fill: What to use where the patch reaches past the image: one
                level for all three channels, or an `(r, g, b)` color.
                Defaults to white.

        Returns:
            The patch as an image of its own, cut only once it is realized.

        """
        return PatchRGBImage(image=self, patch=patch, fill=fill)

    def rot90(self, k: int) -> "PatchRGBImage":
        """Rotate the image counter-clockwise by `k` quarter turns.

        Args:
            k: Number of quarter turns; taken modulo 4.

        Returns:
            A `PatchRGBImage` cut with the full frame patch whose corners are
            shifted by `k`.

        """
        return self.cut(Patch2D.from_xyxy(0.0, 0.0, 1.0, 1.0).shift(k))


class RGBImageState(ImageFormatState[RGBImage]):
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


class RGBImageJpeg(RGBImageState):
    """An `RGBImage` stored as JPEG."""

    __extension__ = Extension.JPEG


class RGBImagePng(RGBImageState):
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
        return self.source.shape[1]

    @property
    def height(self) -> int:
        """Height of the image in pixels."""
        return self.source.shape[0]

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
            array=load_image_safe(self.source),
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

    def data(self, extension: str | Extension) -> Data:
        """Encode the image in the format `extension` names.

        Args:
            extension: The format to encode in, `.jpeg` or `.png`.

        Returns:
            The bytes of the file itself when it already carries that
            extension, which skips a decode and a re-encode; otherwise the
            encoded image.

        Raises:
            NotImplementedError: If `extension` names another format.

        """
        extension = _normalize_extension(extension)
        source_path = Path(self.source)
        if source_path.name.lower().endswith(extension.value):
            return Data(source_path.read_bytes())
        return super().data(extension)


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

    def _fetch_bytes(self) -> BinaryIO:
        """Open the URI and give the response as a stream.

        Returns:
            The open response, for the caller to close.

        Raises:
            ValueError: If the URI uses a scheme other than http, https or data.

        """
        # guard against non-web schemes (file:, ftp:, ...) before opening (S310).
        if not self.source.startswith(("http://", "https://", "data:")):
            msg = "WebImage uri must use the http, https or data scheme"
            raise ValueError(msg)
        request = urllib.request.Request(  # noqa: S310
            self.source,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        return urllib.request.urlopen(request)  # noqa: S310

    def _fetch(self) -> RGBArray:
        """Fetch and decode the image, keeping its size for the size properties.

        Returns:
            The pixels as `uint8` of shape `(height, width, 3)`.

        """
        with self._fetch_bytes() as buffer:
            array = load_image_safe(buffer)
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

    def data(self, extension: str | Extension) -> Data:
        """Encode the image in the format `extension` names.

        Args:
            extension: The format to encode in, `.jpeg` or `.png`.

        Returns:
            The bytes as served when the URI already ends in that extension,
            which skips a decode and a re-encode; otherwise the encoded image.

        Raises:
            NotImplementedError: If `extension` names another format.

        """
        extension = _normalize_extension(extension)
        if self.source.lower().endswith(extension.value):
            return Data(self._fetch_bytes().read())
        return super().data(extension)


def _patch_shape(patch: Patch2D, width: int | None, height: int | None) -> tuple[int, int]:
    """Pick the size of a cut, filling in whichever side the caller left open.

    Args:
        patch: The patch to cut, in pixels.
        width: Requested width, or `None` to take it from the patch.
        height: Requested height, or `None` to take it from the patch.

    Returns:
        The height and width of the result, never smaller than one pixel.

    """
    across, down = patch.width_max, patch.height_max
    if width is None and height is not None:
        width = round(height * across / down) if down else 1
    if height is None and width is not None:
        height = round(width * down / across) if across else 1
    if width is None or height is None:
        width, height = round(across), round(down)
    return max(1, height), max(1, width)


class PatchRGBImage(RGBImage):
    """A region of another image, warped back to an upright rectangle.

    Nothing is cut until `array` is called, so a patch of a patch costs only
    the bookkeeping until one of them is realized.
    """

    __slots__ = ("fill", "patch", "source")

    def __init__(self, image: RGBImage, patch: Patch2D, fill: FillValue = WHITE) -> None:
        """Name a region of `image` without cutting it.

        Args:
            image: The image to cut from.
            patch: Region to cut, in coordinates relative to `image`.
            fill: What to use where the patch reaches past the image: one level
                for all three channels, or an `(r, g, b)` color.

        """
        self.source = image
        self.patch = patch
        self.fill = fill

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
        patch = self.patch.to_pixels(width=source.shape[1], height=source.shape[0])
        return sample_quad_uint8(
            source,
            patch.points,
            _patch_shape(patch, width=width, height=height),
            fill=self.fill,
        )

    @property
    def width(self) -> int:
        """Width of the cut in pixels, taken from the longer horizontal edge."""
        patch = self.patch.to_pixels(width=self.source.width, height=self.source.height)
        return _patch_shape(patch, width=None, height=None)[1]

    @property
    def height(self) -> int:
        """Height of the cut in pixels, taken from the longer vertical edge."""
        patch = self.patch.to_pixels(width=self.source.width, height=self.source.height)
        return _patch_shape(patch, width=None, height=None)[0]

    def rot90(self, k: int) -> "PatchRGBImage":
        """Rotate the cut counter-clockwise by `k` quarter turns.

        Args:
            k: Number of quarter turns; taken modulo 4.

        Returns:
            The same region of the same source, with its corner order shifted
            by `k`, rather than a patch stacked on top of this one.

        """
        return type(self)(image=self.source, patch=self.patch.shift(k), fill=self.fill)
