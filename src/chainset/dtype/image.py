import math
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from iokit import Data, Jpeg, Png
from iokit.state import Image as ImageFormatState
from numpy.typing import NDArray
from PIL import Image as PILImage

from chainset.dtype.patch import Patch2D
from iokit.dtype.extension import Extension

RGBArray = NDArray[np.uint8]
PathLike = str | Path


# TODO(@rilshok): when touching the arr, calculate h and w and vice versa


def _normalize_extension(extension: str | Extension) -> Extension:
    if isinstance(extension, str):
        extension = f".{extension.removeprefix('.')}"
        extension = Extension(extension)
    if extension not in {Extension.JPEG, Extension.PNG}:
        raise NotImplementedError
    return extension


def _assert_rgb_array(array: RGBArray) -> None:
    if array.ndim != 3:
        msg = "RGB array must have 3 dimensions"
        raise ValueError(msg)
    if array.shape[2] != 3:
        msg = "RGB array must have 3 channels"
        raise ValueError(msg)
    if array.dtype != np.uint8:
        msg = "RGB array dtype must be uint8"
        raise ValueError(msg)


def pil_to_rgb(image: PILImage.Image) -> RGBArray:
    """Convert a PIL image to an RGB array, flattening alpha onto white."""
    if image.mode == "RGB":
        return np.array(image, dtype=np.uint8)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    background = PILImage.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    return np.array(background, dtype=np.uint8)


def load_image_safe(buffer: PathLike | BinaryIO) -> RGBArray:
    """Load image and convert to RGB array."""
    with PILImage.open(buffer) as image:
        return _pil_to_rgb(image)


def _resize_array(array: RGBArray, width: int, height: int) -> RGBArray:
    """Resize RGB array using PIL Lanczos resampling."""
    image = PILImage.fromarray(array)
    image = image.resize((width, height), PILImage.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)


def _maybe_resize_array(
    array: RGBArray,
    width: int | None,
    height: int | None,
) -> RGBArray:
    """Resize array if requested, preserving aspect ratio if only one given."""
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
    @abstractmethod
    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        raise NotImplementedError

    @property
    def width(self) -> int:
        raise NotImplementedError

    @property
    def height(self) -> int:
        raise NotImplementedError

    @property
    def pil(self) -> PILImage.Image:
        return PILImage.fromarray(self.array())

    def to_jpeg(self, stem: str | None = None) -> Jpeg:
        return Jpeg(self.pil, stem=stem)

    def to_png(self, stem: str | None = None) -> Png:
        return Png(self.pil, stem=stem)

    def data(self, extension: str | Extension) -> Data:
        match _normalize_extension(extension):
            case Extension.JPEG | Extension.JPG:
                return self.to_jpeg().data
            case Extension.PNG:
                return self.to_png().data
            case _:
                raise NotImplementedError

    def _repr_html_(self) -> str:
        content = self.data(Extension.JPEG).base64
        return f'<img src="data:image/jpeg;base64,{content}" />'

    @property
    def loaded(self) -> "LoadedRGBImage":
        return LoadedRGBImage(self.array())

    def cut(self, patch: Patch2D) -> "PatchRGBImage":
        return PatchRGBImage(image=self, patch=patch)

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
    def dump(self, data: RGBImage) -> PILImage.Image:
        return data.pil

    def parse(self, data: PILImage.Image) -> RGBImage:
        return LoadedRGBImage(_pil_to_rgb(data))


class RGBImageJpeg(RGBImageState):
    __extension__ = Extension.JPEG


class RGBImagePng(RGBImageState):
    __extension__ = Extension.PNG


class LoadedRGBImage(RGBImage):
    __slots__ = ("source",)

    def __init__(self, array: RGBArray) -> None:
        _assert_rgb_array(array)
        self.source = array

    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        if width is None and height is None:
            return self.source.copy()
        return _maybe_resize_array(
            array=self.source,
            width=width,
            height=height,
        )

    @property
    def width(self) -> int:
        return self.source.shape[1]

    @property
    def height(self) -> int:
        return self.source.shape[0]

    @property
    def loaded(self) -> "LoadedRGBImage":
        return self


class FileRGBImage(RGBImage):
    __slots__ = ("_height", "_width", "source")

    def __init__(self, path: PathLike) -> None:
        self.source = Path(path).as_posix()
        self._width: int | None = None
        self._height: int | None = None

    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        return _maybe_resize_array(
            array=load_image_safe(self.source),
            width=width,
            height=height,
        )

    @property
    def width(self) -> int:
        if self._width is None:
            shape = self.array().shape
            self._height = shape[0]
            self._width = shape[1]
        return self._width

    @property
    def height(self) -> int:
        if self._height is None:
            shape = self.array().shape
            self._height = shape[0]
            self._width = shape[1]
        return self._height

    def data(self, extension: str | Extension) -> Data:
        extension = _normalize_extension(extension)
        source_path = Path(self.source)
        if source_path.name.lower().endswith(extension.value):
            return Data(source_path.read_bytes())
        return super().data(extension)


class WebRGBImage(RGBImage):
    __slots__ = ("_height", "_width", "source")

    def __init__(self, uri: str) -> None:
        if not uri.startswith(("http://", "https://", "data:")):
            msg = "WebImage uri must use the http, https or data scheme"
            raise ValueError(msg)
        self.source = uri
        self._width: int | None = None
        self._height: int | None = None

    def _fetch_bytes(self) -> BinaryIO:
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
        with self._fetch_bytes() as buffer:
            array = load_image_safe(buffer)
        self._height, self._width = array.shape[:2]
        return array

    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        return _maybe_resize_array(array=self._fetch(), width=width, height=height)

    @property
    def width(self) -> int:
        if self._width is None:
            self._fetch()
        if self._width is None:
            msg = "Failed to determine image width after fetching"
            raise SystemError(msg)
        return self._width

    @property
    def height(self) -> int:
        if self._height is None:
            self._fetch()
        if self._height is None:
            msg = "Failed to determine image height after fetching"
            raise SystemError(msg)
        return self._height

    def data(self, extension: str | Extension) -> Data:
        extension = _normalize_extension(extension)
        if self.source.lower().endswith(extension.value):
            return Data(self._fetch_bytes().read())
        return super().data(extension)


class PatchRGBImage(RGBImage):
    __slots__ = ("patch", "source")

    def __init__(self, image: RGBImage, patch: Patch2D) -> None:
        self.source = image
        self.patch = patch

    def array(self, *, width: int | None = None, height: int | None = None) -> RGBArray:
        array = self.source.array()
        h, w = array.shape[:2]

        patch = self.patch.to_pixels(width=w, height=h)

        if width is None and height is None:
            width = round(patch.width_max)
            height = round(patch.height_max)
        elif width is None:
            if height is None:
                msg = "Both width and height are None after initial check"
                raise SystemError(msg)
            width = round(height * patch.width_max / patch.height_max)
        elif height is None:
            height = round(width * patch.height_max / patch.width_max)

        x, y = patch.meshgrid(width=width, height=height)

        box = patch.box
        x0 = max(0, math.floor(box.x1) - 1)
        x1 = min(w, math.ceil(box.x2) + 2)
        y0 = max(0, math.floor(box.y1) - 1)
        y1 = min(h, math.ceil(box.y3) + 2)
        crop = array[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape[:2]

        grid = np.stack(
            (
                2 * (x - x0) / crop_w - 1.0,
                2 * (y - y0) / crop_h - 1.0,
            ),
            axis=-1,
        ).astype(np.float32)

        with torch.inference_mode():
            crop_array = np.ascontiguousarray(crop)
            i = torch.from_numpy(crop_array).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
            g = torch.from_numpy(grid).unsqueeze(0)
            r = torch.nn.functional.grid_sample(
                i,
                grid=g,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

        return r[0].permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)

    @property
    def width(self) -> int:
        patch = self.patch.to_pixels(width=self.source.width, height=self.source.height)
        return round(patch.width_max)

    @property
    def height(self) -> int:
        patch = self.patch.to_pixels(width=self.source.width, height=self.source.height)
        return round(patch.height_max)

    def rot90(self, k: int) -> "PatchRGBImage":
        return type(self)(image=self.source, patch=self.patch.shift(k))
