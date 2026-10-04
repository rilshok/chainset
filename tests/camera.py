"""Pinhole camera for tests."""

import math

import numpy as np

Quad = list[tuple[float, float]]


def view_of_rectangle(
    size: tuple[float, float],
    turn: tuple[float, float],
    focal: float,
    frame: tuple[int, int],
) -> Quad:
    """Project a turned rectangle through a pinhole camera centered on `frame`.

    Args:
        size: Width and height of the rectangle.
        turn: Rotation about the x and then the y axis, in radians.
        focal: Focal length, in pixels.
        frame: Width and height of the image, in pixels.

    Returns:
        The corners of the rectangle in the image, in patch order.

    """
    width, height = size
    corners = np.array(
        [
            [-width / 2, -height / 2, 0.0],
            [width / 2, -height / 2, 0.0],
            [width / 2, height / 2, 0.0],
            [-width / 2, height / 2, 0.0],
        ],
    )
    pitch, yaw = turn
    about_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
    )
    about_y = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
    )
    placed = corners @ (about_y @ about_x).T + np.array([0.3, -0.2, 3.0])
    xs = focal * placed[:, 0] / placed[:, 2] + frame[0] / 2
    ys = focal * placed[:, 1] / placed[:, 2] + frame[1] / 2
    return [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
