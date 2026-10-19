"""Stable camera identity on Linux.

cv2 opens cameras by ``/dev/videoN`` index, and the kernel hands out N in
enumeration order — replugging a camera (or booting with a different hub
order) silently swaps indices, so a saved ``camera_index`` can open the wrong
device and record e.g. the wrist view under ``top_cam``. udev's
``/dev/v4l/by-id`` symlinks are named after each camera's USB serial, so we
store that path as ``stable_id`` and re-resolve it to the current index right
before opening the device.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

DEV_DIR = "/dev"
# by-id first (follows the camera across ports); by-path as a fallback for
# cameras without a USB serial, which udev can't list under by-id.
_V4L_LINK_DIRS = ("v4l/by-id", "v4l/by-path")


def stable_id_for_index(index: int) -> str | None:
    """Return the udev symlink pointing at /dev/video{index}, or None."""
    target = os.path.join(DEV_DIR, f"video{index}")
    for sub in _V4L_LINK_DIRS:
        link_dir = os.path.join(DEV_DIR, sub)
        try:
            names = sorted(os.listdir(link_dir))
        except OSError:
            continue
        for name in names:
            path = os.path.join(link_dir, name)
            if os.path.realpath(path) == os.path.realpath(target):
                return path
    return None


def index_for_stable_id(stable_id: str) -> int | None:
    """Current /dev/videoN index behind a stable_id, or None if it's absent."""
    if not stable_id or not os.path.exists(stable_id):
        return None
    real = os.path.realpath(stable_id)
    if os.path.dirname(real) != os.path.realpath(DEV_DIR):
        return None
    match = re.fullmatch(r"video(\d+)", os.path.basename(real))
    return int(match.group(1)) if match else None


def resolve_camera_index(camera_data: dict[str, Any]) -> Any:
    """Index to open for a camera config dict.

    Without a ``stable_id`` (macOS/Windows, or configs saved before this
    existed) the saved ``camera_index`` is used as-is. With one, the saved
    index is never trusted: if the camera is unplugged we raise rather than
    fall back, since the old index may now belong to another camera.
    """
    saved = camera_data.get("camera_index", 0)
    stable_id = camera_data.get("stable_id")
    if not stable_id:
        return saved
    index = index_for_stable_id(stable_id)
    if index is None:
        raise RuntimeError(f"Camera not connected: {stable_id}")
    if index != saved:
        logger.info("Camera %s moved from index %s to %s", stable_id, saved, index)
    return index
