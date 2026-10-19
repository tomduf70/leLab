"""Tests for lelab.utils.cameras — stable_id <-> /dev/videoN resolution,
against a fake /dev tree of udev-style symlinks."""

from __future__ import annotations

import os

import pytest

from lelab.rollout import _format_cameras_arg
from lelab.utils import cameras


@pytest.fixture
def fake_dev(tmp_path, monkeypatch: pytest.MonkeyPatch):
    for n in range(4):
        (tmp_path / f"video{n}").touch()
    by_id = tmp_path / "v4l" / "by-id"
    by_id.mkdir(parents=True)
    os.symlink("../../video0", by_id / "usb-Innomaker-video-index0")
    os.symlink("../../video1", by_id / "usb-Innomaker-video-index1")
    os.symlink("../../video2", by_id / "usb-USB_Camera-video-index0")
    monkeypatch.setattr(cameras, "DEV_DIR", str(tmp_path))
    return tmp_path


def test_stable_id_for_index(fake_dev):
    assert cameras.stable_id_for_index(2) == str(fake_dev / "v4l/by-id/usb-USB_Camera-video-index0")
    assert cameras.stable_id_for_index(3) is None


def test_index_for_stable_id(fake_dev):
    assert cameras.index_for_stable_id(str(fake_dev / "v4l/by-id/usb-Innomaker-video-index0")) == 0
    assert cameras.index_for_stable_id(str(fake_dev / "v4l/by-id/missing")) is None
    assert cameras.index_for_stable_id("") is None


def test_resolve_follows_camera_after_replug(fake_dev):
    link = fake_dev / "v4l/by-id/usb-USB_Camera-video-index0"
    link.unlink()
    os.symlink("../../video0", link)
    assert cameras.resolve_camera_index({"camera_index": 2, "stable_id": str(link)}) == 0


def test_resolve_without_stable_id_keeps_saved_index():
    assert cameras.resolve_camera_index({"camera_index": 4}) == 4
    assert cameras.resolve_camera_index({"camera_index": 4, "stable_id": ""}) == 4


def test_resolve_unplugged_camera_raises(fake_dev):
    with pytest.raises(RuntimeError, match="not connected"):
        cameras.resolve_camera_index({"camera_index": 2, "stable_id": str(fake_dev / "v4l/by-id/gone")})


def test_format_cameras_arg_drops_stable_id(fake_dev):
    link = str(fake_dev / "v4l/by-id/usb-USB_Camera-video-index0")
    arg = _format_cameras_arg({"top": {"type": "opencv", "camera_index": 0, "stable_id": link, "fps": 30}})
    assert arg == "{top: {type: opencv, index_or_path: 2, fps: 30}}"
