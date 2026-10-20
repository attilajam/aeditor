"""Output size inference, validation, and CLI sizing behavior."""

import pytest

from autoedit.cli import build_parser
from autoedit.config import Config
from autoedit.probe import MediaError
from autoedit.video import source_output_size


def _info(width, height):
    # probe() has already converted encoded dimensions through rotation and SAR.
    return {"display_width": width, "display_height": height}


def test_auto_size_uses_dominant_display_aspect_and_common_native_resolution():
    # A rotated 240x160 source is displayed as portrait. Portrait wins by
    # count, then the most common displayed native size within that aspect wins.
    infos = [_info(160, 240), _info(160, 240), _info(160, 240), _info(240, 160), _info(300, 200)]
    assert source_output_size(infos) == "160x240"
    assert source_output_size([_info(160, 240), _info(240, 160), _info(160, 240)]) == "160x240"


def test_auto_size_ties_prefer_larger_aspect_group_then_larger_common_resolution():
    # Equal aspect counts choose the group containing the larger native frame.
    # Within that group, equally common sizes also choose the larger resolution.
    infos = [_info(240, 160), _info(300, 200), _info(160, 240), _info(160, 240)]
    assert source_output_size(infos) == "300x200"
    infos = [_info(240, 160), _info(300, 200)]
    assert source_output_size(infos) == "300x200"


def test_auto_size_rejects_missing_display_dimensions():
    with pytest.raises(MediaError, match="no valid displayed video dimensions"):
        source_output_size([_info(0, 0)])


def test_config_auto_size_is_valid_and_custom_dimensions_require_even_minimum():
    assert Config().validate().size == "auto"
    assert Config(size="160x240").validate().dimensions == (160, 240)
    assert Config(size="16x16").validate().dimensions == (16, 16)
    for size in ("15x240", "321x241", "320x15", "bad"):
        with pytest.raises(ValueError):
            Config(size=size).validate()


def test_resolution_option_is_exclusive_with_legacy_size():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["create", "--audio", "a.wav", "--clips", "clips", "--output", "out.mp4",
                           "--resolution", "320x180", "--size", "160x240"])
    parsed = parser.parse_args(["create", "--audio", "a.wav", "--clips", "clips", "--output", "out.mp4",
                               "--resolution", "320x180"])
    assert parsed.resolution == "320x180"
    assert parsed.size is None


def test_resolution_requires_explicit_even_dimensions():
    # CLI resolution uses the same Config validation and rejects automatic or
    # odd/undersized dimensions before input analysis begins.
    from autoedit.cli import main

    for resolution in ("auto", "321x241", "15x240"):
        assert main(["create", "--audio", "missing.wav", "--clips", "missing", "--output", "out.mp4",
                     "--resolution", resolution]) == 2
