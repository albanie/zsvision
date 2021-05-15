"""Test suite for zsvision/zs_frame_cache.py
"""

from pathlib import Path

import numpy as np
import pytest
from zsvision.zs_frame_cache import (
    ContigFrameCache,
    CacheCapacityError,
    InvalidFrameSequence,
    FrameOutOfBoundsError
)


def test_num_sequence_computer():
    tests = (
        ({"frames": 0, "seq_length": 3, "seq_stride": 1, "pad_last": "no_pad"}, 0),
        ({"frames": 2, "seq_length": 3, "seq_stride": 1, "pad_last": "no_pad"}, 0),
        ({"frames": 2, "seq_length": 50, "seq_stride": 1, "pad_last": "no_pad"}, 0),
        ({"frames": 3, "seq_length": 3, "seq_stride": 1, "pad_last": "no_pad"}, 1),
        ({"frames": 4, "seq_length": 3, "seq_stride": 1, "pad_last": "no_pad"}, 2),

        ({"frames": 0, "seq_length": 3, "seq_stride": 1, "pad_last": "zero_pad"}, 0),
        ({"frames": 2, "seq_length": 3, "seq_stride": 1, "pad_last": "zero_pad"}, 2),
        ({"frames": 2, "seq_length": 50, "seq_stride": 1, "pad_last": "zero_pad"}, 2),

        ({"frames": 0, "seq_length": 3, "seq_stride": 1, "pad_last": "copy_last"}, 0),
        ({"frames": 2, "seq_length": 3, "seq_stride": 1, "pad_last": "copy_last"}, 2),
        ({"frames": 2, "seq_length": 50, "seq_stride": 1, "pad_last": "copy_last"}, 2),
    )
    for test_cfg, expected_sequences in tests:
        cache = ContigFrameCache(
            num_cache_frames=100,
            frame_height=1,
            frame_width=1,
            video_path=Path(""),
            total_video_frames=test_cfg["frames"],
            verbose=False,
            pad_last=test_cfg["pad_last"],
            cache_dtype=np.uint8,
            backend="dummy",
        )
        num_sequences = cache.compute_num_sequences(
            sequence_length=test_cfg["seq_length"],
            sequence_stride=test_cfg["seq_stride"],
        )
        assert num_sequences == expected_sequences, (
            f"Mismatch for {test_cfg}: {num_sequences} vs {expected_sequences} (expected)"
        )


def test_fetched_frames():
    tests = (
        ({"frames": 3, "seq_length": 3, "seq_stride": 1, "pad_last": "no_pad"},
            [[0, 1, 2]]),
        ({"frames": 4, "seq_length": 3, "seq_stride": 1, "pad_last": "no_pad"},
            [[0, 1, 2], [1, 2, 3]]),
        ({"frames": 5, "seq_length": 3, "seq_stride": 1, "pad_last": "no_pad"},
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]]),

        ({"frames": 3, "seq_length": 3, "seq_stride": 1, "pad_last": "zero_pad"},
            [[0, 1, 2], [1, 2, 0], [2, 0, 0]]),
        ({"frames": 3, "seq_length": 3, "seq_stride": 2, "pad_last": "zero_pad"},
            [[0, 1, 2], [2, 0, 0]]),

        ({"frames": 3, "seq_length": 3, "seq_stride": 1, "pad_last": "copy_last"},
            [[0, 1, 2], [1, 2, 2], [2, 2, 2]]),
        ({"frames": 3, "seq_length": 3, "seq_stride": 2, "pad_last": "copy_last"},
            [[0, 1, 2], [2, 2, 2]]),
    )

    for test_cfg, expected_frames in tests:
        cache = ContigFrameCache(
            num_cache_frames=100,
            frame_height=1,
            frame_width=1,
            video_path=Path(""),
            total_video_frames=test_cfg["frames"],
            verbose=False,
            pad_last=test_cfg["pad_last"],
            frame_channels=1,
            cache_dtype=np.uint8,
            backend="dummy",
        )
        num_sequences = cache.compute_num_sequences(
            sequence_length=test_cfg["seq_length"],
            sequence_stride=test_cfg["seq_stride"],
        )
        fetched = []
        for seq_idx in range(num_sequences):
            seq_start = seq_idx * test_cfg["seq_stride"]
            seq_end = seq_start + test_cfg["seq_length"]
            frames = cache.fetch(start_frame=seq_start, end_frame=seq_end).tolist()

            # Dummy frames are stored as H x W x C arrays, where H, W and C are each 1, so
            # we flatten the arrays to get a list of frame numbers
            frames = [x[0][0][0] for x in frames]
            fetched.append(frames)

        assert fetched == expected_frames, (
            f"Mismatch for {test_cfg}: {fetched} vs {expected_frames} (expected)"
        )


def test_out_of_bounds_behaviour():
    total_video_frames = 10
    num_cache_frames = 5
    common_kwargs = dict(
        num_cache_frames=num_cache_frames,
        frame_height=1,
        frame_width=1,
        video_path=Path(""),
        total_video_frames=total_video_frames,
        verbose=False,
        frame_channels=1,
        cache_dtype=np.uint8,
        backend="dummy",
    )
    cache = ContigFrameCache(
        pad_last="no_pad",
        **common_kwargs,
    )
    with pytest.raises(CacheCapacityError):
        cache.fetch(start_frame=0, end_frame=num_cache_frames + 1)

    with pytest.raises(InvalidFrameSequence):
        cache.fetch(start_frame=-1, end_frame=0)

    with pytest.raises(InvalidFrameSequence):
        cache.fetch(start_frame=0, end_frame=0)

    with pytest.raises(FrameOutOfBoundsError):
        cache.fetch(start_frame=total_video_frames - 1, end_frame=total_video_frames + 1)

    cache = ContigFrameCache(
        pad_last="copy_last",
        **common_kwargs,
    )
    # should pass without raising an exception
    cache.fetch(start_frame=total_video_frames - 1, end_frame=total_video_frames + 1)

    with pytest.raises(FrameOutOfBoundsError):
        cache.fetch(start_frame=total_video_frames, end_frame=total_video_frames + 1)


if __name__ == "__main__":
    test_num_sequence_computer()
    test_fetched_frames()
    test_out_of_bounds_behaviour()
