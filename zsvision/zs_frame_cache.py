"""A contiguous frame cache data structure, designed to support efficient data loading
from video for the setting in which:
    * frame accesses are sequential (i.e. not random access)
    * the same frames can be read multiple times

Motivation: Suppose the frames of a video are labelled
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
The frame cache aims to store frames in memory so that it is fast to produce batches that
repeat frames e.g.
    [0, 1, 2, 3, 4],
       [1, 2, 3, 4, 5],
          [2, 3, 4, 5, 6],

It does this by allocating a single cache of shared storage, and loading frames into it.
Repeated local frame access are then fast, provided they fall within the cache. It is also
possible to access strided batches e.g:
    [0, 1, 2, 3, 4],
       [2, 3, 4, 5, 6],
          [4, 5, 6, 7, 8],
However, note that the underlying frame remains contiguous (i.e. in this case, it will
still store [0, 1, 2, 3, 4, 5, 6,.....])
"""
import argparse
from typing import Tuple
from pathlib import Path

import numpy as np
from beartype import beartype
from beartype.cave import NoneTypeOr


class DummyVideoReader:
    """A dummy video reader that mimics the OpenCV VideoCapture interface, but yields
    integers rather than frames. Useful for sanity checking frame indices.
    """

    def __init__(self, total_video_frames: int):
        """Create an instance of DummyVideoReader

        Args:
            total_video_frames: the total number of frames to be used in the dummy video.
        """
        self.ptr = 0
        self.frames = list(range(total_video_frames))

    def read(self) -> Tuple:
        """Mimic the OpenCV VideoCapture read() function.

        Returns:
            A tuple comprising a boolean indicating whether the dummy frame read was
            successful, and integer corresponding to the total number of times the read()
            function has been called. If read() is called more times than the total number
            of dummy video frames, the return code is False and the return value is None
        """
        if self.ptr < len(self.frames):
            val = self.frames[self.ptr]
            self.ptr += 1
        else:
            val = None
        return_code = val is not None
        return return_code, val


class FrameOutOfBoundsError(ValueError):
    """A cache exception that will be raised whenever the user requests frame sequences
    that are not legal for the underlying video."""


class CacheCapacityError(ValueError):
    """A cache exception that will be raised whenever the user requests frame sequences
    that cannot be retrieved due to the limited capacity of the cache."""


class InvalidFrameSequence(ValueError):
    """A cache exception that will be raised whenever the user requests an invalid
    frame sequence."""


class ContigFrameCache:
    """A cache to support efficient sequential access to video frames.
    """

    @beartype
    def __init__(
            self,
            num_cache_frames: int,
            frame_height: int,
            frame_width: int,
            cache_dtype: type,
            video_path: Path,
            total_video_frames: int,
            frame_channels: int = 3,
            channel_order: NoneTypeOr[str] = "rgb",
            pad_last: str = "no_pad",
            verbose: bool = False,
            backend: str = "cv2",
    ):
        """Create an instance of ContigFrameCache

        Args:
            num_cache_frames: the number of frames to be stored in the cache at a time
            frame_height: the height (in pixels) of frames to be stored in the cache
            frame_width: the width (in pixels) of frames to be stored in the cache
            cache_dtype: the data type to be used by the frame cache
            video_path: the location of the video to be read
            total_video_frames: the length of the video to be read (in frames)
            frame_channels: the number of channels for each frame (e.g. 3 for RGB)
            channel_order: if three channels are requested, store frames in this order. Can
                be one of None, "rgb" or "bgr" (the latter being the opencv default).
            pad_last: how to handle frame requests beyond the total number of frames in the
                video. Options are `no_pad`, in which case an exception will be raised, or
                `copy_last` in which case the final frame of the video will be duplicated to
                fill the requested frame range.
            verbose: whether to print out detailed information during operation.
            backend: the backend to use as a video reader. Can be either `dummy` or `cv2`
        """
        self.storage = np.zeros((num_cache_frames, frame_height, frame_width,
                                frame_channels), dtype=cache_dtype)

        assert video_path.exists(), f"Could not find video at {video_path}"
        if backend == "dummy":
            assert frame_channels == 1, "dummy backend should use a single channel"
            self.video_cap = DummyVideoReader(total_video_frames)
        elif backend == "cv2":
            try:
                import cv2  # pylint: disable = import-outside-toplevel
            except ImportError:
                raise ImportError("To use cv2 backend, opencv must be installed") from None
            self.video_cap = cv2.VideoCapture(str(video_path))
        else:
            raise ValueError(f"Unknown backend: {backend}")
        if pad_last not in {"no_pad", "zero_pad", "copy_last"}:
            raise ValueError(f"Unknown pad_last strategy: {pad_last}")
        if channel_order not in {None, "rgb", "bgr"}:
            raise ValueError(f"Unknown order: {channel_order}")
        elif channel_order in {"rgb", "bgr"} and frame_channels != 3:
            raise ValueError(f"Expected 3 channels {frame_channels} for {channel_order}")

        self.num_cache_frames = num_cache_frames
        self.channel_order = channel_order
        self.frame_height = frame_height
        self.total_video_frames = total_video_frames
        self.frame_width = frame_width
        self.frame_channels = frame_channels
        self.pad_last = pad_last
        self.verbose = verbose
        self.head_ptr = None

        # We keep track of the previous frame loaded from the video to ensure that we can
        # implement 'copy_last' frame padding if requested
        self.last_seen_frame = None

    @beartype
    def query(self, start_frame: int, end_frame: int) -> bool:
        """Determine whether the sequence of frames [start_frame, end_frame) is contained
        within the frame cache.

        Args:
            start_frame: starting frame in the sequence
            end_frame: end frame in the sequence

        Returns:
            True if the sequence falls within the cache, False otherwise.
        """
        if end_frame - start_frame > self.num_cache_frames:
            raise CacheCapacityError(f"Requested a sequence of {end_frame - start_frame} "
                                     "frames (larger than cache size of "
                                     f"{self.num_cache_frames} frames)")

        if end_frame - start_frame <= 0:
            raise InvalidFrameSequence("Number of frames must be non-negative, "
                                       f"requested {end_frame - start_frame}")

        if start_frame < 0:
            raise InvalidFrameSequence("Starting frame of segment must be non-negative, "
                                       f"requested {start_frame}")

        if self.pad_last == "no_pad" and end_frame > self.total_video_frames:
            raise FrameOutOfBoundsError("When no_pad strategy is used cannot request end "
                                        f"frame {end_frame} beyond the total video length "
                                        f" ({self.total_video_frames} frames)")
        if (
            self.pad_last in {"zero_pad", "copy_last"}
                and start_frame >= self.total_video_frames
        ):
            raise FrameOutOfBoundsError(f"When {self.pad_last} strategy is used cannot "
                                        f"request start frame {start_frame} beyond the "
                                        f" total video length ({self.total_video_frames} "
                                        "frames)")

        if self.head_ptr is None:
            return False

        last_cache_frame = self.head_ptr + self.num_cache_frames
        hit = start_frame >= self.head_ptr and end_frame <= last_cache_frame
        return hit

    @beartype
    def fetch(self, start_frame: int, end_frame: int, copy: bool = True) -> np.ndarray:
        """Fetch a sequence of frames, updating the cache if necessary.

        Args:
            start_frame: starting frame in the requested sequence
            end_frame: end frame in the requested sequence
            copy: whether to return a copy of cached frames, or a direct reference to the
                underlying data.

        Returns:
            The corresponding frame sequence.
        """
        while not self.query(start_frame, end_frame):
            self._update(start_frame)

        start_idx = start_frame - self.head_ptr
        end_idx = end_frame - self.head_ptr
        frames = self.storage[start_idx:end_idx]
        if copy:
            frames = frames.copy()
        return frames

    @beartype
    def _update(self, start_frame: int):
        """Update the frame storage cache so that it's contents begin at the given
        start_frame.

        Args:
            start_frame: the frame at which the updated cache will begin
        """
        if self.verbose:
            print(f"Re-aligning cache to begin at frame {start_frame}")
        if self.head_ptr is None:
            # fill the empty cache
            start_idx = 0
            num_fill_frames = self.num_cache_frames
            remaining_frames_in_video = self.total_video_frames

        elif start_frame <= self.head_ptr + self.num_cache_frames:
            # shift cache frames so that storage begins with the requested frame
            offset = start_frame - self.head_ptr
            keep = self.head_ptr + self.num_cache_frames - start_frame
            self.storage[0:keep] = self.storage[offset:]
            start_idx = keep
            num_fill_frames = self.num_cache_frames - keep
            remaining_frames_in_video = self.total_video_frames - (start_frame + keep)

        else:
            # we need to skip frames in order to refill the cache
            num_skip = start_frame - (self.head_ptr + self.num_cache_frames)
            self.skip_frames(num_skip)

            # refill the cache
            start_idx = 0
            num_fill_frames = self.num_cache_frames
            remaining_frames_in_video = self.total_video_frames - start_frame

        if self.pad_last == "no_pad":
            num_fill_frames = min(num_fill_frames, remaining_frames_in_video)

        self._fill_cache(start_idx=start_idx, num_frames=num_fill_frames)
        self.head_ptr = start_frame

    @beartype
    def skip_frames(self, num_frames: int):
        """Skip forwards by the given number of frames in the video.

        Args:
            The number of frames to skip
        """
        if self.verbose:
            print(f"Skipping {num_frames} frames")
        for _ in range(num_frames):
            self.video_cap.read()

    @beartype
    def compute_num_sequences(
            self,
            sequence_length: int,
            sequence_stride: int,
    ) -> int:
        """A utility function to compute the total number of sequences that will be produced
        by slicing the frames of the video associated with the current cache into sequences.

        Args:
            sequence_length: the length of each sequence in frames.
            sequence_stride: the stride (number of frames between the start of each
                sequence).

        Returns:
            the total number of sequences that will be produced by the video.
        """
        if sequence_length <= 0:
            raise InvalidFrameSequence("Cannot compute number of sequences for non-positive"
                                       " sequence length")
        if sequence_stride <= 0:
            raise ValueError("Cannot compute number of sequences for non-positive"
                             " sequence stride")

        # handle the edge case in which the video is empty
        if self.total_video_frames == 0:
            return 0

        if self.pad_last == "no_pad":
            # handle the edge case in which the requested sequence is longer than the video
            # itself
            if sequence_length > self.total_video_frames:
                num_seqs = 0
            else:
                # handle more typical cases
                num_seqs = 1 + (self.total_video_frames - sequence_length) / sequence_stride
                num_seqs = int(np.floor(num_seqs))

        elif self.pad_last in {"zero_pad", "copy_last"}:
            num_seqs = int(np.ceil(self.total_video_frames / sequence_stride))
        return num_seqs

    @beartype
    def _fill_cache(self, start_idx: int, num_frames: int):
        """Fill the contents of the cache storage with the next sequence of frames from
        the video.

        Args:
            start_idx: the position in cache storage at which to begin writing new frames.
            num_frames: the number of frames to write into storage.
        """
        for frame_idx in range(num_frames):

            _, im = self.video_cap.read()

            if im is None:
                if self.pad_last == "no_pad":
                    raise FrameOutOfBoundsError(
                        f"Attempted to read frame {frame_idx + start_idx + self.head_ptr}, "
                        " beyond the end of the video"
                    )
                elif self.pad_last == "zero_pad":
                    im = self.last_seen_frame * 0
                elif self.pad_last == "copy_last":
                    im = self.last_seen_frame

            if self.channel_order == "rgb":
                im = im[:, :, ::-1]
            elif self.channel_order == "bgr":
                pass

            self.storage[start_idx + frame_idx] = im

            if im is not None:
                self.last_seen_frame = im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=Path, default="data/dali_mwp/sintel.mp4")
    parser.add_argument("--test_vid_height", type=int, default=256)
    parser.add_argument("--test_vid_width", type=int, default=256)
    parser.add_argument("--num_cache_frames", type=int, default=20)
    parser.add_argument("--total_video_frames", type=int, default=1440)
    parser.add_argument("--pad_last", type=str, default="copy_last")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--backend", type=str, choices=["cv2", "dummy"], default="cv2")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cache = ContigFrameCache(
        num_cache_frames=args.num_cache_frames,
        frame_height=args.test_vid_height,
        frame_width=args.test_vid_width,
        video_path=args.video_path,
        total_video_frames=args.total_video_frames,
        cache_dtype=np.uint8,
        verbose=args.verbose,
        pad_last=args.pad_last,
        backend=args.backend,
    )

    for batch_start in range(0, args.total_video_frames + 1, 1):
        batch_end = batch_start + args.batch_size
        frames_ = cache.fetch(start_frame=batch_start, end_frame=batch_end)
        print(f"Batch offset: {batch_start}, frames: {frames_.shape}")
        cache_content = [x[0][0][0] for x in cache.storage]
        batch_content = [x[0][0][0] for x in frames_]
        print("Batch content: " + ",".join(str(x) for x in batch_content))
        print("Cache content: " + ",".join(str(x) for x in cache_content))
