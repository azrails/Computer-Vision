"""
Набор функций для декодирования видео
"""

from pathlib import Path
from os import PathLike
import numpy.typing as npt
import numpy as np
import av
import multiprocessing as mp
import time

try:
    from decord import VideoReader, gpu, cpu
except ImportError as e:
    print("Warning decord is unvailable")


def read_clip(
    filename: PathLike,
    start: int = 0,
    num_frames: int = 16,
    stride: int = 2,
    additional_info: bool = False,
) -> npt.NDArray:
    """Возвращает кадры из клипа в соответствии с указанными параметрами, использует av для декодирования"""
    filename = Path(filename).resolve()
    frames = []
    indices = []
    frame_times = []
    with av.open(filename) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= start and (i - start) % stride == 0:
                frames.append(frame.to_rgb().to_ndarray())
                indices.append(i)
                frame_times.append(frame.pts)
            if len(frames) >= num_frames:
                break
    clip = np.stack(frames, axis=0)
    if additional_info:
        return clip, indices, frame_times
    return clip


def read_clip_decord(
    filename: PathLike, start: int = 0, num_frames: int = 16, stride: int = 2
) -> npt.NDArray:
    """Возвращает кадры из клипа в соответствии с указанными параметрами, использует decord для декодирования"""
    filename = Path(filename).resolve()
    vr = VideoReader(str(filename), ctx=cpu(0))

    max_valid_start = len(vr) - (num_frames - 1) * stride
    if start >= max_valid_start:
        raise ValueError(f"Invalid start {start}, video too short ({len(vr)} frames)")

    indices = [start + i * stride for i in range(num_frames)]
    frames = vr.get_batch(indices).asnumpy()
    return frames


def read_stream(
    filename: PathLike,
    frame_queue: mp.Queue,
    stop_event,
    stride: int = 2,
    max_frames: int | None = None,
) -> npt.NDArray:
    """Последовательно читает кадры либо из rtsp либо с диска со страйдом и до определенной длинны либо до конца"""
    recived = 0
    with av.open(filename) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if stop_event.is_set():
                break
            if i % stride != 0:
                continue
            recived += 1
            img = frame.to_rgb().to_ndarray()
            frame_queue.put((img, time.monotonic()))
            if max_frames is not None and recived >= max_frames:
                break
    frame_queue.put(None)
