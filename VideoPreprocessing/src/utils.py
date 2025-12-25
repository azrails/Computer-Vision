"""
Утилиты для фильтрации кадров и построения списка видео из вложенных директорий
"""

from typing import Any, Callable
from os import PathLike
from pathlib import Path
import av


EXTENSIONS = {".mp4", ".mov", ".mkv"}


def frame_filter(clip_len: int, stride: int) -> Callable[[dict[str, Any]], bool]:
    """Check that num frames more than clip len offset"""

    def _filter(meta: dict[str, Any]) -> bool:
        return meta["frames"] >= ((clip_len - 1) * stride + 1)

    return _filter


def count_frames(filename: PathLike) -> int:
    """Calculate clip length"""
    with av.open(filename) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return stream.frames
        return int(
            container.duration * container.time_base * float(stream.average_rate)
        )


def build_index(
    dir_path: PathLike,
    filters: list[Callable] | None = None,
    extensions: set[str] = EXTENSIONS,
) -> list[tuple[PathLike, dict[str, Any]]]:
    """Recursively check directories and save paths to videofiles"""
    dir_path = Path(dir_path).resolve()
    filters = filters or []
    video_index = []
    for fo in dir_path.iterdir():
        if fo.is_dir():
            video_index.extend(build_index(fo.as_posix(), filters, extensions))
        else:
            if fo.suffix.lower() in extensions:
                meta = {"frames": count_frames(fo), "label": fo.parent.name}
                if all([filter(meta) for filter in filters]):
                    video_index.append((fo, meta))
    return video_index
