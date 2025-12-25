from typing import Any, Callable
from os import PathLike
from torch.utils.data import Dataset
import random
import torch
from torch.profiler import record_function
from .read_clip import read_clip, read_clip_decord


class VideoDataset(Dataset):
    def __init__(
        self,
        index: list[(PathLike, dict[str, Any])],
        clip_len: int = 60,
        stride: int = 1,
        transform=None,
        decoder: Callable = "av",
    ):
        super().__init__()
        if decoder == "av":
            self.decoder = read_clip
        else:
            self.decoder = read_clip_decord
        self.index = index
        self.clip_len = clip_len
        self.stride = stride
        self.offset = (clip_len - 1) * stride + 1
        self.transform = transform

    def __getitem__(self, idx):
        clip_path, meta = self.index[idx]
        start_frame = random.randint(0, meta["frames"] - self.offset)
        with record_function("clip_decoding"):
            clip = self.decoder(clip_path, start_frame, self.clip_len, self.stride)
        clip = torch.from_numpy(clip)
        with record_function("preprocessing"):
            if self.transform is not None:
                clip = self.transform(clip)
        return clip, meta

    def __len__(self):
        return len(self.index)
