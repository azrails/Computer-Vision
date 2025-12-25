from typing import Any, Optional, Callable
import torch.nn as nn
import multiprocessing as mp
from os import PathLike
from .read_clip import read_clip, read_clip_decord, read_stream
import torch
import time
import argparse
from .utils import frame_filter, build_index
from pathlib import Path
import torchvision.transforms.v2 as transforms
import torchvision
import numpy as np
import pandas as pd
import math
import cv2
from collections import deque
from .workers import _decoder_worker_function, _stream_decoder_worker_function
from .visualizer import LiveVisualizer


def standart_transform_gen():
    """
    Стандартное преобразование для данных, в виде обертки что бы создавать в дочерних процессах.
    """
    return transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            torchvision.ops.Permute((0, 1, 4, 2, 3)),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ParallelPipeline:
    def __init__(
        self,
        index: list[tuple[PathLike, dict[str, Any]]] | PathLike,
        model: nn.Module,
        clip_len: int = 60,
        stride: int = 2,
        queue_size: int = 10,
        device="cuda",
        transform_gen=None,
        cuda_transform=True,
        decoder: Optional[Callable] = read_clip,
        batch_size: int = 1,
        is_stream: bool = False,
        stream_max_frames: int | None = None,
        visualize: bool = False,
        visualize_scale: float = 0.5,
        visualize_every_n: int = 1,
    ):
        self.device = device
        self.batch_size = batch_size
        self.cuda_transform = cuda_transform
        self.model = model
        self.index = index
        self.clip_len = clip_len
        self.stride = stride
        self.offset = (clip_len - 1) * stride + 1

        self.transfer_stream = None
        self.model_stream = None
        if decoder is None:
            raise ValueError("decoder function must be provided")
        self.decoder = decoder
        self.manager = mp.Manager()
        self.decode_queue = self.manager.Queue(maxsize=queue_size)
        self.stop_event = self.manager.Event()
        self.decode_process = None
        self.transform_gen = (
            transform_gen if transform_gen is not None else standart_transform_gen
        )

        self.stream_read_proc = None
        self.frame_queue = self.manager.Queue(maxsize=queue_size) if is_stream else None
        self.is_stream = is_stream
        self.stream_max_frames = stream_max_frames
        self.visualize = visualize
        self.visualize_scale = visualize_scale
        self.visualize_every_n = visualize_every_n
        self.visualizer = None
        if self.visualize:
            self.visualizer = LiveVisualizer(
                window_name="Live RTSP Stream", scale=visualize_scale
            )

    def _prepare_frame_for_visualization(self, clip: torch.Tensor) -> np.ndarray:
        if clip.ndim == 5:
            if clip.shape[2] in [1, 3, 4]:
                frame = clip[0, 0].cpu().numpy()
                frame = np.transpose(frame, (1, 2, 0))
            else:
                frame = clip[0, 0].cpu().numpy()
        elif clip.ndim == 4:
            if clip.shape[1] in [1, 3, 4]:
                frame = clip[0].cpu().numpy()
                frame = np.transpose(frame, (1, 2, 0))
            else:
                frame = clip[0].cpu().numpy()
        else:
            raise ValueError(f"Unexpected clip shape: {clip.shape}")

        if frame.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = frame * std + mean

        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def start_decoder(self):
        """
        Запуск дочерних процессов
        """
        ctx = mp.get_context("spawn")
        if self.is_stream:
            self.stream_read_proc = ctx.Process(
                target=read_stream,
                args=(
                    self.index,
                    self.frame_queue,
                    self.stop_event,
                    self.stride,
                    self.stream_max_frames,
                ),
                daemon=False,
            )
            self.decode_process = ctx.Process(
                target=_stream_decoder_worker_function,
                args=(
                    self.frame_queue,
                    self.decode_queue,
                    self.stop_event,
                    self.clip_len,
                    self.transform_gen
                    if not self.cuda_transform or self.device != "cuda"
                    else None,
                ),
                daemon=False,
            )
            self.stream_read_proc.start()
        else:
            self.decode_process = ctx.Process(
                target=_decoder_worker_function,
                args=(
                    self.index,
                    self.decode_queue,
                    self.stop_event,
                    self.clip_len,
                    self.stride,
                    self.offset,
                    self.decoder,
                    self.transform_gen
                    if not self.cuda_transform or self.device != "cuda"
                    else None,
                    self.batch_size,
                ),
                daemon=False,
            )
        self.decode_process.start()

    @torch.inference_mode()
    def run(self):
        """
        Основной цикл в главном процессе в котором происходит inference модели
        """
        latencies = []
        self.start_decoder()
        worker_transform = None
        if self.device == "cuda":
            assert torch.cuda.is_available(), "CUDA not available"
            self.model = self.model.to(self.device).eval()
            self.transfer_stream = torch.cuda.Stream()
            self.model_stream = torch.cuda.Stream()

            if self.cuda_transform:
                # это для трансформаций на cuda, поскольку контекст cuda можно инициализировать лишь в главном процессе
                worker_transform = self.transform_gen().to("cuda")

        t_total_start = time.monotonic()
        print("Processing batch...")
        clips_processed = 0
        clip = None
        while True:
            if not self.decode_process.is_alive() and self.decode_queue.empty():
                print("Decoder process died unexpectedly")
                break
            try:
                item = self.decode_queue.get()
            except Exception as e:
                if not self.decode_process.is_alive():
                    break
                continue

            if item is None:
                if clip is not None:
                    if self.device == "cuda":
                        with torch.cuda.stream(self.model_stream):
                            self.model_stream.wait_stream(self.transfer_stream)
                            _ = self.model(clip)
                    else:
                        _ = self.model(clip)

                    latency = time.monotonic() - ts
                    latencies.append(latency)

                break

            if clip is not None:
                if self.device == "cuda":
                    with torch.cuda.stream(self.model_stream):
                        self.model_stream.wait_stream(self.transfer_stream)
                        _ = self.model(clip)

                elif self.device != "cuda":
                    _ = self.model(clip)
                latency = time.monotonic() - ts
                latencies.append(latency)

            next_clip, meta, ts = item

            if self.visualize and clips_processed % self.visualize_every_n == 0:
                frame_to_show = self._prepare_frame_for_visualization(next_clip)
                latency = None

                # Показываем кадр
                should_continue = self.visualizer.show(frame_to_show, latency)
                if not should_continue:
                    print("User requested stop")
                    self.stop()
                    break

            if self.device == "cuda":
                with torch.cuda.stream(self.transfer_stream):
                    if not next_clip.is_pinned():
                        next_clip = next_clip.pin_memory()
                    next_clip = next_clip.to(device="cuda", non_blocking=True)
                    if worker_transform is not None:
                        next_clip = worker_transform(next_clip)
                    next_clip.record_stream(self.transfer_stream)
            clip = next_clip
            clips_processed += 1

        total_time = time.monotonic() - t_total_start
        latencies = latencies

        if total_time > 0:
            throughput = (len(latencies) * self.batch_size) / total_time
        else:
            throughput = 0

        if self.visualizer:
            self.visualizer.close()

        latencies = np.array(latencies)
        fps = self.clip_len * throughput
        jitter = np.std(np.diff(latencies))
        return latencies, throughput, fps, jitter

    def stop(self):
        self.stop_event.set()

        try:
            while not self.decode_queue.empty():
                try:
                    self.decode_queue.get_nowait()
                except:
                    break
        except:
            pass

        if self.decode_process is not None:
            self.decode_process.join(timeout=3.0)
            if self.decode_process.is_alive():
                self.decode_process.terminate()
                self.decode_process.join(timeout=1.0)
                if self.decode_process.is_alive():
                    self.decode_process.kill()

        if self.stream_read_proc is not None:
            self.stream_read_proc.join(timeout=3.0)
            if self.stream_read_proc.is_alive():
                self.stream_read_proc.terminate()
                self.stream_read_proc.join(timeout=1.0)
                if self.stream_read_proc.is_alive():
                    self.stream_read_proc.kill()

        try:
            self.manager.shutdown()
        except:
            pass


def measure_fps_pipeline(
    pipe: ParallelPipeline,
    num_iters: int = 100,
    warmup: int = 10,
):
    """
    Функция для обрезания фпс от разогрева и более чистого подсчета статистик
    """
    latencies, throughput, _, _ = pipe.run()
    # Пересчитыванием для честной оценки на числе батчей
    fps = (pipe.batch_size * pipe.clip_len) / latencies
    fps = fps[warmup : warmup + num_iters]

    cv = fps.std() / fps.mean()
    return fps, cv


def main():
    parser = argparse.ArgumentParser(description="Run ParallelPipeline on video clips")

    parser.add_argument(
        "--folder", type=str, required=True, help="Path to the folder with video files"
    )
    parser.add_argument(
        "--clip-len", type=int, default=30, help="Number of frames per clip"
    )
    parser.add_argument("--stride", type=int, default=2, help="Stride between frames")
    parser.add_argument(
        "--queue_size", type=int, default=10, help="Max size of decoding queue"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for model inference"
    )
    parser.add_argument("--decoder", type=str, default="av", choices=["av", "decord"], help="Which decoder use for decode video")
    parser.add_argument("--cuda-transform", type=bool, default=True, help="Make tensor transofrms on cpu or gpu")
    parser.add_argument(
        "--is-stream",
        action="store_true",
    )
    parser.add_argument(
        "--stream-max-frames", type=int, default=100, help="Max input processing frames for rt-like streams"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run FPS stability benchmark"
    )

    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=100,
        help="Number of iterations for FPS benchmark measurement",
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of iterations for FPS benchmark measurement",
    )

    parser.add_argument("--queue-sizes", type=int, nargs="+", default=[8, 16, 32])

    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument(
        "--visualize", action="store_true", help="Show live visualization of the stream"
    )

    parser.add_argument(
        "--visualize-scale",
        type=float,
        default=0.5,
        help="Scale factor for visualization window (default: 0.5)",
    )

    parser.add_argument(
        "--visualize-every-n",
        type=int,
        default=1,
        help="Visualize every N frames (default: 1 = all frames)",
    )

    args = parser.parse_args()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    decoder_fn = read_clip if args.decoder == "av" else read_clip_decord
    if not args.is_stream:
        filters = [frame_filter(args.clip_len, args.stride)]
        index = build_index(Path(args.folder), filters=filters)
        index_str = [(str(path.absolute()), meta) for path, meta in index]
    else:
        index_str = args.folder

    if args.benchmark:
        results = []

        for bs in args.batch_sizes:
            for qs in args.queue_sizes:
                print(f"\nRunning benchmark: batch={bs}, queue={qs}")
                required_clips = (args.benchmark_iters + args.warmup_steps) * bs
                repeat = math.ceil(required_clips / len(index_str))
                index_str = index_str * repeat
                index_str = index_str[:required_clips]
                pipe = ParallelPipeline(
                    index=index_str,
                    model=nn.Identity(),
                    clip_len=args.clip_len,
                    stride=args.stride,
                    queue_size=qs,
                    device=args.device,
                    cuda_transform=args.cuda_transform,
                    decoder=decoder_fn,
                    batch_size=bs,
                )

                fps, cv = measure_fps_pipeline(
                    pipe, num_iters=args.benchmark_iters, warmup=args.warmup_steps
                )

                if fps is None:
                    continue

                results.append(
                    {
                        "batch_size": bs,
                        "queue_size": qs,
                        "mean_fps": fps.mean(),
                        "std_fps": fps.std(),
                        "cv": cv,
                        "stable": cv < 0.05,
                    }
                )

        df = pd.DataFrame(results)
        print("=" * 10 + "FPS Stability Benchmark", 10 * "===")
        print(df.to_string(index=False))

        print("\nStable configurations (CV < 0.05):")
        print(df[df["stable"]].to_string(index=False))
        return

    pipe = ParallelPipeline(
        index=index_str,
        model=nn.Identity(),
        clip_len=args.clip_len,
        stride=args.stride,
        queue_size=args.queue_size,
        device=args.device,
        cuda_transform=args.cuda_transform,
        decoder=decoder_fn,
        batch_size=args.batch_size,
        is_stream=args.is_stream,
        stream_max_frames=args.stream_max_frames,
        visualize=args.visualize,
        visualize_scale=args.visualize_scale,
        visualize_every_n=args.visualize_every_n,
    )

    latencies, th, fps, jitter = pipe.run()

    if latencies is not None:
        print(f"Processed {len(latencies)} clips")
        print(f"Mean latency: {sum(latencies) / len(latencies):.4f}s")
        print(f"Pipeline throughput: {th:.2f} clips/sec")
        print(f"Mean FPS: {np.mean(fps)}")
        print(f"Jitter: {jitter}")
    else:
        print("No clips processed")


if __name__ == "__main__":
    main()
