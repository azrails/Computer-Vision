"""
Процессы воркеры для пайлпайна с перекрытием декодирования/инференса под батчи и стриминговый сетап
"""

import numpy as np
import torch
import time
import random


def _decoder_worker_function(
    index,
    decode_queue,
    stop_event,
    clip_len,
    stride,
    offset,
    decoder_func,
    transform_gen,
    batch_size,
):
    worker_transform = None
    if transform_gen is not None:
        worker_transform = transform_gen()

    chunk = []
    meta_chunk = []
    for clip_path, meta in index:
        if stop_event.is_set():
            break

        try:
            if meta.get("frames", 0) < offset:
                print(f"Skipping {clip_path}: not enough frames")
                continue

            start_frame = random.randint(0, meta["frames"] - offset)
            clip = decoder_func(str(clip_path), start_frame, clip_len, stride)
            clip = torch.from_numpy(clip)

            chunk.append(clip)
            meta_chunk.append(meta)
            if len(chunk) == batch_size:
                chunk = torch.stack(chunk)
                # Если cpu трансформации делаем здесь что бы не занимать время главного процесса
                # но если трансформации на gpu то делаем в главном процессе
                # из-за того что cuda контекст можно инициализировать только в главном процесе
                if worker_transform is not None:
                    chunk = worker_transform(chunk)
                decode_queue.put((chunk, meta_chunk, time.monotonic()))
                meta_chunk = []
                chunk = []

        except Exception as e:
            print(f"Error processing {clip_path}: {e}")
            import traceback

            traceback.print_exc()
            continue
    if chunk:
        chunk = torch.stack(chunk)
        if worker_transform is not None:
            chunk = worker_transform(chunk)
        decode_queue.put((chunk, meta_chunk, time.monotonic()))
    decode_queue.put(None)


def _stream_decoder_worker_function(
    frame_queue, decode_queue, stop_event, clip_len, transform_gen
):
    worker_transform = None
    if transform_gen is not None:
        worker_transform = transform_gen()
    chunk = []
    ts_chunk = []
    while True:
        if stop_event.is_set():
            break

        item = frame_queue.get()
        if item is None:
            break
        frame, ts = item
        clip = torch.from_numpy(frame)

        chunk.append(clip)
        ts_chunk.append(ts)
        if len(chunk) == clip_len:
            chunk = torch.stack(chunk).unsqueeze(0)
            # Если cpu трансформации делаем здесь что бы не занимать время главного процесса
            # но если трансформации на gpu то делаем в главном процессе
            # из-за того что cuda контекст можно инициализировать только в главном процесе
            if worker_transform is not None:
                chunk = worker_transform(chunk)
            # Поскольку читаем 1 стрим пайплайн работы тот же что и для чтения клипов но уже клеем в батч не клипы а набор кадров
            decode_queue.put((chunk, ts_chunk, time.monotonic()))
            ts_chunk = []
            chunk = []

    if chunk:
        chunk = torch.stack(chunk).unsqueeze(0)
        if worker_transform is not None:
            chunk = worker_transform(chunk)
        decode_queue.put((chunk, ts_chunk, time.monotonic()))
    decode_queue.put(None)
