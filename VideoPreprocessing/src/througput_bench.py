import time
from torch.utils.data import DataLoader, ConcatDataset


def througput_bench(workers, dataset, batch_size=4, device="cpu", dataset_scale=1):
    """
    Функция бенчмарк для замера кадров/сек в зависимости от числа потоков в лоадере
    """
    througput = []
    processing_times = []
    latencies = []
    for num_workers in workers:
        loader = DataLoader(
            ConcatDataset([dataset] * dataset_scale),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
        )
        total_frames = batch_size * dataset.clip_len * len(loader)
        start = time.monotonic()
        for _ in loader:
            pass
        end = time.monotonic()
        working_time = end - start
        througput.append(total_frames / working_time)
        processing_times.append(working_time)
        latencies.append(working_time / total_frames)
        print(
            15 * "="
            + f"NUM WORKERS: {num_workers}, DATASET SIZE: {len(dataset)}"
            + 15 * "="
        )
        print(f"time in seconds: {processing_times[-1]:6f}")
        print(f"througput: {througput[-1]}")
        print(f"latency: {latencies[-1]}")
    return througput, processing_times, latencies
