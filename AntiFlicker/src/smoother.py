import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, median_filter


class MaskSmoother:
    def __init__(self, inp, out):
        self.inp = Path(inp)
        self.out = Path(out)
        self.out.mkdir(exist_ok=True, parents=True)

    def smooth(self, method="gaussian", window=5, sigma=None):
        masks = sorted(self.inp.glob("*.jpg"))
        stack = []

        for m in masks:
            stack.append(cv2.imread(str(m), 0))

        stack = np.array(stack) / 255.0  # [T,H,W]

        if method == "gaussian":
            sigma = sigma or (window / 3)
            sm = gaussian_filter1d(stack, sigma=sigma, axis=0)
        elif method == "median":
            sm = median_filter(stack, size=(window, 1, 1))
        elif method == "mean":
            kernel = np.ones(window) / window
            sm = np.zeros_like(stack)
            for i in range(stack.shape[1]):
                for j in range(stack.shape[2]):
                    sm[:, i, j] = np.convolve(stack[:, i, j], kernel, mode="same")
        elif method == "prob":
            sm = np.zeros_like(stack)
            for i in range(len(stack)):
                start = max(0, i - window // 2)
                end = min(len(stack), i + window // 2 + 1)
                sm[i] = np.mean(stack[start:end], axis=0)
        else:
            raise ValueError("Unknown method")

        for i, m in enumerate(sm):
            out = (m > 0.5).astype(np.uint8) * 255
            cv2.imwrite(str(self.out / f"{i:05d}.jpg"), out)
