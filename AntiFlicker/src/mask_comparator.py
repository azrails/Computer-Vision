import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


class MaskComparator:
    def __init__(self, raw, smooth):
        self.raw = Path(raw)
        self.smooth = Path(smooth)

    def compare(self):
        raw = sorted(self.raw.glob("*.jpg"))
        sm = sorted(self.smooth.glob("*.jpg"))
        ious = []
        for r, s in zip(raw, sm):
            r = cv2.imread(str(r), 0) > 128
            s = cv2.imread(str(s), 0) > 128
            inter = np.logical_and(r, s).sum()
            union = np.logical_or(r, s).sum()
            ious.append(inter / union if union > 0 else 1)
        return np.array(ious)

    def plot_comparison(self, iou_raw, iou_smooth):
        plt.plot(iou_raw, label="Raw")
        plt.plot(iou_smooth, label="Smoothed")
        plt.legend()
        plt.title("Comparison: Raw vs Smoothed")
        plt.savefig("output/plots/compare.png")
        plt.close()
