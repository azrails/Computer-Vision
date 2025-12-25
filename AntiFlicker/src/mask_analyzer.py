import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class MaskAnalyzer:
    def __init__(self, mask_dir, out_dir):
        self.mask_dir = Path(mask_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def iou(self, m1, m2):
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return inter / union if union > 0 else 1

    def analyze_stability(self):
        masks = sorted(self.mask_dir.glob("*.jpg"))
        ious = []
        for i in range(len(masks) - 1):
            m1 = cv2.imread(str(masks[i]), 0) > 128
            m2 = cv2.imread(str(masks[i + 1]), 0) > 128
            ious.append(self.iou(m1, m2))
        self.ious = np.array(ious)
        return self.ious

    def plot_stability(self):
        plt.plot(self.ious)
        plt.title("Temporal IoU Consistency (raw)")
        plt.savefig(self.out_dir / "stability_raw.png")
        plt.close()
