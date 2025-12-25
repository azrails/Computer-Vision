import torch
import torchvision
import cv2
from pathlib import Path
import numpy as np


class SemanticSegmentation:
    def __init__(self, output_dir="output/masks_raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights="DEFAULT"
        ).eval()

        self.preprocess = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

    def process(self, frame_dir):
        for frame in sorted(Path(frame_dir).glob("*.jpg")):
            img = cv2.imread(str(frame))
            t = self.preprocess(img).unsqueeze(0)
            with torch.no_grad():
                pred = self.model(t)["out"].argmax(1)[0].byte().cpu().numpy()
            mask = (pred > 0).astype(np.uint8) * 255
            cv2.imwrite(str(self.output_dir / frame.name), mask)
