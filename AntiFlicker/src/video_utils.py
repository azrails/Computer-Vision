import cv2
from pathlib import Path


class VideoExtractor:
    def __init__(self, output_dir="output/frames"):
        self.output_dir = Path(output_dir)

    def extract(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(self.output_dir / f"{i:05d}.jpg"), frame)
            i += 1
        cap.release()
        return i, fps
