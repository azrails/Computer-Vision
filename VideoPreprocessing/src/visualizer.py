import time
import numpy as np
import cv2
from collections import deque


class LiveVisualizer:
    """
    Визуализация обработки видео в реальном времени
    """

    def __init__(
        self,
        window_name: str = "Live Stream",
        fps_history_size: int = 30,
        show_fps: bool = True,
        show_latency: bool = True,
        scale: float = 1.0,
    ):
        self.window_name = window_name
        self.show_fps = show_fps
        self.show_latency = show_latency
        self.scale = scale

        # История для расчёта FPS
        self.fps_history = deque(maxlen=fps_history_size)
        self.last_frame_time = time.monotonic()

        # Инициализация окна
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def draw_info(self, frame: np.ndarray, latency: float = None) -> np.ndarray:
        """
        Рисует информацию на кадре
        """
        frame = frame.copy()
        h, w = frame.shape[:2]

        # Расчёт FPS
        current_time = time.monotonic()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        if frame_time > 0:
            instant_fps = 1.0 / frame_time
            self.fps_history.append(instant_fps)

        avg_fps = np.mean(self.fps_history) if self.fps_history else 0

        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        if self.show_fps:
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, y_offset),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
            )
            y_offset += 30

        if self.show_latency and latency is not None:
            lat_text = f"Latency: {latency * 1000:.1f} ms"
            cv2.putText(
                frame,
                lat_text,
                (10, y_offset),
                font,
                font_scale,
                (0, 255, 255),
                thickness,
            )
            y_offset += 30

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, h - 10), font, 0.5, (200, 200, 200), 1)

        return frame

    def show(self, frame: np.ndarray, latency: float = None, wait_key: int = 1):
        """
        Отображает кадр с информацией

        Returns:
            True если нужно продолжать, False если нажата клавиша выхода
        """
        frame_with_info = self.draw_info(frame, latency)

        if self.scale != 1.0:
            h, w = frame_with_info.shape[:2]
            new_h, new_w = int(h * self.scale), int(w * self.scale)
            frame_with_info = cv2.resize(frame_with_info, (new_w, new_h))

        # Отображение
        cv2.imshow(self.window_name, frame_with_info)

        key = cv2.waitKey(wait_key) & 0xFF
        return key not in [ord("q"), 27]  # 27 = ESC

    def close(self):
        cv2.destroyWindow(self.window_name)
