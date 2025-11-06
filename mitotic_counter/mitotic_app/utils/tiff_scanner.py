# utils/tiff_scanner.py
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

class TIFFScanner:
    def __init__(self, slide_path):
        path = Path(slide_path)
        if path.suffix.lower() not in ['.tif', '.tiff']:
            raise ValueError("Only TIFF formats are supported")

        self.slide = Image.open(slide_path)
        self.dimensions = self.slide.size

    def smooth_scan(self, output_dir, window_size=(256, 256), speed=20):
        print(f"Scanning with window size {window_size}")
        x_steps = np.arange(0, self.dimensions[0] - window_size[0] + 1, speed)
        y_steps = np.arange(0, self.dimensions[1] - window_size[1] + 1, speed * 12.5)

        # Prepare output directories
        video_path = os.path.join(output_dir, "tiff_scan.mp4")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize video writer
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, window_size)

        frame_count = 0
        for step_y, y in enumerate(y_steps):
            for step_x, x in enumerate(x_steps):
                frame_count += 1

                region = self.slide.crop((
                    int(x), int(y),
                    int(x + window_size[0]), int(y + window_size[1])
                ))

                frame = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
                out.write(frame)

        out.release()
        print(f"Saved video to {video_path}")
        return video_path
    
import ffmpeg

def convert_to_mp4(input_path, output_path):
    (
        ffmpeg
        .input(input_path)
        .output(output_path, vcodec='libx264', crf=23, pix_fmt='yuv420p')
        .overwrite_output()
        .run()
    )