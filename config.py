
from __future__ import annotations
from typing import Optional
import argparse
import re
from dataclasses import dataclass, field

@dataclass
class Config:
    device_path: str = "/dev/video1"
    device_index: Optional[int] = None  # derived from device_path if None
    cam_width: int = 1280
    cam_height: int = 720
    cam_fps: int = 120
    output_width: int = 960
    output_height: int = 540
    # Size (in *source* pixels, before resize) of the region we crop out
    # of the full sensor frame. Aspect ratio is forced to match
    # output_width/output_height so the resize never stretches geometry.
    crop_width: int = 280
    landmarker_model_path: str = "face_landmarker.task"
    model_path: Optional[str] = None  # load/save trained gaze model here
    output_dir: str = "."
    min_calibration_frames: int = 20
    mad_outlier_threshold: float = 3.5
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.3  # [FIX 11] was 0.007; see docstring below
    median_window: int = 5


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device-path", default="/dev/video1")
    p.add_argument("--device-index", type=int, default=None)
    p.add_argument("--cam-width", type=int, default=1280)
    p.add_argument("--cam-height", type=int, default=720)
    p.add_argument("--cam-fps", type=int, default=120)
    p.add_argument("--crop-width", type=int, default=280,
                    help="Source-pixel width of the crop region before resize.")
    p.add_argument("--landmarker-model", default="face_landmarker.task")
    p.add_argument("--model-path", default=None,
                    help="Load a previously-trained gaze model from here, "
                         "or save the newly trained one here if it doesn't exist.")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()

    cfg = Config(
        device_path=args.device_path,
        device_index=args.device_index,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        cam_fps=args.cam_fps,
        crop_width=args.crop_width,
        landmarker_model_path=args.landmarker_model,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )
    if cfg.device_index is None:
        # [FIX 17] Derive the index cv2 needs from the /dev/videoN path so
        # the v4l2-ctl calls and cv2.VideoCapture can't silently diverge.
        match = re.search(r"(\d+)$", cfg.device_path)
        cfg.device_index = int(match.group(1)) if match else 0
    return cfg