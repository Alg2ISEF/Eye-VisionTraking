
import numpy as np
from pathlib import Path
import cv2
from dataclasses import dataclass , field
import time
from camera import HoldToRecord
from typing import Optional

@dataclass
class Paths:
    output_dir: Path
    validation_output_dir: Path
    jitter_output_dir: Path
    run_stamp: str




def reject_outliers(samples: np.ndarray, threshold: float) -> np.ndarray:
    """Drop rows whose per-dimension deviation from the median exceeds
    `threshold` MADs in any dimension.

    [FIX 9] Applied to every calibration burst before it's accepted, so
    a blink or a stray saccade in the middle of a burst can't poison an
    entire calibration point the way it could when the old code only
    required >= 2 raw frames with no filtering at all.
    """
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median), axis=0)
    mad = np.where(mad < 1e-6, 1e-6, mad)
    deviation = np.abs(samples - median) / mad
    keep = np.all(deviation <= threshold, axis=1)
    return samples[keep]


# --------------------------------------------------------------------------
# Small math helpers
# --------------------------------------------------------------------------

def to_pixel(norm_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """[FIX 1] The ONE place normalized [0,1] coords become pixel coords.

    Used identically by calibration, validation, and jitter so their
    coordinate spaces can never drift apart again.
    """
    return np.multiply(norm_xy, np.array([width - 1, height - 1], dtype=np.float64))


def rotation_matrix_to_euler(rot: np.ndarray) -> np.ndarray:
    """Yaw/pitch/roll (radians) from a 3x3 rotation matrix."""
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rot[2, 1], rot[2, 2])
        yaw = np.arctan2(-rot[2, 0], sy)
        roll = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        pitch = np.arctan2(-rot[1, 2], rot[1, 1])
        yaw = np.arctan2(-rot[2, 0], sy)
        roll = 0.0
    return np.array([yaw, pitch, roll], dtype=np.float64)

def setup_paths(cfg) -> Paths:
    output_dir = Path(cfg.output_dir)
    validation_output_dir = output_dir / "validation_data" / "validation"
    jitter_output_dir = output_dir / "validation_data" / "jitter"
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    jitter_output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    return Paths(output_dir, validation_output_dir, jitter_output_dir, run_stamp)


def setup_window(window_name: str) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)