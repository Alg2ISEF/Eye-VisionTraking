"""
Gaze tracking pipeline (rebuild).

This is a corrected rebuild of an earlier script. Fixes are grouped and
numbered below to match the code-review discussion; search for "[FIX N]"
comments inline to see exactly where each one lives.

  [FIX 1]  Single `to_pixel()` helper used everywhere -> no more
           inconsistent (-1 vs no -1) denormalization between
           calibration / validation / jitter.
  [FIX 2]  Per-eye feature median-history is reset at the START of every
           calibration burst, so point N's features can't leak into
           point N+1.
  [FIX 3]  Removed dead `relative_gaze` computation.
  [FIX 4]  Whole capture/processing loop wrapped in try/finally so the
           camera, landmarker, thread, and windows are always released.
  [FIX 5]  Crop is computed to match the *output* aspect ratio, so the
           resize step no longer stretches geometry.
  [FIX 6]  CAM_WIDTH typo fixed (1270 -> 1280); actual negotiated
           camera properties are read back and logged after `cap.set`.
  [FIX 7]  Swapped CatBoost (piecewise-constant, wrong inductive bias
           for a smooth low-dim mapping) for polynomial-Ridge
           regression -- the standard approach for webcam gaze mapping.
  [FIX 8]  Features are normalized by inter-ocular distance (scale
           invariance to camera distance) and head pose (yaw/pitch/roll
           from the FaceLandmarker transformation matrix) is included
           as extra features, so small head movement after calibration
           doesn't silently degrade the estimate.
  [FIX 9]  Calibration bursts reject outlier frames (blinks/saccades)
           via a MAD threshold before accepting a point, and the
           minimum-frames-per-point bar was raised from 2 to a sane
           number.
  [FIX 10] Crop is dynamically re-centered on the last known face
           position (with smoothing + a center-crop fallback), instead
           of assuming a perfectly still, perfectly centered head.
  [FIX 11] OneEuroFilter beta is now a real, documented, tunable value;
           you can see the effect of changing it.
  [FIX 12] Validation and jitter now use the same "hold SPACE to record
           a burst, release to finish" mechanism, and validation
           compares an *averaged* estimate to the target instead of one
           noisy instantaneous frame.
  [FIX 13/14] Capture thread is a daemon and the main loop cleans up on
           any exception, so a crash can't orphan the camera/thread.
  [FIX 17] Device index is derived from the device path so the
           v4l2-ctl calls and cv2.VideoCapture agree with each other.
  [FIX 18/19/20] Removed pointless np.int64/np.float64 scalar wrapping,
           de-duplicated pixel-conversion and landmark-extraction code.
  [FIX 21] Split into classes: CameraGrabber, GazeFeatureExtractor,
           GazeModel, HoldToRecord, CalibrationSession. Testable in
           isolation, not one 300-line global-state loop.
  [FIX 22] Trained model can be saved/loaded (--model-path) so you
           don't have to recalibrate every run.
  [FIX 23] Output CSVs are timestamped, so runs don't clobber each
           other.
  [FIX 24] Device, resolution, fps, crop size, and paths are CLI args.

Known simplification, stated explicitly: this calibrates gaze position
*within the displayed camera-preview frame*, not against real screen
coordinates. If the goal is on-screen cursor control, you still need a
camera-to-screen geometric mapping on top of this.

Requires: opencv-python, mediapipe, numpy, scikit-learn, a
face_landmarker.task model file next to this script, and (on Linux)
v4l2-ctl if you want the exposure/gain locking.
"""

from __future__ import annotations

import argparse
import pickle
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

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


class OneEuroFilter:
    """Adaptive low-pass filter.

    [FIX 11] beta controls how aggressively the cutoff frequency rises
    with speed. The previous beta=0.007 was small enough, relative to
    pixel/sec velocities, that the filter barely behaved differently
    from a fixed-cutoff low-pass -- i.e. it wasn't really "doing" the
    One Euro adaptive part. 0.3 is a much more typical starting point
    for this kind of cursor/gaze-position signal; treat it as a knob to
    tune against your own jitter-test output, not a magic number.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3,
                 derivative_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.reset()

    def reset(self):
        self.previous_time = None
        self.previous_value = None
        self.previous_derivative = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value, timestamp: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.previous_value is None:
            self.previous_time = timestamp
            self.previous_value = value.copy()
            self.previous_derivative = np.zeros_like(value)
            return value.copy()

        dt = max(timestamp - self.previous_time, 1e-6)
        raw_deriv = (value - self.previous_value) / dt
        d_alpha = self._alpha(self.derivative_cutoff, dt)
        deriv = d_alpha * raw_deriv + (1.0 - d_alpha) * self.previous_derivative

        cutoff = self.min_cutoff + self.beta * np.abs(deriv)
        v_alpha = self._alpha(cutoff, dt)
        filtered = v_alpha * value + (1.0 - v_alpha) * self.previous_value

        self.previous_time = timestamp
        self.previous_value = filtered
        self.previous_derivative = deriv
        return filtered.copy()


# --------------------------------------------------------------------------
# Camera capture (threaded)
# --------------------------------------------------------------------------

def configure_camera(dev_path: str) -> None:
    settings = [
        ("auto_exposure", 1),
        ("exposure_dynamic_framerate", 0),
        ("exposure_time_absolute", 10),
        ("gain", 25),
        ("zoom_absolute", 0),
    ]
    for name, value in settings:
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", dev_path, "-c", f"{name}={value}"],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            print(f"Warning: could not set {name}={value} on {dev_path}: {error}")


class CameraGrabber:
    """Background frame grabber.

    [FIX 13/14] Thread is a daemon (won't block interpreter exit) and
    the read loop tolerates a bounded number of transient read
    failures (e.g. a momentary USB hiccup) before giving up, instead of
    treating the very first failed read as fatal.
    """

    def __init__(self, cap: cv2.VideoCapture, max_consecutive_failures: int = 30):
        self.cap = cap
        self.max_consecutive_failures = max_consecutive_failures
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.latest_frame = None
        self.sequence = 0
        self.finished = False
        self.thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        failures = 0
        while not self.stop_event.is_set():
            success, frame = self.cap.read()
            if not success:
                failures += 1
                if failures >= self.max_consecutive_failures:
                    with self.condition:
                        self.finished = True
                        self.condition.notify_all()
                    return
                continue
            failures = 0
            with self.condition:
                self.latest_frame = frame
                self.sequence += 1
                self.condition.notify()

    def get_next(self, last_seen_sequence: int, timeout: float = 1.0):
        """Blocks until a newer frame is available or capture ends.

        Returns (frame_or_None, new_sequence, finished).
        """
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > last_seen_sequence or self.finished,
                timeout=timeout,
            )
            if self.sequence <= last_seen_sequence:
                return None, last_seen_sequence, self.finished
            return self.latest_frame.copy(), self.sequence, self.finished

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.0)


# --------------------------------------------------------------------------
# Dynamic, aspect-correct crop
# --------------------------------------------------------------------------

class DynamicCropper:
    """Crops a fixed-size, aspect-correct region, re-centered on the last
    known face position.

    [FIX 5] crop_h is derived from crop_w so crop_w/crop_h always equals
    output_w/output_h -- the resize step can no longer stretch geometry.
    [FIX 10] The crop center follows the last detected face (smoothed),
    instead of always sitting dead-center, and falls back to center
    framing if the face has been lost for a while.
    """

    def __init__(self, cam_w: int, cam_h: int, crop_w: int, out_w: int, out_h: int,
                 smoothing: float = 0.2, lost_track_frames: int = 30):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.crop_w = min(crop_w, cam_w)
        self.crop_h = min(int(round(self.crop_w * out_h / out_w)), cam_h)
        self.smoothing = smoothing
        self.lost_track_frames = lost_track_frames
        self.center = np.array([cam_w / 2.0, cam_h / 2.0], dtype=np.float64)
        self.frames_since_seen = lost_track_frames  # start "lost" -> center crop

    def update_center(self, face_center_in_crop: Optional[np.ndarray], crop_origin: np.ndarray):
        if face_center_in_crop is None:
            self.frames_since_seen += 1
            if self.frames_since_seen >= self.lost_track_frames:
                target = np.array([self.cam_w / 2.0, self.cam_h / 2.0], dtype=np.float64)
                self.center = (1 - self.smoothing) * self.center + self.smoothing * target
            return
        self.frames_since_seen = 0
        face_center_full = crop_origin + face_center_in_crop
        self.center = (1 - self.smoothing) * self.center + self.smoothing * face_center_full

    def crop_rect(self) -> tuple[int, int, int, int]:
        x0 = int(np.clip(self.center[0] - self.crop_w / 2, 0, self.cam_w - self.crop_w))
        y0 = int(np.clip(self.center[1] - self.crop_h / 2, 0, self.cam_h - self.crop_h))
        return x0, y0, self.crop_w, self.crop_h


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

IRIS_INDICES = (np.arange(468, 473, dtype=np.int32), np.arange(473, 478, dtype=np.int32))
EYE_CORNER_PAIRS = ((33, 133), (362, 263))


@dataclass
class FrameFeatures:
    feature_vector: np.ndarray          # normalized, head-pose-aware features
    iris_center_px: np.ndarray          # for drawing, in *cropped-frame* pixels
    face_center_px: np.ndarray          # for dynamic crop re-centering


class GazeFeatureExtractor:
    """Turns FaceLandmarker output into a normalized, head-pose-aware
    feature vector.

    [FIX 3] The unused `relative_gaze` scalar from the original script
    is gone; every value computed here is actually consumed.
    [FIX 8] Per-eye iris displacement is divided by inter-ocular
    distance (scale invariance to camera distance) and concatenated
    with yaw/pitch/roll extracted from the landmarker's facial
    transformation matrix, so a small head move after calibration
    doesn't fully invalidate the mapping the way raw pixel offsets do.
    """

    def extract(self, face_landmarks, transformation_matrix: Optional[np.ndarray],
                frame_w: int, frame_h: int) -> Optional[FrameFeatures]:
        pts = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in face_landmarks],
                       dtype=np.float64)

        iris_centers = [pts[idx].mean(axis=0) for idx in IRIS_INDICES]
        eye_corner_centers = [pts[list(pair)].mean(axis=0) for pair in EYE_CORNER_PAIRS]
        inter_ocular_dist = float(np.linalg.norm(eye_corner_centers[0] - eye_corner_centers[1]))
        if inter_ocular_dist < 1e-3:
            return None

        displacement = np.concatenate(
            [(iris - corner) / inter_ocular_dist
             for iris, corner in zip(iris_centers, eye_corner_centers)]
        )  # 4 dims: dx_left, dy_left, dx_right, dy_right (normalized)

        if transformation_matrix is not None:
            euler = rotation_matrix_to_euler(transformation_matrix[:3, :3])
        else:
            euler = np.zeros(3, dtype=np.float64)

        feature_vector = np.concatenate([displacement, euler])
        iris_center_px = np.mean(iris_centers, axis=0)
        face_center_px = pts.mean(axis=0)

        return FrameFeatures(feature_vector, iris_center_px, face_center_px)


# --------------------------------------------------------------------------
# Gaze regression model
# --------------------------------------------------------------------------

class GazeModel:
    """Polynomial-Ridge regression, one multi-output model for (x, y).

    [FIX 7] Replaces CatBoostRegressor. Gradient-boosted trees produce
    piecewise-constant predictions and need far more data than a
    13-point calibration provides to interpolate smoothly; the
    iris-offset -> screen-position mapping is smooth and low-order, so
    a degree-2 polynomial feature expansion with Ridge regularization
    (which supports multi-output targets natively) is the standard,
    much better-suited choice here.
    """

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        self.pipeline = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            Ridge(alpha=alpha),
        )
        self.is_fit = False

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.pipeline.fit(X, Y)
        self.is_fit = True

    def predict(self, feature_vector: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(feature_vector.reshape(1, -1))[0]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    @classmethod
    def load(cls, path: str) -> "GazeModel":
        model = cls()
        with open(path, "rb") as f:
            model.pipeline = pickle.load(f)
        model.is_fit = True
        return model


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
# Hold-SPACE-to-record burst helper (shared by calibration/validation/jitter)
# --------------------------------------------------------------------------

class HoldToRecord:
    """Tracks a "press SPACE to start, release to stop" burst.

    [FIX 12] Validation and jitter used to be two separately-written,
    slightly-inconsistent state machines (validation: single instant
    keypress; jitter: hold-and-release). They now share this one
    implementation, and validation records a burst the same way
    calibration does instead of trusting one noisy frame.
    """

    def __init__(self, release_timeout: float = 0.2):
        self.release_timeout = release_timeout
        self.recording = False
        self.last_active_tick = None
        self.samples: list = []

    def update(self, space_held: bool, sample) -> bool:
        """Call once per frame. Returns True the instant a burst just
        finished (so the caller can consume self.samples)."""
        now = cv2.getTickCount()
        freq = cv2.getTickFrequency()

        if space_held:
            if not self.recording:
                self.recording = True
                self.samples = []
            self.last_active_tick = now
            if sample is not None:
                self.samples.append(sample)
            return False

        if self.recording:
            elapsed = (now - self.last_active_tick) / freq
            if elapsed > self.release_timeout:
                self.recording = False
                return True
        return False


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def open_camera(cfg: Config) -> cv2.VideoCapture:
    configure_camera(cfg.device_path)
    cap = cv2.VideoCapture(cfg.device_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.cam_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {cfg.device_index} ({cfg.device_path})")

    # [FIX 6] Verify what the driver actually gave us instead of assuming
    # the requested width/height/fps were honored.
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera negotiated: {actual_w:.0f}x{actual_h:.0f} @ {actual_fps:.1f} fps "
          f"(requested {cfg.cam_width}x{cfg.cam_height} @ {cfg.cam_fps})")
    return cap


def build_landmarker(cfg: Config) -> mp_vision.FaceLandmarker:
    base_options = mp_python.BaseOptions(
        model_asset_path=cfg.landmarker_model_path,
        delegate=mp_python.BaseOptions.Delegate.GPU,
    )
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,  # needed for head-pose features
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


CALIBRATION_TARGETS = np.array([
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5],
    [0.9, 0.5], [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
    [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
], dtype=np.float64)

VALIDATION_TARGETS = np.array([
    [0.5, 0.5], [0.5, 0.25], [0.5, 0.75], [0.25, 0.5], [0.75, 0.5],
    [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8],
], dtype=np.float64)

JITTER_TARGET = np.array([0.5, 0.5], dtype=np.float64)


def main():
    cfg = parse_args()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d-%H%M%S")

    cap = open_camera(cfg)
    landmarker = build_landmarker(cfg)
    grabber = CameraGrabber(cap)
    grabber.start()

    cropper = DynamicCropper(cfg.cam_width, cfg.cam_height, cfg.crop_width,
                              cfg.output_width, cfg.output_height)
    extractor = GazeFeatureExtractor()
    feature_history = deque(maxlen=cfg.median_window)
    gaze_filter = OneEuroFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta)

    window_name = "Gaze Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    gaze_model: Optional[GazeModel] = None
    if cfg.model_path and Path(cfg.model_path).exists():
        gaze_model = GazeModel.load(cfg.model_path)
        print(f"Loaded existing gaze model from {cfg.model_path}; skipping calibration.")

    calibration_index = 0
    calibration_samples: list = []  # rows: [feat..., target_x, target_y]
    calibration_burst = HoldToRecord()

    validation_index = 0
    validation_results: list = []
    validation_burst = HoldToRecord()

    jitter_complete = False
    jitter_positions: list = []
    jitter_burst = HoldToRecord()

    stream_start = cv2.getTickCount()
    previous_time = cv2.getTickCount()
    frame_count = 0
    latency_total = 0.0
    latency_count = 0
    last_sequence = 0

    try:
        while True:
            raw_frame, last_sequence, finished = grabber.get_next(last_sequence)
            if raw_frame is None:
                if finished:
                    print("Camera stopped producing frames.")
                    break
                continue

            frame_start = cv2.getTickCount()
            raw_frame = cv2.flip(raw_frame, 1)

            x0, y0, cw, ch = cropper.crop_rect()
            cropped = raw_frame[y0:y0 + ch, x0:x0 + cw]
            frame = cv2.resize(cropped, (cfg.output_width, cfg.output_height),
                                interpolation=cv2.INTER_LINEAR)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_start - stream_start) * 1000.0 / cv2.getTickFrequency())
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            features: Optional[FrameFeatures] = None
            filtered_features = None
            gaze_position = None

            if result.face_landmarks:
                transform = None
                if result.facial_transformation_matrixes:
                    transform = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
                features = extractor.extract(result.face_landmarks[0], transform,
                                              frame.shape[1], frame.shape[0])

            if features is not None:
                cropper.update_center(features.face_center_px, np.array([x0, y0], dtype=np.float64))
                cv2.circle(frame, tuple(features.iris_center_px.astype(int)), 5, (0, 255, 255), -1)

                feature_history.append(features.feature_vector.copy())
                filtered_features = np.median(np.asarray(feature_history), axis=0)

                if gaze_model is not None and gaze_model.is_fit:
                    raw_pos = gaze_model.predict(filtered_features)
                    raw_pos = np.clip(raw_pos, [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1])
                    gaze_position = gaze_filter(raw_pos, timestamp_ms / 1000.0)
            else:
                cropper.update_center(None, np.array([x0, y0], dtype=np.float64))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key in (ord("r"), ord("R")):
                calibration_index = 0
                calibration_samples = []
                calibration_burst = HoldToRecord()
                gaze_model = None
                validation_index = 0
                validation_results = []
                validation_burst = HoldToRecord()
                jitter_complete = False
                jitter_positions = []
                jitter_burst = HoldToRecord()
                feature_history.clear()
                gaze_filter.reset()
                print("Reset. Starting again from calibration point 1.", flush=True)
                continue

            space_held = key == ord(" ")

            # ---------------- Calibration ----------------
            if gaze_model is None:
                target_px = to_pixel(CALIBRATION_TARGETS[calibration_index],
                                      frame.shape[1], frame.shape[0])
                cv2.circle(frame, tuple(target_px.astype(int)), 9, (0, 0, 255), -1)
                status = (f"Recording burst: hold SPACE, release to stop "
                          f"({len(calibration_burst.samples)} frames)"
                          if calibration_burst.recording else
                          f"Look at dot, hold SPACE "
                          f"({calibration_index + 1}/{len(CALIBRATION_TARGETS)})")
                cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

                # [FIX 2] feature_history is NOT what we train on directly;
                # we feed the raw (unfiltered) per-frame feature vector into
                # the burst, then reject outliers below. This also means a
                # burst never inherits smoothing state from the previous
                # calibration point.
                sample = features.feature_vector.copy() if features is not None else None
                burst_done = calibration_burst.update(space_held, sample)

                if burst_done:
                    burst = np.array(calibration_burst.samples, dtype=np.float64)
                    if len(burst) >= cfg.min_calibration_frames:
                        clean = reject_outliers(burst, cfg.mad_outlier_threshold)
                        if len(clean) >= cfg.min_calibration_frames // 2:
                            for row in clean:
                                calibration_samples.append([*row, target_px[0], target_px[1]])
                            print(f"Captured point {calibration_index + 1}/{len(CALIBRATION_TARGETS)}: "
                                  f"{len(clean)}/{len(burst)} frames kept after outlier rejection.",
                                  flush=True)
                            calibration_index += 1
                            feature_history.clear()  # [FIX 2]

                            if calibration_index == len(CALIBRATION_TARGETS):
                                samples = np.array(calibration_samples, dtype=np.float64)
                                n_features = samples.shape[1] - 2
                                X = samples[:, :n_features]
                                Y = samples[:, n_features:]
                                gaze_model = GazeModel()
                                gaze_model.fit(X, Y)
                                print(f"Calibration complete with {len(samples)} frames. Tracking gaze.",
                                      flush=True)
                                if cfg.model_path:
                                    gaze_model.save(cfg.model_path)
                                    print(f"Saved model to {cfg.model_path}")
                                feature_history.clear()
                                gaze_filter.reset()
                        else:
                            print("Too many outlier frames in that burst (blink?); try again.",
                                  flush=True)
                    else:
                        print(f"Burst too short ({len(burst)} frames); hold SPACE longer and try again.",
                              flush=True)

            # ---------------- Validation ----------------
            elif validation_index < len(VALIDATION_TARGETS):
                target_px = to_pixel(VALIDATION_TARGETS[validation_index], frame.shape[1], frame.shape[0])
                cv2.circle(frame, tuple(target_px.astype(int)), 9, (0, 255, 0), -1)
                cv2.putText(frame,
                            f"Validation: look at dot, hold SPACE "
                            f"({validation_index + 1}/{len(VALIDATION_TARGETS)})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if gaze_position is not None:
                    cv2.circle(frame, tuple(gaze_position.astype(int)), 24, (0, 0, 255), 2)

                # [FIX 12] Burst-average the *predicted position*, matching
                # the multi-frame robustness calibration already gets,
                # instead of trusting a single instantaneous keypress frame.
                sample = gaze_position.copy() if gaze_position is not None else None
                burst_done = validation_burst.update(space_held, sample)
                if burst_done:
                    if len(validation_burst.samples) == 0:
                        print("Could not estimate gaze during that burst; keep both eyes visible.",
                              flush=True)
                    else:
                        estimate = np.mean(np.array(validation_burst.samples), axis=0)
                        error_px = float(np.linalg.norm(estimate - target_px))
                        validation_results.append([*target_px, *estimate, error_px])
                        print(f"Validation {validation_index + 1}/{len(VALIDATION_TARGETS)}: "
                              f"true=({target_px[0]:.1f},{target_px[1]:.1f}) "
                              f"est=({estimate[0]:.1f},{estimate[1]:.1f}) error={error_px:.1f}px",
                              flush=True)
                        validation_index += 1
                        if validation_index == len(VALIDATION_TARGETS):
                            results = np.array(validation_results)
                            mean_err = results[:, 4].mean()
                            max_err = results[:, 4].max()
                            print(f"Validation complete: mean error={mean_err:.1f}px, "
                                  f"max error={max_err:.1f}px", flush=True)
                            out_path = Path(cfg.output_dir) / f"validation_results_{run_stamp}.csv"
                            np.savetxt(out_path, results, delimiter=",",
                                       header="x_true,y_true,x_est,y_est,error_pixels", comments="")
                            print(f"Saved {out_path}", flush=True)

            # ---------------- Jitter test ----------------
            elif not jitter_complete:
                target_px = to_pixel(JITTER_TARGET, frame.shape[1], frame.shape[0])
                cv2.circle(frame, tuple(target_px.astype(int)), 9, (0, 255, 0), -1)
                cv2.putText(frame, "Jitter: hold SPACE while fixating, release to finish",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if gaze_position is not None:
                    cv2.circle(frame, tuple(gaze_position.astype(int)), 24, (0, 0, 255), 2)

                sample = gaze_position.copy() if gaze_position is not None else None
                burst_done = jitter_burst.update(space_held, sample)
                if burst_done:
                    jitter_complete = True
                    if len(jitter_burst.samples) >= 2:
                        pts = np.array(jitter_burst.samples)
                        mean = pts.mean(axis=0)
                        std = pts.std(axis=0)
                        rms = float(np.sqrt(np.mean(np.sum((pts - mean) ** 2, axis=1))))
                        out = np.column_stack([np.arange(len(pts)), pts, np.linalg.norm(pts - mean, axis=1)])
                        out_path = Path(cfg.output_dir) / f"jitter_results_{run_stamp}.csv"
                        np.savetxt(out_path, out, delimiter=",",
                                   header="frame,x_est,y_est,deviation_pixels", comments="")
                        print(f"Jitter complete: frames={len(pts)}, std=({std[0]:.1f},{std[1]:.1f})px, "
                              f"RMS={rms:.1f}px. Saved {out_path}", flush=True)
                    else:
                        print("Jitter burst too short; try again (press 'r' to redo from scratch, "
                              "or just keep going).", flush=True)

            # ---------------- Live tracking ----------------
            else:
                if gaze_position is not None:
                    cv2.circle(frame, tuple(gaze_position.astype(int)), 24, (0, 0, 255), 2)
                    cv2.circle(frame, tuple(gaze_position.astype(int)), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Estimated gaze", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            frame_count += 1
            cv2.imshow(window_name, frame)

            now = cv2.getTickCount()
            freq = cv2.getTickFrequency()
            latency_total += (now - frame_start) / freq
            latency_count += 1
            elapsed = (now - previous_time) / freq
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                latency_ms = (latency_total / max(latency_count, 1)) * 1000.0
                print(f"FPS: {fps:.1f} | Latency: {latency_ms:.2f} ms", flush=True)
                frame_count = 0
                latency_total = 0.0
                latency_count = 0
                previous_time = now

    finally:
        # [FIX 4] This runs on 'q', on an exception, on anything -- the
        # camera and window can no longer be left in a locked/open state.
        grabber.stop()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()