import threading
import numpy as np
from typing import Optional
import cv2
import subprocess
import config
import time
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

def open_camera(cfg: config.Config) -> cv2.VideoCapture:
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

# --------------------------------------------------------------------------
# Hold-SPACE-to-record burst helper (shared by calibration/validation/jitter)
# --------------------------------------------------------------------------

class HoldToRecord:
    """Records a burst between two distinct Space presses."""

    def __init__(self, toggle_cooldown: float = 0.25):
        self.toggle_cooldown = toggle_cooldown
        self.recording = False
        self.last_toggle_time = -np.inf
        self.samples: list = []

    def update(self, space_pressed: bool, sample) -> bool:
        now = time.monotonic()
        if space_pressed and now - self.last_toggle_time >= self.toggle_cooldown:
            self.last_toggle_time = now
            if self.recording:
                self.recording = False
                return True
            self.recording = True
            self.samples = []
            if sample is not None:
                self.samples.append(sample)
            return False
        if self.recording and sample is not None:
            self.samples.append(sample)
        return False

    
def grab_and_prepare_frame(grabber: CameraGrabber, last_sequence: int,
                            cropper: DynamicCropper, cfg):
    """Pull the next raw frame and crop/resize it. Returns
    (frame, last_sequence, finished, frame_start) where frame is None if no
    new frame was available yet."""
    raw_frame, last_sequence, finished = grabber.get_next(last_sequence)
    if raw_frame is None:
        return None, last_sequence, finished, None

    frame_start = cv2.getTickCount()
    raw_frame = cv2.flip(raw_frame, 1)

    x0, y0, cw, ch = cropper.crop_rect()
    cropped = raw_frame[y0:y0 + ch, x0:x0 + cw]
    frame = cv2.resize(cropped, (cfg.output_width, cfg.output_height),
                        interpolation=cv2.INTER_LINEAR)
    return frame, last_sequence, finished, frame_start