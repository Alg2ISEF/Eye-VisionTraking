from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from filter import OneEuroFilter
import gaze_model as gaze
from gaze_model import Session
import config
import camera
from camera import open_camera, grab_and_prepare_frame
import cv2
import numpy as np
from helper import reject_outliers, to_pixel , Paths , setup_paths , setup_window 
from calibration_phase import handle_calibration_phase , handle_reset
from validation_phase import handle_validation_phase , handle_jitter_phase , handle_live_tracking , VALIDATION_TARGETS  
import mediapipe as mp

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

@dataclass
class FpsCounter:
    previous_time: int
    frame_count: int = 0
    latency_total: float = 0.0
    latency_count: int = 0

    def update(self, frame_start: int) -> None:
        now = cv2.getTickCount()
        freq = cv2.getTickFrequency()
        self.latency_total += (now - frame_start) / freq
        self.latency_count += 1
        self.frame_count += 1

        elapsed = (now - self.previous_time) / freq
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            latency_ms = (self.latency_total / max(self.latency_count, 1)) * 1000.0
            print(f"FPS: {fps:.1f} | Latency: {latency_ms:.2f} ms", flush=True)
            self.frame_count = 0
            self.latency_total = 0.0
            self.latency_count = 0
            self.previous_time = now


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------


def load_existing_model(cfg) -> Optional[gaze.GazeModel]:
    if cfg.model_path and Path(cfg.model_path).exists():
        model = gaze.GazeModel.load(cfg.model_path)
        print(f"Loaded existing gaze model from {cfg.model_path}; skipping calibration.")
        return model
    return None


# --------------------------------------------------------------------------
# Per-frame acquisition / feature extraction
# --------------------------------------------------------------------------


def detect_features(frame, landmarker, extractor: gaze.GazeFeatureExtractor,
                     stream_start: int, frame_start: int) -> Optional[gaze.FrameFeatures]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((frame_start - stream_start) * 1000.0 / cv2.getTickFrequency())
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return None, timestamp_ms

    transform = None
    if result.facial_transformation_matrixes:
        transform = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
    features = extractor.extract(result.face_landmarks[0], transform,
                                  frame.shape[1], frame.shape[0])
    return features, timestamp_ms


def update_filtered_gaze(frame, features: Optional[gaze.FrameFeatures],
                          feature_history: deque, gaze_model: Optional[gaze.GazeModel],
                          gaze_filter: OneEuroFilter, timestamp_ms: int):
    """Draws the iris marker, updates the median feature history, and (if a
    model is fitted) returns a filtered gaze position. Returns
    (filtered_features, gaze_position)."""
    if features is None:
        return None, None

    cv2.circle(frame, tuple(features.iris_center_px.astype(int)), 5, (0, 255, 255), -1)

    feature_history.append(features.feature_vector.copy())
    filtered_features = np.median(np.asarray(feature_history), axis=0)

    gaze_position = None
    if gaze_model is not None and gaze_model.is_fit:
        raw_pos = gaze_model.predict(filtered_features)
        raw_pos = np.clip(raw_pos, [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1])
        gaze_position = gaze_filter(raw_pos, timestamp_ms / 1000.0)

    return filtered_features, gaze_position


def dispatch_phase(frame, features, gaze_position, session: Session,
                    space_held: bool, cfg, paths: Paths, feature_history: deque) -> None:
    """Runs whichever phase (calibration / validation / jitter / live) is
    currently active."""
    if session.gaze_model is None:
        handle_calibration_phase(frame, features, feature_history, session, space_held, cfg)
    elif session.validation_index < len(VALIDATION_TARGETS):
        handle_validation_phase(frame, gaze_position, session, space_held, paths)
    elif not session.jitter_complete:
        handle_jitter_phase(frame, gaze_position, session, space_held, paths)
    else:
        handle_live_tracking(frame, gaze_position)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def run_loop(cfg, paths: Paths, window_name: str) -> None:
    cap = open_camera(cfg)
    landmarker = gaze.build_landmarker(cfg)
    grabber = camera.CameraGrabber(cap)
    grabber.start()

    cropper = camera.DynamicCropper(cfg.cam_width, cfg.cam_height, cfg.crop_width,
                                     cfg.output_width, cfg.output_height)
    extractor = gaze.GazeFeatureExtractor()
    feature_history = deque(maxlen=cfg.median_window)
    gaze_filter = OneEuroFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta)

    session = Session(gaze_model=load_existing_model(cfg))

    stream_start = cv2.getTickCount()
    fps_counter = FpsCounter(previous_time=cv2.getTickCount())
    last_sequence = 0

    try:
        while True:
            frame, last_sequence, finished, frame_start = grab_and_prepare_frame(
                grabber, last_sequence, cropper, cfg)
            if frame is None:
                if finished:
                    print("Camera stopped producing frames.")
                    break
                continue

            features, timestamp_ms = detect_features(frame, landmarker, extractor,
                                                       stream_start, frame_start)
            _, gaze_position = update_filtered_gaze(
                frame, features, feature_history, session.gaze_model,
                gaze_filter, timestamp_ms)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("r"), ord("R")):
                handle_reset(session, feature_history, gaze_filter)
                continue

            space_held = key == ord(" ")
            dispatch_phase(frame, features, gaze_position, session, space_held,
                            cfg, paths, feature_history)

            cv2.imshow(window_name, frame)
            fps_counter.update(frame_start)

    finally:
        grabber.stop()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


def main():
    cfg = config.parse_args()
    paths = setup_paths(cfg)
    window_name = "Gaze Tracker"
    setup_window(window_name)
    run_loop(cfg, paths, window_name)


if __name__ == "__main__":
    main()