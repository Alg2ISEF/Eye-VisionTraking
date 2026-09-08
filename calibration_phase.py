
from camera import HoldToRecord 
from helper import to_pixel  , reject_outliers

from filter import OneEuroFilter
import cv2
import numpy as np
import gaze_model as gaze
from gaze_model import Session
from collections import deque
from typing import Optional
# --------------------------------------------------------------------------
# Phase handlers
# --------------------------------------------------------------------------

CALIBRATION_TARGETS = np.array([
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5],
    [0.9, 0.5], [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
    [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
], dtype=np.float64)


def handle_reset(session: Session, feature_history: deque, gaze_filter: OneEuroFilter) -> None:
    session.reset()
    feature_history.clear()
    gaze_filter.reset()
    print("Reset. Starting again from calibration point 1.", flush=True)


def handle_calibration_phase(frame, features: Optional[gaze.FrameFeatures],
                              feature_history: deque, session: Session,
                              space_held: bool, cfg) -> None:
    target_px = to_pixel(CALIBRATION_TARGETS[session.calibration_index],
                          frame.shape[1], frame.shape[0])
    cv2.circle(frame, tuple(target_px.astype(int)), 9, (0, 0, 255), -1)
    status = (f"Recording burst: press SPACE to stop "
              f"({len(session.calibration_burst.samples)} frames)"
              if session.calibration_burst.recording else
              f"Look at dot, press SPACE to start "
              f"({session.calibration_index + 1}/{len(CALIBRATION_TARGETS)})")
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # [FIX 2] feature_history is NOT what we train on directly;
    # we feed the raw (unfiltered) per-frame feature vector into
    # the burst, then reject outliers below. This also means a
    # burst never inherits smoothing state from the previous
    # calibration point.
    sample = features.feature_vector.copy() if features is not None else None
    if space_held and not session.calibration_burst.recording:
        feature_history.clear()
    burst_done = session.calibration_burst.update(space_held, sample)

    if burst_done:
        _finish_calibration_burst(session, feature_history, cfg, target_px)


def _finish_calibration_burst(session: Session, feature_history: deque, cfg,
                               target_px) -> None:
    burst = np.array(session.calibration_burst.samples, dtype=np.float64)
    if len(burst) < cfg.min_calibration_frames:
        print(f"Burst too short ({len(burst)} frames); press SPACE again to retry.",
              flush=True)
        return

    clean = reject_outliers(burst, cfg.mad_outlier_threshold)
    if len(clean) < cfg.min_calibration_frames // 2:
        print("Too many outlier frames in that burst (blink?); try again.",
              flush=True)
        return

    for row in clean:
        session.calibration_samples.append([*row, target_px[0], target_px[1]])
    print(f"Captured point {session.calibration_index + 1}/{len(CALIBRATION_TARGETS)}: "
          f"{len(clean)}/{len(burst)} frames kept after outlier rejection.",
          flush=True)
    session.calibration_index += 1
    feature_history.clear()  # [FIX 2]

    if session.calibration_index == len(CALIBRATION_TARGETS):
        _fit_gaze_model(session, feature_history, cfg)


def _fit_gaze_model(session: Session, feature_history: deque, cfg,
                     gaze_filter: Optional[OneEuroFilter] = None) -> None:
    samples = np.array(session.calibration_samples, dtype=np.float64)
    n_features = samples.shape[1] - 2
    X = samples[:, :n_features]
    Y = samples[:, n_features:]
    session.gaze_model = gaze.GazeModel()
    session.gaze_model.fit(X, Y)
    print(f"Calibration complete with {len(samples)} frames. Tracking gaze.",
          flush=True)
    if cfg.model_path:
        session.gaze_model.save(cfg.model_path)
        print(f"Saved model to {cfg.model_path}")
    feature_history.clear()
    if gaze_filter is not None:
        gaze_filter.reset()