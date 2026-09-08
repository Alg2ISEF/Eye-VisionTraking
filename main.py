from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from filter import OneEuroFilter
import gaze_model as gaze
import config
import camera
from camera import open_camera , HoldToRecord
import cv2
import numpy as np
from helper import reject_outliers
import mediapipe as mp

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

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
    cfg = config.parse_args()
    output_dir = Path(cfg.output_dir)
    validation_output_dir = output_dir / "validation_data" / "validation"
    jitter_output_dir = output_dir / "validation_data" / "jitter"
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    jitter_output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d-%H%M%S")

    cap = open_camera(cfg)
    landmarker = gaze.build_landmarker(cfg)
    grabber = camera.CameraGrabber(cap)
    grabber.start()

    cropper = camera.DynamicCropper(cfg.cam_width, cfg.cam_height, cfg.crop_width,
                              cfg.output_width, cfg.output_height)
    extractor = gaze.GazeFeatureExtractor()
    feature_history = deque(maxlen=cfg.median_window)
    gaze_filter = OneEuroFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta)

    window_name = "Gaze Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    gaze_model: Optional[gaze.GazeModel] = None
    if cfg.model_path and Path(cfg.model_path).exists():
        gaze_model = gaze.GazeModel.load(cfg.model_path)
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

            features: Optional[gaze.FrameFeatures] = None
            filtered_features = None
            gaze_position = None

            if result.face_landmarks:
                transform = None
                if result.facial_transformation_matrixes:
                    transform = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
                features = extractor.extract(result.face_landmarks[0], transform,
                                              frame.shape[1], frame.shape[0])

            if features is not None:
                cv2.circle(frame, tuple(features.iris_center_px.astype(int)), 5, (0, 255, 255), -1)

                feature_history.append(features.feature_vector.copy())
                filtered_features = np.median(np.asarray(feature_history), axis=0)

                if gaze_model is not None and gaze_model.is_fit:
                    raw_pos = gaze_model.predict(filtered_features)
                    raw_pos = np.clip(raw_pos, [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1])
                    gaze_position = gaze_filter(raw_pos, timestamp_ms / 1000.0)
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
                status = (f"Recording burst: press SPACE to stop "
                          f"({len(calibration_burst.samples)} frames)"
                          if calibration_burst.recording else
                          f"Look at dot, press SPACE to start "
                          f"({calibration_index + 1}/{len(CALIBRATION_TARGETS)})")
                cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

                # [FIX 2] feature_history is NOT what we train on directly;
                # we feed the raw (unfiltered) per-frame feature vector into
                # the burst, then reject outliers below. This also means a
                # burst never inherits smoothing state from the previous
                # calibration point.
                sample = features.feature_vector.copy() if features is not None else None
                if space_held and not calibration_burst.recording:
                    feature_history.clear()
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
                                gaze_model = gaze.GazeModel()
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
                        print(f"Burst too short ({len(burst)} frames); press SPACE again to retry.",
                              flush=True)

            # ---------------- Validation ----------------
            elif validation_index < len(VALIDATION_TARGETS):
                target_px = to_pixel(VALIDATION_TARGETS[validation_index], frame.shape[1], frame.shape[0])
                cv2.circle(frame, tuple(target_px.astype(int)), 9, (0, 255, 0), -1)
                cv2.putText(frame,
                            f"Validation: press SPACE to start/stop "
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
                            out_path = (
                                validation_output_dir
                                / f"validation_results_{run_stamp}.csv"
                            )
                            np.savetxt(out_path, results, delimiter=",",
                                       header="x_true,y_true,x_est,y_est,error_pixels", comments="")
                            print(f"Saved {out_path}", flush=True)

            # ---------------- Jitter test ----------------
            elif not jitter_complete:
                target_px = to_pixel(JITTER_TARGET, frame.shape[1], frame.shape[0])
                cv2.circle(frame, tuple(target_px.astype(int)), 9, (0, 255, 0), -1)
                cv2.putText(frame, "Jitter: press SPACE to start/stop while fixating",
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
                        out_path = (
                            jitter_output_dir
                            / f"jitter_results_{run_stamp}.csv"
                        )
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
        grabber.stop()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()