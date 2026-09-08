import cv2
from helper import Paths , to_pixel
from gaze_model import Session
import numpy as np



VALIDATION_TARGETS = np.array([
    [0.5, 0.5], [0.5, 0.25], [0.5, 0.75], [0.25, 0.5], [0.75, 0.5],
    [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8],
], dtype=np.float64)

JITTER_TARGET = np.array([0.5, 0.5], dtype=np.float64)


def handle_validation_phase(frame, gaze_position, session: Session,
                             space_held: bool, paths: Paths) -> None:
    target_px = to_pixel(VALIDATION_TARGETS[session.validation_index],
                          frame.shape[1], frame.shape[0])
    cv2.circle(frame, tuple(target_px.astype(int)), 9, (0, 255, 0), -1)
    cv2.putText(frame,
                f"Validation: press SPACE to start/stop "
                f"({session.validation_index + 1}/{len(VALIDATION_TARGETS)})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if gaze_position is not None:
        cv2.circle(frame, tuple(gaze_position.astype(int)), 24, (0, 0, 255), 2)

    # [FIX 12] Burst-average the *predicted position*, matching
    # the multi-frame robustness calibration already gets,
    # instead of trusting a single instantaneous keypress frame.
    sample = gaze_position.copy() if gaze_position is not None else None
    burst_done = session.validation_burst.update(space_held, sample)
    if not burst_done:
        return

    if len(session.validation_burst.samples) == 0:
        print("Could not estimate gaze during that burst; keep both eyes visible.",
              flush=True)
        return

    estimate = np.mean(np.array(session.validation_burst.samples), axis=0)
    error_px = float(np.linalg.norm(estimate - target_px))
    session.validation_results.append([*target_px, *estimate, error_px])
    print(f"Validation {session.validation_index + 1}/{len(VALIDATION_TARGETS)}: "
          f"true=({target_px[0]:.1f},{target_px[1]:.1f}) "
          f"est=({estimate[0]:.1f},{estimate[1]:.1f}) error={error_px:.1f}px",
          flush=True)
    session.validation_index += 1
    if session.validation_index == len(VALIDATION_TARGETS):
        _save_validation_results(session, paths)


def _save_validation_results(session: Session, paths: Paths) -> None:
    results = np.array(session.validation_results)
    mean_err = results[:, 4].mean()
    max_err = results[:, 4].max()
    print(f"Validation complete: mean error={mean_err:.1f}px, "
          f"max error={max_err:.1f}px", flush=True)
    out_path = paths.validation_output_dir / f"validation_results_{paths.run_stamp}.csv"
    np.savetxt(out_path, results, delimiter=",",
               header="x_true,y_true,x_est,y_est,error_pixels", comments="")
    print(f"Saved {out_path}", flush=True)


def handle_jitter_phase(frame, gaze_position, session: Session,
                         space_held: bool, paths: Paths) -> None:
    target_px = to_pixel(JITTER_TARGET, frame.shape[1], frame.shape[0])
    cv2.circle(frame, tuple(target_px.astype(int)), 9, (0, 255, 0), -1)
    cv2.putText(frame, "Jitter: press SPACE to start/stop while fixating",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if gaze_position is not None:
        cv2.circle(frame, tuple(gaze_position.astype(int)), 24, (0, 0, 255), 2)

    sample = gaze_position.copy() if gaze_position is not None else None
    burst_done = session.jitter_burst.update(space_held, sample)
    if not burst_done:
        return

    session.jitter_complete = True
    if len(session.jitter_burst.samples) < 2:
        print("Jitter burst too short; try again (press 'r' to redo from scratch, "
              "or just keep going).", flush=True)
        return

    _save_jitter_results(session, paths)


def _save_jitter_results(session: Session, paths: Paths) -> None:
    pts = np.array(session.jitter_burst.samples)
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((pts - mean) ** 2, axis=1))))
    out = np.column_stack([np.arange(len(pts)), pts, np.linalg.norm(pts - mean, axis=1)])
    out_path = paths.jitter_output_dir / f"jitter_results_{paths.run_stamp}.csv"
    np.savetxt(out_path, out, delimiter=",",
               header="frame,x_est,y_est,deviation_pixels", comments="")
    print(f"Jitter complete: frames={len(pts)}, std=({std[0]:.1f},{std[1]:.1f})px, "
          f"RMS={rms:.1f}px. Saved {out_path}", flush=True)


def handle_live_tracking(frame, gaze_position) -> None:
    if gaze_position is not None:
        cv2.circle(frame, tuple(gaze_position.astype(int)), 24, (0, 0, 255), 2)
        cv2.circle(frame, tuple(gaze_position.astype(int)), 5, (0, 0, 255), -1)
    cv2.putText(frame, "Estimated gaze", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)