import cv2
import mediapipe as mp
import numpy as np
import subprocess
import threading
from collections import deque
from catboost import CatBoostRegressor
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Lock camera parameters via v4l2-ctl for 120 FPS (WITHOUT hardware zoom)
dev_path = "/dev/video1"
try:
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "auto_exposure=1"], check=True)
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "exposure_dynamic_framerate=0"], check=True)
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "exposure_time_absolute=10"], check=True)
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "gain=25"], check=True)  
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "zoom_absolute=0"], check=True)

except (subprocess.CalledProcessError, FileNotFoundError) as error:
    print(f"Warning during V4L2 config: {error}")

CAM_WIDTH = 1270
CAM_HEIGHT = 720
try:
    camera_parameters = np.load("camera_params.npz")
    camera_matrix = camera_parameters["mtx"].astype(np.float64)
    distortion_coefficients = camera_parameters["dist"].astype(np.float64)
except (FileNotFoundError, KeyError) as error:
    print(f"Warning: could not load camera calibration: {error}")
    camera_matrix = None
    distortion_coefficients = None

# Capture at full sensor resolution to enable clean software cropping
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera at {dev_path}")

base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task",
    delegate=python.BaseOptions.Delegate.GPU,
)
landmarker_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)

window_name = 'Live Camera Stream'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)

calibration_targets = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.9, 0.5],
        [0.1, 0.9],
        [0.5, 0.9],
        [0.9, 0.9],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.3, 0.7],
        [0.7, 0.7],
    ],
    dtype=np.float64,
)
calibration_point_count = len(calibration_targets)
calibration_index = np.int64(0)
calibration_samples = []
calibration_burst = []
calibration_recording = False
calibration_space_armed = True
calibration_last_toggle_tick = None
gaze_models = None
iris_eye_indices = (
    np.arange(468, 473, dtype=np.int32),
    np.arange(473, 478, dtype=np.int32),
)
eye_corner_pairs = ((33, 133), (362, 263))
pose_landmark_indices = np.array([1, 33, 263, 133, 362], dtype=np.int32)
pose_object_points = np.array(
    [
        [0.0, 0.0, 0.0],
        [-30.0, 20.0, -30.0],
        [30.0, 20.0, -30.0],
        [-10.0, 15.0, -15.0],
        [10.0, 15.0, -15.0],
    ],
    dtype=np.float64,
)
validation_targets = np.array(
    [
        [0.5, 0.5],
        [0.5, 0.25],
        [0.5, 0.75],
        [0.25, 0.5],
        [0.75, 0.5],
        [0.2, 0.2],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.8, 0.8],
    ],
    dtype=np.float64,
)
validation_index = np.int64(0)
validation_results = []
jitter_target = np.array([0.5, 0.5], dtype=np.float64)
jitter_positions = []
jitter_recording = False
jitter_complete = False
jitter_last_space_tick = None
jitter_release_timeout = 0.2
MEDIAN_WINDOW_SIZE = 5
feature_history = deque(maxlen=MEDIAN_WINDOW_SIZE)


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, derivative_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.previous_time = None
        self.previous_value = None
        self.previous_derivative = None

    @staticmethod
    def smoothing_factor(cutoff, delta_time):
        time_constant = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + time_constant / delta_time)

    def reset(self):
        self.previous_time = None
        self.previous_value = None
        self.previous_derivative = None

    def __call__(self, value, timestamp):
        value = np.asarray(value, dtype=np.float64)
        timestamp = float(timestamp)
        if self.previous_value is None:
            self.previous_time = timestamp
            self.previous_value = value.copy()
            self.previous_derivative = np.zeros_like(value)
            return value.copy()

        delta_time = max(timestamp - self.previous_time, 1e-6)
        raw_derivative = (value - self.previous_value) / delta_time
        derivative_alpha = self.smoothing_factor(
            self.derivative_cutoff,
            delta_time,
        )
        filtered_derivative = (
            derivative_alpha * raw_derivative
            + (1.0 - derivative_alpha) * self.previous_derivative
        )
        cutoff = self.min_cutoff + self.beta * np.abs(filtered_derivative)
        value_alpha = self.smoothing_factor(cutoff, delta_time)
        filtered_value = (
            value_alpha * value
            + (1.0 - value_alpha) * self.previous_value
        )

        self.previous_time = timestamp
        self.previous_value = filtered_value
        self.previous_derivative = filtered_derivative
        return filtered_value.copy()


gaze_filter = OneEuroFilter()


capture_condition = threading.Condition()
capture_stop = threading.Event()
latest_frame = None
capture_sequence = 0
capture_finished = False


def capture_frames():
    global latest_frame, capture_sequence, capture_finished

    while not capture_stop.is_set():
        success, captured_frame = cap.read()
        with capture_condition:
            if not success:
                capture_finished = True
                capture_condition.notify_all()
                break
            latest_frame = captured_frame
            capture_sequence += 1
            capture_condition.notify()


capture_thread = threading.Thread(target=capture_frames, name="camera-capture")
capture_thread.start()

stream_start = np.int64(cv2.getTickCount())
previous_time = np.int64(cv2.getTickCount())
timestamp_ms = np.int64(0)
fps = np.float64(0.0)
frame_count = np.int64(0)
latency_total = np.float64(0.0)
latency_count = np.int64(0)

last_sequence = 0
while not capture_stop.is_set():
    with capture_condition:
        capture_condition.wait_for(
            lambda: capture_sequence > last_sequence or capture_finished,
            timeout=1.0,
        )
        if capture_sequence <= last_sequence:
            if capture_finished:
                print("Failed to grab frame.")
                break
            continue
        raw_frame = latest_frame.copy()
        last_sequence = capture_sequence

    frame_start = np.int64(cv2.getTickCount())
    raw_frame = cv2.flip(raw_frame, 1)


    crop_w = int(CAM_WIDTH / 4.5)  
    crop_h = int(CAM_HEIGHT / 4.5)
    x_start = (CAM_WIDTH - crop_w) // 2
    y_start = (CAM_HEIGHT - crop_h) // 2

    cropped = raw_frame[y_start:y_start+crop_h, x_start:x_start+crop_w]
    frame = cv2.resize(cropped, (960, 540), interpolation=cv2.INTER_LINEAR)
    # ---------------------------------------

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = np.int64(
        np.divide(
            np.subtract(frame_start, stream_start) * np.float64(1000.0),
            np.float64(cv2.getTickFrequency()),
        )
    )
    result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

    iris_center = None
    dual_eye_gaze = None
    filtered_dual_eye_gaze = None
    relative_gaze = None
    gaze_position = None
    pose_rotation = None
    pose_translation = None
    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]
        pose_image_points = np.array(
            [
                [
                    face_landmarks[index].x * frame.shape[1],
                    face_landmarks[index].y * frame.shape[0],
                ]
                for index in pose_landmark_indices
            ],
            dtype=np.float64,
        )
        if camera_matrix is not None:
            resized_camera_matrix = camera_matrix.copy()
            resized_camera_matrix[0, 0] *= frame.shape[1] / CAM_WIDTH
            resized_camera_matrix[1, 1] *= frame.shape[0] / CAM_HEIGHT
            resized_camera_matrix[0, 2] = (
                resized_camera_matrix[0, 2] - x_start
            ) * frame.shape[1] / CAM_WIDTH
            resized_camera_matrix[1, 2] = (
                resized_camera_matrix[1, 2] - y_start
            ) * frame.shape[0] / CAM_HEIGHT
            pnp_success, pose_rotation, pose_translation = cv2.solvePnP(
                pose_object_points,
                pose_image_points,
                resized_camera_matrix,
                distortion_coefficients,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if pnp_success:
                for axis_length in (40.0, 20.0, 10.0, 5.0):
                    axis_points = np.array(
                        [
                            [0.0, 0.0, 0.0],
                            [axis_length, 0.0, 0.0],
                            [0.0, axis_length, 0.0],
                            [0.0, 0.0, axis_length],
                        ],
                        dtype=np.float64,
                    )
                    projected_axis_points, _ = cv2.projectPoints(
                        axis_points,
                        pose_rotation,
                        pose_translation,
                        resized_camera_matrix,
                        distortion_coefficients,
                    )
                    projected_axis_points = projected_axis_points.reshape(-1, 2)
                    if np.all(
                        (projected_axis_points[:, 0] >= 0)
                        & (projected_axis_points[:, 0] < frame.shape[1])
                        & (projected_axis_points[:, 1] >= 0)
                        & (projected_axis_points[:, 1] < frame.shape[0])
                    ):
                        cv2.drawFrameAxes(
                            frame,
                            resized_camera_matrix,
                            distortion_coefficients,
                            pose_rotation,
                            pose_translation,
                            axis_length,
                            2,
                        )
                        break
        iris_indices = np.concatenate(iris_eye_indices)
        iris_pixels = np.array(
            [
                [face_landmarks[index].x * frame.shape[1],
                 face_landmarks[index].y * frame.shape[0]]
                for index in iris_indices
            ],
            dtype=np.float64,
        )
        for pixel_x, pixel_y in iris_pixels:
            cv2.circle(frame, (int(pixel_x), int(pixel_y)), 2, (0, 255, 0), -1)

        if len(iris_pixels) > 0:
            eye_iris_centers = [
                np.mean(
                    [
                        [
                            face_landmarks[index].x * frame.shape[1],
                            face_landmarks[index].y * frame.shape[0],
                        ]
                        for index in eye_indices
                    ],
                    axis=0,
                )
                for eye_indices in iris_eye_indices
            ]
            eye_centers = [
                np.mean(
                    [
                        [
                            face_landmarks[index].x * frame.shape[1],
                            face_landmarks[index].y * frame.shape[0],
                        ]
                        for index in corner_pair
                    ],
                    axis=0,
                )
                for corner_pair in eye_corner_pairs
            ]
            dual_eye_gaze = np.concatenate(
                [iris - center for iris, center in zip(eye_iris_centers, eye_centers)]
            )
            feature_history.append(dual_eye_gaze.copy())
            filtered_dual_eye_gaze = np.median(
                np.asarray(feature_history, dtype=np.float64),
                axis=0,
            )
            relative_gaze = np.mean(
                np.asarray(filtered_dual_eye_gaze, dtype=np.float64).reshape(2, 2),
                axis=0,
            )
            iris_center = np.mean(eye_iris_centers, axis=0)
            cv2.circle(
                frame,
                (int(iris_center[0]), int(iris_center[1])),
                5,
                (0, 255, 255),
                -1,
            )

            if gaze_models is not None:
                raw_gaze_position = np.array(
                    [
                        gaze_models[0].predict([filtered_dual_eye_gaze])[0],
                        gaze_models[1].predict([filtered_dual_eye_gaze])[0],
                    ],
                    dtype=np.float64,
                )
                raw_gaze_position = np.clip(
                    raw_gaze_position,
                    np.array([0.0, 0.0], dtype=np.float64),
                    np.array(
                        [frame.shape[1] - 1, frame.shape[0] - 1],
                        dtype=np.float64,
                    ),
                )
                gaze_position = gaze_filter(
                    raw_gaze_position,
                    timestamp_ms / 1000.0,
                )

    if gaze_models is None:
        target = calibration_targets[calibration_index]
        target_pixel = np.multiply(
            target,
            np.array([frame.shape[1] - 1, frame.shape[0] - 1], dtype=np.float64),
        )
        cv2.circle(
            frame,
            (int(target_pixel[0]), int(target_pixel[1])),
            9,
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            frame,
            (
                f"Look at dot and press SPACE to start "
                f"({calibration_index + 1}/{calibration_point_count})"
                if not calibration_recording
                else f"Recording burst: press SPACE to stop ({len(calibration_burst)} frames)"
            ),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    elif validation_index < len(validation_targets):
        validation_target = validation_targets[validation_index]
        validation_target_pixel = np.multiply(
            validation_target,
            np.array([frame.shape[1], frame.shape[0]], dtype=np.float64),
        )
        cv2.circle(
            frame,
            (int(validation_target_pixel[0]), int(validation_target_pixel[1])),
            9,
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame,
            f"Validation: look at dot and press SPACE ({validation_index + 1}/{len(validation_targets)})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        if gaze_position is not None:
            cv2.circle(
                frame,
                (int(gaze_position[0]), int(gaze_position[1])),
                24,
                (0, 0, 255),
                2,
            )
            cv2.circle(
                frame,
                (int(gaze_position[0]), int(gaze_position[1])),
                5,
                (0, 0, 255),
                -1,
            )
    elif not jitter_complete:
        jitter_target_pixel = np.multiply(
            jitter_target,
            np.array([frame.shape[1], frame.shape[0]], dtype=np.float64),
        )
        cv2.circle(
            frame,
            (int(jitter_target_pixel[0]), int(jitter_target_pixel[1])),
            9,
            (0, 255, 0),
            -1,
        )
        instruction = (
            "Jitter: hold SPACE while fixating, release to stop"
            if not jitter_recording
            else f"Recording jitter: {len(jitter_positions)} frames"
        )
        cv2.putText(
            frame,
            instruction,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        if gaze_position is not None:
            cv2.circle(
                frame,
                (int(gaze_position[0]), int(gaze_position[1])),
                24,
                (0, 0, 255),
                2,
            )
            cv2.circle(
                frame,
                (int(gaze_position[0]), int(gaze_position[1])),
                5,
                (0, 0, 255),
                -1,
            )
    elif gaze_position is not None:
        cv2.circle(
            frame,
            (int(gaze_position[0]), int(gaze_position[1])),
            24,
            (0, 0, 255),
            2,
        )
        cv2.circle(
            frame,
            (int(gaze_position[0]), int(gaze_position[1])),
            5,
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            frame,
            "Estimated gaze",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    frame_count = np.add(frame_count, np.int64(1))
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key in (ord('r'), ord('R')):
        calibration_index = np.int64(0)
        calibration_samples = []
        calibration_burst = []
        calibration_recording = False
        calibration_space_armed = True
        calibration_last_toggle_tick = None
        gaze_models = None
        validation_index = np.int64(0)
        validation_results = []
        jitter_positions = []
        jitter_recording = False
        jitter_complete = False
        jitter_last_space_tick = None
        feature_history.clear()
        gaze_filter.reset()
        print("Calibration reset. Starting again from point 1.", flush=True)
        continue
    validation_just_completed = False
    if gaze_models is None:
        current_tick = cv2.getTickCount()
        if key != ord(' ') and calibration_last_toggle_tick is not None:
            elapsed_since_toggle = (
                current_tick - calibration_last_toggle_tick
            ) / cv2.getTickFrequency()
            if elapsed_since_toggle > jitter_release_timeout:
                calibration_space_armed = True

        if key == ord(' ') and calibration_space_armed:
            calibration_space_armed = False
            calibration_last_toggle_tick = current_tick
            if calibration_recording:
                calibration_recording = False
            else:
                calibration_recording = True
                calibration_burst = []

        if calibration_recording and filtered_dual_eye_gaze is not None:
            target_pixel = np.multiply(
                calibration_targets[calibration_index],
                np.array([frame.shape[1] - 1, frame.shape[0] - 1], dtype=np.float64),
            )
            calibration_burst.append(
                [
                    *filtered_dual_eye_gaze,
                    target_pixel[0],
                    target_pixel[1],
                ]
            )

        if not calibration_recording and calibration_burst:
            if len(calibration_burst) >= 2:
                calibration_samples.extend(calibration_burst)
                print(
                    f"Captured calibration point "
                    f"{calibration_index + 1}/{calibration_point_count} "
                    f"with {len(calibration_burst)} frames",
                    flush=True,
                )
                calibration_index = np.add(calibration_index, np.int64(1))
                if calibration_index == calibration_point_count:
                    samples = np.array(calibration_samples, dtype=np.float64)
                    feature_matrix = samples[:, :4]
                    gaze_models = (
                        CatBoostRegressor(
                            iterations=300,
                            depth=6,
                            learning_rate=0.05,
                            loss_function="RMSE",
                            verbose=False,
                            random_seed=42,
                        ),
                        CatBoostRegressor(
                            iterations=300,
                            depth=6,
                            learning_rate=0.05,
                            loss_function="RMSE",
                            verbose=False,
                            random_seed=42,
                        ),
                    )
                    gaze_models[0].fit(feature_matrix, samples[:, 4])
                    gaze_models[1].fit(feature_matrix, samples[:, 5])
                    print(
                        f"Calibration complete with {len(samples)} burst frames. "
                        "Tracking gaze.",
                        flush=True,
                    )
                    feature_history.clear()
                    gaze_filter.reset()
                    validation_just_completed = True
            else:
                print(
                    "Calibration burst stopped before enough frames were captured.",
                    flush=True,
                )
            calibration_burst = []

    if (
        key == ord(' ')
        and gaze_models is not None
        and not validation_just_completed
        and validation_index < len(validation_targets)
    ):
        validation_target = validation_targets[validation_index]
        target_pixel = np.multiply(
            validation_target,
            np.array([frame.shape[1] - 1, frame.shape[0] - 1], dtype=np.float64),
        )
        if gaze_position is None:
            print("Could not estimate gaze; keep both eyes visible.", flush=True)
        else:
            error_pixels = float(np.linalg.norm(gaze_position - target_pixel))
            validation_results.append(
                [
                    target_pixel[0],
                    target_pixel[1],
                    gaze_position[0],
                    gaze_position[1],
                    error_pixels,
                ]
            )
            print(
                f"Validation point {validation_index + 1}/{len(validation_targets)}: "
                f"true=({target_pixel[0]:.1f}, {target_pixel[1]:.1f}), "
                f"estimated=({gaze_position[0]:.1f}, {gaze_position[1]:.1f}), "
                f"error={error_pixels:.1f}px",
                flush=True,
            )
            validation_index = np.add(validation_index, np.int64(1))
            if validation_index == len(validation_targets):
                results = np.array(validation_results, dtype=np.float64)
                print(
                    f"Validation complete: mean error={np.mean(results[:, 4]):.1f}px, "
                    f"max error={np.max(results[:, 4]):.1f}px",
                    flush=True,
                )
                np.savetxt(
                    "validation_results.csv",
                    results,
                    delimiter=",",
                    header="x_true,y_true,x_est,y_est,error_pixels",
                    comments="",
                )
                print("Saved validation results to validation_results.csv", flush=True)
                validation_just_completed = True

    if (
        key == ord(' ')
        and gaze_models is not None
        and validation_index == len(validation_targets)
        and not validation_just_completed
        and not jitter_complete
    ):
        jitter_recording = True
        jitter_last_space_tick = cv2.getTickCount()

    if jitter_recording and not jitter_complete and gaze_position is not None:
        if key == ord(' '):
            jitter_last_space_tick = cv2.getTickCount()
        elif jitter_last_space_tick is None:
            jitter_last_space_tick = cv2.getTickCount()
        elif (
            cv2.getTickCount() - jitter_last_space_tick
        ) / cv2.getTickFrequency() <= jitter_release_timeout:
            jitter_positions.append(gaze_position.copy())

    if (
        jitter_recording
        and not jitter_complete
        and jitter_last_space_tick is not None
        and (cv2.getTickCount() - jitter_last_space_tick) / cv2.getTickFrequency()
        > jitter_release_timeout
    ):
        jitter_recording = False
        if len(jitter_positions) >= 2:
            jitter_samples = np.array(jitter_positions, dtype=np.float64)
            jitter_mean = np.mean(jitter_samples, axis=0)
            jitter_deviation = jitter_samples - jitter_mean
            jitter_std = np.std(jitter_samples, axis=0)
            jitter_rms = float(
                np.sqrt(np.mean(np.sum(jitter_deviation * jitter_deviation, axis=1)))
            )
            jitter_output = np.column_stack(
                (
                    np.arange(len(jitter_samples)),
                    jitter_samples,
                    np.linalg.norm(jitter_deviation, axis=1),
                )
            )
            np.savetxt(
                "jitter_results.csv",
                jitter_output,
                delimiter=",",
                header="frame,x_est,y_est,deviation_pixels",
                comments="",
            )
            print(
                f"Jitter complete: frames={len(jitter_samples)}, "
                f"std=({jitter_std[0]:.1f}, {jitter_std[1]:.1f})px, "
                f"RMS={jitter_rms:.1f}px",
                flush=True,
            )
            print("Saved jitter results to jitter_results.csv", flush=True)
        else:
            print("Jitter test stopped before enough gaze frames were captured.", flush=True)
        jitter_complete = True

    current_time = np.int64(cv2.getTickCount())
    tick_frequency = np.float64(cv2.getTickFrequency())
    frame_duration = np.divide(
        np.subtract(current_time, frame_start), tick_frequency
    )
    latency_total = np.add(latency_total, frame_duration)
    latency_count = np.add(latency_count, np.int64(1))
    elapsed = np.divide(np.subtract(current_time, previous_time), tick_frequency)
    if elapsed >= np.float64(1.0):
        fps = np.divide(frame_count, elapsed)
        latency_ms = np.multiply(np.divide(latency_total, latency_count), np.float64(1000.0))
        print(f"FPS: {fps:.1f} | Latency: {latency_ms:.2f} ms", flush=True)
        frame_count = np.int64(0)
        latency_total = np.float64(0.0)
        latency_count = np.int64(0)
        previous_time = current_time

capture_stop.set()
with capture_condition:
    capture_condition.notify_all()
capture_thread.join(timeout=2.0)
cap.release()
landmarker.close()
cv2.destroyAllWindows()