import cv2
import mediapipe as mp
import numpy as np
import subprocess
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

CAM_WIDTH = 960
CAM_HEIGHT = 540
# Capture at full sensor resolution to enable clean software cropping
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera at {dev_path}")

base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
landmarker_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)

window_name = 'Live Camera Stream (Software Zoom)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

calibration_targets = np.array(
    [
        [0.1, 0.1],
        [0.5, 0.1],
        [0.9, 0.1],
        [0.1, 0.5],
        [0.5, 0.5],
        [0.9, 0.5],
        [0.9, 0.1],
        [0.5, 0.9],
        [0.9, 0.9],
        [0.1, 0.9],
    ],
    dtype=np.float64,
)
calibration_index = np.int64(0)
calibration_samples = []
gaze_coefficients = None

stream_start = np.int64(cv2.getTickCount())
previous_time = np.int64(cv2.getTickCount())
timestamp_ms = np.int64(0)
fps = np.float64(0.0)
frame_count = np.int64(0)
latency_total = np.float64(0.0)
latency_count = np.int64(0)

while cap.isOpened():
    success, raw_frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

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
    gaze_position = None
    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]
        iris_indices = np.arange(468, min(478, len(face_landmarks)))
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
            iris_center = np.mean(iris_pixels, axis=0)
            cv2.circle(
                frame,
                (int(iris_center[0]), int(iris_center[1])),
                5,
                (0, 255, 255),
                -1,
            )

            if gaze_coefficients is not None:
                gaze_input = np.array(
                    [iris_center[0], iris_center[1], 1.0],
                    dtype=np.float64,
                )
                gaze_position = np.array(
                    [
                        np.dot(gaze_input, gaze_coefficients[:, 0]),
                        np.dot(gaze_input, gaze_coefficients[:, 1]),
                    ],
                    dtype=np.float64,
                )
                gaze_position = np.clip(
                    gaze_position,
                    np.array([0.0, 0.0], dtype=np.float64),
                    np.array(
                        [frame.shape[1] - 1, frame.shape[0] - 1],
                        dtype=np.float64,
                    ),
                )

    if gaze_coefficients is None:
        target = calibration_targets[calibration_index]
        target_pixel = np.multiply(
            target,
            np.array([frame.shape[1], frame.shape[0]], dtype=np.float64),
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
            f"Look at the dot and press SPACE ({calibration_index + 1}/9)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
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
    if key == ord(' ') and gaze_coefficients is None:
        if iris_center is None:
            print("Could not capture iris position; keep both eyes visible.")
        else:
            target_pixel = np.multiply(
                calibration_targets[calibration_index],
                np.array([frame.shape[1], frame.shape[0]], dtype=np.float64),
            )
            calibration_samples.append(
                [iris_center[0], iris_center[1], target_pixel[0], target_pixel[1]]
            )
            print(
                f"Captured calibration point {calibration_index + 1}/4: "
                f"iris=({iris_center[0]:.1f}, {iris_center[1]:.1f})",
                flush=True,
            )
            calibration_index = np.add(calibration_index, np.int64(1))
            if calibration_index == np.int64(9):
                samples = np.array(calibration_samples, dtype=np.float64)
                design_matrix = np.column_stack(
                    (samples[:, 0], samples[:, 1], np.ones(len(samples)))
                )
                gaze_coefficients = np.linalg.lstsq(
                    design_matrix,
                    samples[:, 2:4],
                    rcond=None,
                )[0]
                print("Calibration complete. Tracking gaze.", flush=True)

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

cap.release()
landmarker.close()
cv2.destroyAllWindows()