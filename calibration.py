import cv2
import numpy as np
import subprocess

dev_path = "/dev/video1"
try:
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "auto_exposure=1"], check=True)
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "exposure_dynamic_framerate=0"], check=True)
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "exposure_time_absolute=10"], check=True)
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "gain=25"], check=True)
    subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "zoom_absolute=0"], check=True)
except (subprocess.CalledProcessError, FileNotFoundError) as error:
    print(f"Warning during V4L2 config: {error}")
# 1. Define board parameters matching your generated PDF
squares_x = 5
squares_y = 7
square_length = 0.04  # 40mm
marker_length = 0.03  # 30mm

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
params = cv2.aruco.DetectorParameters()
aruco_detector = None
charuco_detector = None

if hasattr(cv2.aruco, "ArucoDetector"):
    aruco_detector = cv2.aruco.ArucoDetector(dictionary, params)
if hasattr(cv2.aruco, "CharucoDetector"):
    charuco_detector = cv2.aruco.CharucoDetector(board)

# 2. Open Camera stream (/dev/video1)
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 120)

all_charuco_corners = []
all_charuco_ids = []
image_size = None

print("=== ChArUco Calibration Pipeline ===")
print("Instructions:")
print(" - Hold the printed board in front of the camera at different angles, tilts, and distances.")
print(" - Press 'c' to capture a frame when the grid turns green (aim for 15-20 captures).")
print(" - Press 'q' when finished to compute the camera matrix.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]

    display_frame = frame.copy()
    board_detected = False

    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            charuco_detector.detectBoard(gray)
        )
    elif aruco_detector is not None:
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if charuco_retval <= 0:
            charuco_corners, charuco_ids = None, None
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=params
        )
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if charuco_retval <= 0:
            charuco_corners, charuco_ids = None, None

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(display_frame, marker_corners, marker_ids)

    if charuco_corners is not None and charuco_ids is not None:
        normalized_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
        normalized_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
        if len(normalized_corners) == len(normalized_ids) and len(normalized_ids) > 4:
            charuco_corners = normalized_corners.reshape(-1, 1, 2)
            charuco_ids = normalized_ids
            board_detected = True

    if board_detected:
        cv2.aruco.drawDetectedCornersCharuco(
            display_frame, charuco_corners, charuco_ids, (0, 255, 0)
        )

    # Display HUD text overlay
    status_color = (0, 255, 0) if len(all_charuco_corners) >= 15 else (0, 0, 255)
    cv2.putText(display_frame, f"Captured Frames: {len(all_charuco_corners)} / 15", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(display_frame, "Press 'c' to capture | 'q' to compute", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("ChArUco Calibration", display_frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c') and board_detected:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        print(f"Captured frame successfully! Total: {len(all_charuco_corners)}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 3. Perform Calibration Calculation
if len(all_charuco_corners) >= 5:
    print("\nComputing camera matrix and distortion coefficients...")
    board_corners_3d = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    object_points = [
        np.take(board_corners_3d, ids.reshape(-1), axis=0).reshape(-1, 1, 3)
        for ids in all_charuco_ids
    ]
    image_points = [
        np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        for corners in all_charuco_corners
    ]
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    
    print("\n--- Calibration Results ---")
    print(f"Reprojection Error: {retval:.4f} pixels")
    print("Camera Matrix ($K$):\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    
    # Save parameters for your gaze pipeline
    np.savez("camera_params.npz", mtx=camera_matrix, dist=dist_coeffs)
    print("\nSuccessfully saved calibration data to 'camera_params.npz'!")
else:
    print("\nNot enough frames captured. Please run again and capture at least 10-15 frames.")