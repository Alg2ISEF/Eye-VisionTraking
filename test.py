import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get full screen dimensions (Using default fallback or dummy desktop resolution)
screen_w, screen_h = 1920, 1080 

# Setup full-screen OpenCV window
window_name = 'Gaze Tracker & Calibration'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Calibration points (Normalized screen coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left)
calib_targets = [
    (int(screen_w * 0.1), int(screen_h * 0.1)),
    (int(screen_w * 0.9), int(screen_h * 0.1)),
    (int(screen_w * 0.9), int(screen_h * 0.9)),
    (int(screen_w * 0.1), int(screen_h * 0.9))
]

calib_data = [] # Stores (eye_x, eye_y, target_screen_x, target_screen_y)
current_target_idx = 0
calib_phase = True
timestamp = 0

# Mapping coefficients
coeffs_x = None
coeffs_y = None

def get_eye_coords(frame, detector):
    global timestamp
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp += 1
    result = detector.detect_for_video(mp_image, timestamp)
    
    if result.face_landmarks:
        # Landmark 468 is the center of the right iris
        iris = result.face_landmarks[0][468]
        return iris.x * w, iris.y * h
    return None, None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8) # Fullscreen black canvas
    
    eye_x, eye_y = get_eye_coords(frame, detector)

    if calib_phase:
        if current_target_idx < len(calib_targets):
            tx, ty = calib_targets[current_target_idx]
            
            # Draw calibration target circle
            cv2.circle(canvas, (tx, ty), 20, (0, 0, 255), -1)
            cv2.putText(canvas, f"Look at the red dot and press SPACE ({current_target_idx+1}/4)", 
                        (int(screen_w*0.3), int(screen_h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Press space to capture calibration point
                if eye_x is not None:
                    calib_data.append((eye_x, eye_y, tx, ty))
                    print(f"Captured point {current_target_idx+1}: Eye=({eye_x:.1f}, {eye_y:.1f}) -> Screen=({tx}, {ty})")
                    current_target_idx += 1
                else:
                    print("Eye not detected! Look at the dot.")
        else:
            # Calibration complete: Compute least squares mapping coefficients
            if len(calib_data) >= 4:
                A = np.array([[d[0], d[1], 1] for d in calib_data])
                X_screen = np.array([d[2] for d in calib_data])
                Y_screen = np.array([d[3] for d in calib_data])
                
                coeffs_x, _, _, _ = np.linalg.lstsq(A, X_screen, rcond=None)
                coeffs_y, _, _, _ = np.linalg.lstsq(A, Y_screen, rcond=None)
                calib_phase = False
                print("Calibration complete! Switching to tracking mode.")
    else:
        # Tracking Phase with Confidence Circle
        if eye_x is not None and coeffs_x is not None:
            est_x = coeffs_x[0]*eye_x + coeffs_x[1]*eye_y + coeffs_x[2]
            est_y = coeffs_y[0]*eye_x + coeffs_y[1]*eye_y + coeffs_y[2]
            
            # Draw confidence circle (Simulated radius reflecting variance/stability)
            cv2.circle(canvas, (int(est_x), int(est_y)), 35, (255, 0, 0), 2)
            cv2.circle(canvas, (int(est_x), int(est_y)), 5, (0, 255, 0), -1)
            cv2.putText(canvas, f"Estimated Gaze: ({int(est_x)}, {int(est_y)})", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(canvas, "Searching for eyes...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(canvas, "Press 'q' to quit", (50, screen_h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow(window_name, canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()