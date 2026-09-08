from typing import Optional
import pickle
import numpy as np 
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from helper import rotation_matrix_to_euler
from catboost import CatBoostRegressor
from config import Config
from dataclasses import dataclass, field




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
    """Two CatBoost regressors predicting screen X and Y."""

    def __init__(self):
        self.models = (
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
        self.is_fit = False

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.models[0].fit(X, Y[:, 0])
        self.models[1].fit(X, Y[:, 1])
        self.is_fit = True

    def predict(self, feature_vector: np.ndarray) -> np.ndarray:
        features = feature_vector.reshape(1, -1)
        return np.array(
            [self.models[0].predict(features)[0], self.models[1].predict(features)[0]],
            dtype=np.float64,
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    @classmethod
    def load(cls, path: str) -> "GazeModel":
        model = cls()
        with open(path, "rb") as f:
            model.models = pickle.load(f)
        model.is_fit = True
        return model



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