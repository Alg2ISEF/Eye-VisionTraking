import numpy as np

class OneEuroFilter:
    """Adaptive low-pass filter for the estimated gaze position."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3,
                 derivative_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.reset()

    def reset(self):
        self.previous_time = None
        self.previous_value = None
        self.previous_derivative = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value, timestamp: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.previous_value is None:
            self.previous_time = timestamp
            self.previous_value = value.copy()
            self.previous_derivative = np.zeros_like(value)
            return value.copy()

        dt = max(timestamp - self.previous_time, 1e-6)
        raw_derivative = (value - self.previous_value) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        derivative = (
            derivative_alpha * raw_derivative
            + (1.0 - derivative_alpha) * self.previous_derivative
        )
        cutoff = self.min_cutoff + self.beta * np.abs(derivative)
        value_alpha = self._alpha(cutoff, dt)
        filtered = value_alpha * value + (1.0 - value_alpha) * self.previous_value
        self.previous_time = timestamp
        self.previous_value = filtered
        self.previous_derivative = derivative
        return filtered.copy()

