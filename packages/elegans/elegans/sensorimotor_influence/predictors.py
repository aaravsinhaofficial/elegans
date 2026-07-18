"""Small online Gaussian predictors for sensorimotor-influence estimation.

The classes in this module are deliberately independent of the nematode and
reinforcement-learning stacks.  A prediction is always scored before the
corresponding transition is used for learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import numpy as np
from numpy.typing import NDArray

from .config import LearnerConfig, VarianceMode

FloatArray = NDArray[np.float64]
LOG_TWO_PI = log(2.0 * pi)


@dataclass(frozen=True, slots=True)
class GaussianPrediction:
    """A pre-update probabilistic prediction for one scalar observation."""

    mean: float
    variance: float
    error: float
    squared_error: float
    negative_log_likelihood: float


class OnlineLinearGaussian:
    """Linear mean with an online scalar residual-variance estimate.

    The mean update is normalized least mean squares.  Normalization changes
    only the numerical step size; the update remains an online delta rule using
    the current local features and residual.
    """

    def __init__(self, n_features: int, config: LearnerConfig) -> None:
        if n_features < 1:
            message = "n_features must be positive"
            raise ValueError(message)
        self.config = config
        self.weights: FloatArray = np.zeros(n_features, dtype=np.float64)
        self.variance = float(
            config.fixed_variance
            if config.variance_mode is VarianceMode.FIXED
            else config.initial_variance,
        )

    def score(self, features: FloatArray, target: float) -> GaussianPrediction:
        """Score ``target`` without changing any learner state."""
        self._validate_features(features)
        mean = float(self.weights @ features)
        error = float(target - mean)
        squared_error = error * error
        variance = float(
            self.config.fixed_variance
            if self.config.variance_mode is VarianceMode.FIXED
            else self.variance,
        )
        negative_log_likelihood = 0.5 * (LOG_TWO_PI + log(variance) + squared_error / variance)
        return GaussianPrediction(
            mean=mean,
            variance=variance,
            error=error,
            squared_error=squared_error,
            negative_log_likelihood=negative_log_likelihood,
        )

    def update(self, features: FloatArray, prediction: GaussianPrediction) -> None:
        """Update from a previously computed, pre-update prediction."""
        self._validate_features(features)
        scale = max(float(features @ features), 1.0)
        self.weights += self.config.mean_learning_rate * prediction.error * features / scale
        if self.config.variance_mode is VarianceMode.ADAPTIVE:
            rate = self.config.variance_learning_rate
            next_variance = (1.0 - rate) * self.variance + rate * prediction.squared_error
            self.variance = float(
                np.clip(
                    next_variance,
                    self.config.min_variance,
                    self.config.max_variance,
                ),
            )

    def step(self, features: FloatArray, target: float) -> GaussianPrediction:
        """Score first, then learn from the current target."""
        prediction = self.score(features, target)
        self.update(features, prediction)
        return prediction

    def _validate_features(self, features: FloatArray) -> None:
        if features.shape != self.weights.shape:
            message = f"expected feature shape {self.weights.shape}, got {features.shape}"
            raise ValueError(message)
        if not np.all(np.isfinite(features)):
            message = "features must be finite"
            raise ValueError(message)


def predictor_features(state: float, auxiliary_input: float) -> FloatArray:
    """Return the common, equal-width feature vector used by every predictor."""
    return np.asarray([state, 1.0, auxiliary_input], dtype=np.float64)
