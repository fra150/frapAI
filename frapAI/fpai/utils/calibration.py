"""Calibration methods for probabilistic predictions.

Implements various post-hoc calibration techniques to improve
the reliability of quantum machine learning model predictions.
"""

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar


class Calibrator(ABC):
    """Abstract base class for calibration methods."""
    
    def __init__(self):
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> 'Calibrator':
        """Fit calibrator on validation data.
        
        Args:
            probs: Predicted probabilities
            y_true: True labels
            
        Returns:
            Self (fitted calibrator)
        """
        pass
        
    @abstractmethod
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities.
        
        Args:
            probs: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        pass
        
    def fit_calibrate(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and apply calibration in one step.
        
        Args:
            probs: Predicted probabilities
            y_true: True labels
            
        Returns:
            Calibrated probabilities
        """
        self.fit(probs, y_true)
        return self.calibrate(probs)


class TemperatureScaling(Calibrator):
    """Temperature scaling calibration.
    
    Applies a single temperature parameter T to all predictions:
    P_calibrated = softmax(logits / T)
    
    For binary classification: P_calibrated = sigmoid(logit / T)
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        """Initialize temperature scaling.
        
        Args:
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.temperature = 1.0
        
    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> 'TemperatureScaling':
        """Fit temperature parameter.
        
        Args:
            probs: Predicted probabilities [n_samples, n_classes]
            y_true: True labels [n_samples]
            
        Returns:
            Self (fitted calibrator)
        """
        # Convert probabilities to logits
        logits = self._probs_to_logits(probs)
        
        # Optimize temperature to minimize NLL
        def nll_loss(temperature):
            if temperature <= 0:
                return np.inf
            calibrated_logits = logits / temperature
            calibrated_probs = self._logits_to_probs(calibrated_logits)
            return self._negative_log_likelihood(calibrated_probs, y_true)
            
        # Find optimal temperature
        result = minimize_scalar(
            nll_loss, 
            bounds=(0.01, 10.0), 
            method='bounded',
            options={'maxiter': self.max_iter, 'xatol': self.tol}
        )
        
        self.temperature = result.x
        self.is_fitted = True
        
        return self
        
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling.
        
        Args:
            probs: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
            
        # Convert to logits, scale, convert back
        logits = self._probs_to_logits(probs)
        scaled_logits = logits / self.temperature
        calibrated_probs = self._logits_to_probs(scaled_logits)
        
        return calibrated_probs
        
    def _probs_to_logits(self, probs: np.ndarray) -> np.ndarray:
        """Convert probabilities to logits.
        
        Args:
            probs: Probability array
            
        Returns:
            Logit array
        """
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        
        if probs.shape[1] == 2:  # Binary classification
            # Use log-odds
            return np.log(probs[:, 1] / probs[:, 0]).reshape(-1, 1)
        else:  # Multi-class
            # Use log probabilities (subtract max for numerical stability)
            log_probs = np.log(probs)
            return log_probs - np.max(log_probs, axis=1, keepdims=True)
            
    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities.
        
        Args:
            logits: Logit array
            
        Returns:
            Probability array
        """
        if logits.shape[1] == 1:  # Binary classification
            # Sigmoid
            sigmoid = 1 / (1 + np.exp(-logits.flatten()))
            return np.column_stack([1 - sigmoid, sigmoid])
        else:  # Multi-class
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
    def _negative_log_likelihood(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Compute negative log-likelihood.
        
        Args:
            probs: Predicted probabilities
            y_true: True labels
            
        Returns:
            NLL loss
        """
        # Extract probabilities for true classes
        true_probs = probs[np.arange(len(y_true)), y_true]
        true_probs = np.clip(true_probs, 1e-12, 1.0)  # Avoid log(0)
        
        return -np.mean(np.log(true_probs))
        
    def get_temperature(self) -> float:
        """Get fitted temperature parameter.
        
        Returns:
            Temperature value
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted first")
        return self.temperature


class PlattScaling(Calibrator):
    """Platt scaling calibration.
    
    Fits a sigmoid function to the predictions:
    P_calibrated = sigmoid(A * score + B)
    
    Where score is the decision function output.
    """
    
    def __init__(self, max_iter: int = 1000):
        """Initialize Platt scaling.
        
        Args:
            max_iter: Maximum optimization iterations
        """
        super().__init__()
        self.max_iter = max_iter
        self.lr = LogisticRegression(max_iter=max_iter)
        
    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> 'PlattScaling':
        """Fit Platt scaling parameters.
        
        Args:
            probs: Predicted probabilities [n_samples, n_classes]
            y_true: True labels [n_samples]
            
        Returns:
            Self (fitted calibrator)
        """
        if probs.shape[1] != 2:
            raise ValueError("Platt scaling only supports binary classification")
            
        # Use log-odds as features
        log_odds = np.log(probs[:, 1] / probs[:, 0]).reshape(-1, 1)
        
        # Fit logistic regression
        self.lr.fit(log_odds, y_true)
        self.is_fitted = True
        
        return self
        
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling.
        
        Args:
            probs: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
            
        if probs.shape[1] != 2:
            raise ValueError("Platt scaling only supports binary classification")
            
        # Convert to log-odds
        log_odds = np.log(probs[:, 1] / probs[:, 0]).reshape(-1, 1)
        
        # Apply calibration
        calibrated_prob_pos = self.lr.predict_proba(log_odds)[:, 1]
        calibrated_probs = np.column_stack([1 - calibrated_prob_pos, calibrated_prob_pos])
        
        return calibrated_probs


class IsotonicCalibration(Calibrator):
    """Isotonic regression calibration.
    
    Fits a monotonic function to map predictions to calibrated probabilities.
    Non-parametric method that can capture complex calibration curves.
    """
    
    def __init__(self, out_of_bounds: str = 'clip'):
        """Initialize isotonic calibration.
        
        Args:
            out_of_bounds: How to handle out-of-bounds predictions ('clip' or 'nan')
        """
        super().__init__()
        self.out_of_bounds = out_of_bounds
        self.calibrators = []
        
    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibration':
        """Fit isotonic regression.
        
        Args:
            probs: Predicted probabilities [n_samples, n_classes]
            y_true: True labels [n_samples]
            
        Returns:
            Self (fitted calibrator)
        """
        n_classes = probs.shape[1]
        self.calibrators = []
        
        if n_classes == 2:
            # Binary classification: fit one calibrator
            calibrator = IsotonicRegression(out_of_bounds=self.out_of_bounds)
            calibrator.fit(probs[:, 1], y_true)
            self.calibrators.append(calibrator)
        else:
            # Multi-class: fit one-vs-rest calibrators
            for class_idx in range(n_classes):
                y_binary = (y_true == class_idx).astype(int)
                calibrator = IsotonicRegression(out_of_bounds=self.out_of_bounds)
                calibrator.fit(probs[:, class_idx], y_binary)
                self.calibrators.append(calibrator)
                
        self.is_fitted = True
        return self
        
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration.
        
        Args:
            probs: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
            
        n_classes = probs.shape[1]
        calibrated_probs = np.zeros_like(probs)
        
        if n_classes == 2:
            # Binary classification
            calibrated_prob_pos = self.calibrators[0].predict(probs[:, 1])
            calibrated_probs[:, 0] = 1 - calibrated_prob_pos
            calibrated_probs[:, 1] = calibrated_prob_pos
        else:
            # Multi-class: calibrate each class separately then normalize
            for class_idx in range(n_classes):
                calibrated_probs[:, class_idx] = self.calibrators[class_idx].predict(probs[:, class_idx])
                
            # Normalize to ensure probabilities sum to 1
            row_sums = np.sum(calibrated_probs, axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
            calibrated_probs = calibrated_probs / row_sums
            
        return calibrated_probs


class VectorScaling(Calibrator):
    """Vector scaling calibration.
    
    Extends temperature scaling to use a different temperature
    for each class: P_calibrated = softmax(logits / T_vector)
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        """Initialize vector scaling.
        
        Args:
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.temperatures = None
        
    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> 'VectorScaling':
        """Fit temperature vector.
        
        Args:
            probs: Predicted probabilities [n_samples, n_classes]
            y_true: True labels [n_samples]
            
        Returns:
            Self (fitted calibrator)
        """
        n_classes = probs.shape[1]
        logits = self._probs_to_logits(probs)
        
        # Optimize temperature vector
        def nll_loss(temperatures):
            if np.any(temperatures <= 0):
                return np.inf
            calibrated_logits = logits / temperatures.reshape(1, -1)
            calibrated_probs = self._logits_to_probs(calibrated_logits)
            return self._negative_log_likelihood(calibrated_probs, y_true)
            
        # Initialize temperatures
        initial_temps = np.ones(n_classes)
        
        # Optimize using scipy
        from scipy.optimize import minimize
        result = minimize(
            nll_loss,
            initial_temps,
            method='L-BFGS-B',
            bounds=[(0.01, 10.0)] * n_classes,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        self.temperatures = result.x
        self.is_fitted = True
        
        return self
        
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply vector scaling.
        
        Args:
            probs: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
            
        # Convert to logits, scale, convert back
        logits = self._probs_to_logits(probs)
        scaled_logits = logits / self.temperatures.reshape(1, -1)
        calibrated_probs = self._logits_to_probs(scaled_logits)
        
        return calibrated_probs
        
    def _probs_to_logits(self, probs: np.ndarray) -> np.ndarray:
        """Convert probabilities to logits."""
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        log_probs = np.log(probs)
        return log_probs - np.max(log_probs, axis=1, keepdims=True)
        
    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
    def _negative_log_likelihood(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Compute negative log-likelihood."""
        true_probs = probs[np.arange(len(y_true)), y_true]
        true_probs = np.clip(true_probs, 1e-12, 1.0)
        return -np.mean(np.log(true_probs))
        
    def get_temperatures(self) -> np.ndarray:
        """Get fitted temperature vector.
        
        Returns:
            Temperature array
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted first")
        return self.temperatures.copy()
