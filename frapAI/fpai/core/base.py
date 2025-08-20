"""Base classes for FPAI framework."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class FPAIModel(ABC):
    """Base class for FPAI models implementing the quantum-probabilistic pipeline.
    
    Pipeline: Data → Preprocess → Feature map Φ → Evolution Uθ → Measurement (POVM) 
             → P(y|x) → Calibration → Decision
    """
    
    def __init__(self, feature_map, ansatz, povm, optimizer=None):
        """Initialize FPAI model.
        
        Args:
            feature_map: Quantum feature map Φ: R^m → D(H)
            ansatz: Parametric quantum circuit/evolution
            povm: POVM measurement {E_y}
            optimizer: Classical optimizer for hybrid training
        """
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.povm = povm
        self.optimizer = optimizer
        self._is_trained = False
        
    @abstractmethod
    def forward_probs(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: compute P(y|x) = Tr(ρ_θ(x) E_y).
        
        Args:
            x: Input features
            
        Returns:
            Probability vector P(y|x)
        """
        pass
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.
        
        Args:
            X: Input samples (n_samples, n_features)
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
            
        probs = np.array([self.forward_probs(x) for x in X])
        return probs
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.
        
        Args:
            X: Input samples
            
        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
        
    @abstractmethod
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss function (typically NLL).
        
        Args:
            X: Input samples
            y: True labels
            
        Returns:
            Loss value
        """
        pass
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'FPAIModel':
        """Train the model using hybrid optimization.
        
        Args:
            X: Training samples
            y: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            Self (fitted model)
        """
        pass
        
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score.
        
        Args:
            X: Test samples
            y: True labels
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'feature_map': self.feature_map,
            'ansatz': self.ansatz,
            'povm': self.povm,
            'optimizer': self.optimizer
        }
        
    def set_params(self, **params) -> 'FPAIModel':
        """Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self