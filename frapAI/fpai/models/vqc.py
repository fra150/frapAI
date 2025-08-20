"""Variational Quantum Classifier (VQC) implementation."""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from ..core.base import FPAIModel
from ..core.quantum_state import QuantumState
from ..core.feature_map import FeatureMap
from ..core.povm import POVM
from .ansatz import Ansatz, HardwareEfficientAnsatz


class VQC(FPAIModel):
    """Variational Quantum Classifier.
    Implements the full FPAI pipeline:
    x → Φ(x) → U_θ(Φ(x)) → POVM → P(y|x)
    Uses hybrid classical-quantum optimization with parameter-shift rule.
    """
    
    def __init__(self, feature_map: FeatureMap, ansatz: Ansatz, 
                 povm: POVM, optimizer: Optional[Any] = None,
                 shots: int = 1024, backend: str = 'statevector'):
        """Initialize VQC.
        
        Args:
            feature_map: Quantum feature map Φ
            ansatz: Parametric quantum circuit
            povm: POVM measurement
            optimizer: Classical optimizer
            shots: Number of measurement shots
            backend: Quantum backend ('statevector', 'qasm')
        """
        super().__init__(feature_map, ansatz, povm, optimizer)
        self.shots = shots
        self.backend = backend
        self.training_history = []
        
        # Initialize random parameters
        self.ansatz.initialize_params()
        
    def forward_probs(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: compute P(y|x) = Tr(ρ_θ(x) E_y).
        
        Args:
            x: Input features
            
        Returns:
            Probability vector P(y|x)
        """
        # Embed classical data
        rho = self.feature_map.embed(x)
        
        # Apply parametric evolution
        rho_evolved = self.ansatz.evolve(rho)
        
        # Measure with POVM
        if self.backend == 'statevector':
            probs = self.povm.measure_probs(rho_evolved)
        else:
            # Simulate shot noise
            probs = self._measure_with_shots(rho_evolved)
            
        return probs
        
    def _measure_with_shots(self, state: QuantumState) -> np.ndarray:
        """Simulate measurement with finite shots.
        
        Args:
            state: Quantum state to measure
            
        Returns:
            Empirical probability distribution
        """
        true_probs = self.povm.measure_probs(state)
        outcomes = self.povm.sample_outcome(state, self.shots)
        
        # Count outcomes
        counts = np.bincount(outcomes, minlength=self.povm.n_outcomes)
        empirical_probs = counts / self.shots
        
        return empirical_probs
        
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute negative log-likelihood loss.
        
        Args:
            X: Input samples
            y: True labels
            
        Returns:
            NLL loss
        """
        total_loss = 0.0
        
        for i, (x_i, y_i) in enumerate(zip(X, y)):
            probs = self.forward_probs(x_i)
            # Avoid log(0)
            prob_y = max(probs[y_i], 1e-12)
            total_loss -= np.log(prob_y)
            
        return total_loss / len(X)
        
    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient using parameter-shift rule.
        
        Args:
            X: Input samples
            y: True labels
            
        Returns:
            Gradient vector
        """
        n_params = len(self.ansatz.params)
        grad = np.zeros(n_params)
        
        for param_idx in range(n_params):
            # Parameter-shift rule: ∂⟨O⟩/∂θ = (⟨O⟩_{θ+π/2} - ⟨O⟩_{θ-π/2})/2
            shift = np.pi / 2
            
            # Forward shift
            self.ansatz.params[param_idx] += shift
            loss_plus = self.loss(X, y)
            
            # Backward shift
            self.ansatz.params[param_idx] -= 2 * shift
            loss_minus = self.loss(X, y)
            
            # Restore parameter
            self.ansatz.params[param_idx] += shift
            
            # Gradient
            grad[param_idx] = (loss_plus - loss_minus) / 2
            
        return grad
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, learning_rate: float = 0.01,
            batch_size: Optional[int] = None, 
            validation_split: float = 0.0,
            early_stopping: bool = False,
            patience: int = 10,
            verbose: bool = True) -> 'VQC':
        """Train VQC using hybrid optimization.
        
        Args:
            X: Training samples
            y: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size (None for full batch)
            validation_split: Fraction for validation
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Self (fitted model)
        """
        # Validation split
        if validation_split > 0:
            n_val = int(len(X) * validation_split)
            indices = np.random.permutation(len(X))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Batch training
            if batch_size is None:
                batch_size = len(X_train)
                
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Compute gradient
                grad = self.gradient(batch_X, batch_y)
                
                # Update parameters
                self.ansatz.params -= learning_rate * grad
                
                # Track loss
                batch_loss = self.loss(batch_X, batch_y)
                epoch_loss += batch_loss
                n_batches += 1
                
            epoch_loss /= n_batches
            
            # Validation
            val_loss = None
            if X_val is not None:
                val_loss = self.loss(X_val, y_val)
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
                        
            # Log progress
            self.training_history.append({
                'epoch': epoch,
                'train_loss': epoch_loss,
                'val_loss': val_loss
            })
            
            if verbose and epoch % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}")
                    
        self._is_trained = True
        return self
        
    def predict_proba_with_uncertainty(self, X: np.ndarray, 
                                      n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with epistemic uncertainty estimation.
        
        Uses parameter sampling to estimate uncertainty.
        
        Args:
            X: Input samples
            n_samples: Number of parameter samples
            
        Returns:
            (mean_probs, std_probs): Mean and standard deviation of predictions
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Store original parameters
        original_params = self.ansatz.params.copy()
        
        # Sample predictions
        all_probs = []
        param_std = 0.01  # Small noise for parameter sampling
        
        for _ in range(n_samples):
            # Add noise to parameters
            noisy_params = original_params + np.random.normal(0, param_std, len(original_params))
            self.ansatz.params = noisy_params
            
            # Predict
            probs = self.predict_proba(X)
            all_probs.append(probs)
            
        # Restore original parameters
        self.ansatz.params = original_params
        
        # Compute statistics
        all_probs = np.array(all_probs)
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        
        return mean_probs, std_probs
        
    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute feature importance via sensitivity analysis.
        
        Args:
            X: Input samples
            y: True labels
            
        Returns:
            Feature importance scores
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before feature importance")
            
        importance = np.zeros(X.shape[1])
        baseline_loss = self.loss(X, y)
        
        for feature_idx in range(X.shape[1]):
            # Permute feature
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
            
            # Compute loss increase
            permuted_loss = self.loss(X_permuted, y)
            importance[feature_idx] = permuted_loss - baseline_loss
            
        return importance
        
    def save_model(self, filepath: str):
        """Save model parameters and configuration.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'params': self.ansatz.params,
            'feature_map_config': self.feature_map.__dict__,
            'ansatz_config': self.ansatz.__dict__,
            'povm_config': self.povm.__dict__,
            'training_history': self.training_history,
            'is_trained': self._is_trained
        }
        
        np.savez(filepath, **model_data)
        
    def load_model(self, filepath: str):
        """Load model parameters and configuration.
        
        Args:
            filepath: Path to load model from
        """
        data = np.load(filepath, allow_pickle=True)
        
        self.ansatz.params = data['params']
        self.training_history = data['training_history'].tolist()
        self._is_trained = bool(data['is_trained'])
        
    def __repr__(self) -> str:
        return f"VQC(n_qubits={self.feature_map.n_qubits}, n_params={len(self.ansatz.params)}, trained={self._is_trained})"