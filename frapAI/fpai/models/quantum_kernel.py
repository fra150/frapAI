"""Quantum Kernel methods for FPAI framework."""

import numpy as np
from typing import Optional, Callable, Union
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from ..core.base import FPAIModel
from ..core.quantum_state import QuantumState
from ..core.feature_map import FeatureMap
from ..core.povm import POVM


class QuantumKernel(FPAIModel):
    """Quantum Kernel Classifier.
    
    Uses quantum feature maps to compute kernel matrix K(x,x') = |⟨Φ(x)|Φ(x')⟩|²
    and trains classical SVM on the quantum kernel.
    """
    
    def __init__(self, feature_map: FeatureMap, 
                 kernel_type: str = 'fidelity',
                 svm_params: Optional[dict] = None,
                 shots: int = 1024,
                 backend: str = 'statevector'):
        """Initialize Quantum Kernel.
        
        Args:
            feature_map: Quantum feature map Φ
            kernel_type: Type of quantum kernel ('fidelity', 'projected')
            svm_params: Parameters for classical SVM
            shots: Number of measurement shots
            backend: Quantum backend
        """
        # Create dummy ansatz and POVM for base class
        from .ansatz import HardwareEfficientAnsatz
        dummy_ansatz = HardwareEfficientAnsatz(feature_map.n_qubits, 1)
        dummy_povm = POVM.projective_measurement(feature_map.n_qubits)
        
        super().__init__(feature_map, dummy_ansatz, dummy_povm)
        
        self.kernel_type = kernel_type
        self.shots = shots
        self.backend = backend
        
        # Initialize SVM
        if svm_params is None:
            svm_params = {'C': 1.0, 'kernel': 'precomputed'}
        else:
            svm_params['kernel'] = 'precomputed'  # Force precomputed kernel
            
        self.svm = SVC(**svm_params)
        self.X_train = None
        self.kernel_matrix_train = None
        
    def quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel between two samples.
        
        Args:
            x1: First sample
            x2: Second sample
            
        Returns:
            Kernel value K(x1, x2)
        """
        # Embed samples
        state1 = self.feature_map.embed(x1)
        state2 = self.feature_map.embed(x2)
        
        if self.kernel_type == 'fidelity':
            # Fidelity kernel: K(x,x') = |⟨Φ(x)|Φ(x')⟩|²
            return state1.fidelity(state2)
        elif self.kernel_type == 'projected':
            # Projected kernel with POVM measurement
            # K(x,x') = ∑ᵧ √P(y|x)P(y|x')
            probs1 = self.povm.measure_probs(state1)
            probs2 = self.povm.measure_probs(state2)
            return np.sum(np.sqrt(probs1 * probs2))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
    def compute_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix between sets of samples.
        
        Args:
            X1: First set of samples
            X2: Second set of samples (if None, use X1)
            
        Returns:
            Kernel matrix K[i,j] = K(X1[i], X2[j])
        """
        if X2 is None:
            X2 = X1
            
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                if self.backend == 'statevector':
                    K[i, j] = self.quantum_kernel(X1[i], X2[j])
                else:
                    # Simulate shot noise
                    K[i, j] = self._quantum_kernel_with_shots(X1[i], X2[j])
                    
        return K
        
    def _quantum_kernel_with_shots(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel with finite shots.
        
        Args:
            x1: First sample
            x2: Second sample
            
        Returns:
            Empirical kernel value
        """
        if self.kernel_type == 'fidelity':
            # Use swap test to estimate fidelity
            return self._swap_test_fidelity(x1, x2)
        else:
            # Use measurement statistics
            state1 = self.feature_map.embed(x1)
            state2 = self.feature_map.embed(x2)
            
            # Sample measurements
            outcomes1 = self.povm.sample_outcome(state1, self.shots)
            outcomes2 = self.povm.sample_outcome(state2, self.shots)
            
            # Compute empirical probabilities
            probs1 = np.bincount(outcomes1, minlength=self.povm.n_outcomes) / self.shots
            probs2 = np.bincount(outcomes2, minlength=self.povm.n_outcomes) / self.shots
            
            return np.sum(np.sqrt(probs1 * probs2))
            
    def _swap_test_fidelity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Estimate fidelity using swap test.
        
        Args:
            x1: First sample
            x2: Second sample
            
        Returns:
            Estimated fidelity
        """
        # Simplified swap test simulation
        # In practice, this would require quantum hardware
        
        # For now, compute exact fidelity and add noise
        state1 = self.feature_map.embed(x1)
        state2 = self.feature_map.embed(x2)
        true_fidelity = state1.fidelity(state2)
        
        # Add shot noise
        # Swap test gives P(0) = (1 + F)/2, so F = 2*P(0) - 1
        p_zero_true = (1 + true_fidelity) / 2
        
        # Sample from binomial
        successes = np.random.binomial(self.shots, p_zero_true)
        p_zero_empirical = successes / self.shots
        
        fidelity_empirical = 2 * p_zero_empirical - 1
        return max(0, fidelity_empirical)  # Ensure non-negative
        
    def forward_probs(self, x: np.ndarray) -> np.ndarray:
        """Compute class probabilities using SVM decision function.
        
        Args:
            x: Input sample
            
        Returns:
            Class probabilities
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Compute kernel with training data
        K_test = self.compute_kernel_matrix(np.array([x]), self.X_train)
        
        # Get SVM decision function
        decision = self.svm.decision_function(K_test)
        
        # Convert to probabilities using sigmoid
        if len(decision.shape) == 1:  # Binary classification
            prob_pos = 1 / (1 + np.exp(-decision[0]))
            return np.array([1 - prob_pos, prob_pos])
        else:  # Multi-class
            # Use softmax
            exp_scores = np.exp(decision[0] - np.max(decision[0]))
            return exp_scores / np.sum(exp_scores)
            
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute hinge loss (SVM loss).
        
        Args:
            X: Input samples
            y: True labels
            
        Returns:
            Hinge loss
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before computing loss")
            
        # Compute kernel matrix
        K = self.compute_kernel_matrix(X, self.X_train)
        
        # Get SVM decision values
        decision_values = self.svm.decision_function(K)
        
        # Compute hinge loss
        if len(np.unique(y)) == 2:  # Binary
            y_signed = 2 * y - 1  # Convert {0,1} to {-1,1}
            margins = y_signed * decision_values
            hinge_losses = np.maximum(0, 1 - margins)
        else:  # Multi-class
            # One-vs-rest hinge loss
            n_classes = len(np.unique(y))
            hinge_losses = []
            
            for i, class_label in enumerate(np.unique(y)):
                y_binary = (y == class_label).astype(int)
                y_signed = 2 * y_binary - 1
                margins = y_signed * decision_values[:, i]
                hinge_losses.extend(np.maximum(0, 1 - margins))
                
        return np.mean(hinge_losses)
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'QuantumKernel':
        """Train quantum kernel classifier.
        
        Args:
            X: Training samples
            y: Training labels
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Self (fitted model)
        """
        self.X_train = X.copy()
        
        # Compute kernel matrix
        print("Computing quantum kernel matrix...")
        self.kernel_matrix_train = self.compute_kernel_matrix(X)
        
        # Train SVM
        print("Training SVM...")
        self.svm.fit(self.kernel_matrix_train, y)
        
        self._is_trained = True
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input samples
            
        Returns:
            Class probabilities
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
            
        probs = []
        for x in X:
            prob = self.forward_probs(x)
            probs.append(prob)
            
        return np.array(probs)
        
    def kernel_alignment(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute kernel-target alignment.
        
        Measures how well the quantum kernel aligns with the target labels.
        
        Args:
            X: Input samples
            y: Target labels
            
        Returns:
            Kernel alignment score
        """
        # Compute quantum kernel matrix
        K = self.compute_kernel_matrix(X)
        
        # Compute ideal kernel (target kernel)
        Y = np.outer(y, y)  # Y[i,j] = y[i] * y[j]
        
        # Normalize kernels
        K_norm = K / np.sqrt(np.trace(K @ K))
        Y_norm = Y / np.sqrt(np.trace(Y @ Y))
        
        # Compute alignment
        alignment = np.trace(K_norm @ Y_norm)
        
        return alignment
        
    def kernel_matrix_analysis(self, X: np.ndarray) -> dict:
        """Analyze properties of the quantum kernel matrix.
        
        Args:
            X: Input samples
            
        Returns:
            Dictionary with kernel matrix properties
        """
        K = self.compute_kernel_matrix(X)
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(K)
        eigenvals = np.real(eigenvals)  # Should be real for valid kernel
        
        analysis = {
            'rank': np.linalg.matrix_rank(K),
            'condition_number': np.max(eigenvals) / np.max(eigenvals[eigenvals > 1e-12]),
            'trace': np.trace(K),
            'frobenius_norm': np.linalg.norm(K, 'fro'),
            'min_eigenvalue': np.min(eigenvals),
            'max_eigenvalue': np.max(eigenvals),
            'effective_dimension': np.sum(eigenvals) ** 2 / np.sum(eigenvals ** 2),
            'is_positive_semidefinite': np.all(eigenvals >= -1e-12)
        }
        
        return analysis
        
    def visualize_kernel_matrix(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Visualize the quantum kernel matrix.
        
        Args:
            X: Input samples
            y: Labels for ordering (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for visualization")
            return
            
        K = self.compute_kernel_matrix(X)
        
        # Sort by labels if provided
        if y is not None:
            sort_idx = np.argsort(y)
            K = K[np.ix_(sort_idx, sort_idx)]
            
        plt.figure(figsize=(8, 6))
        plt.imshow(K, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Kernel Value')
        plt.title('Quantum Kernel Matrix')
        plt.xlabel('Sample Index')
        plt.ylabel('Sample Index')
        plt.show()
        
    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors from trained SVM.
        
        Returns:
            Indices of support vectors
        """
        if not self._is_trained:
            raise ValueError("Model must be trained first")
            
        return self.svm.support_
        
    def get_dual_coefficients(self) -> np.ndarray:
        """Get dual coefficients from trained SVM.
        
        Returns:
            Dual coefficients
        """
        if not self._is_trained:
            raise ValueError("Model must be trained first")
            
        return self.svm.dual_coef_
        
    def save_model(self, filepath: str):
        """Save quantum kernel model.
        
        Args:
            filepath: Path to save model
        """
        import pickle
        
        model_data = {
            'feature_map': self.feature_map,
            'kernel_type': self.kernel_type,
            'svm': self.svm,
            'X_train': self.X_train,
            'kernel_matrix_train': self.kernel_matrix_train,
            'is_trained': self._is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, filepath: str):
        """Load quantum kernel model.
        
        Args:
            filepath: Path to load model from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.feature_map = model_data['feature_map']
        self.kernel_type = model_data['kernel_type']
        self.svm = model_data['svm']
        self.X_train = model_data['X_train']
        self.kernel_matrix_train = model_data['kernel_matrix_train']
        self._is_trained = model_data['is_trained']
        
    def __repr__(self) -> str:
        return f"QuantumKernel(kernel_type={self.kernel_type}, n_qubits={self.feature_map.n_qubits}, trained={self._is_trained})"


class QuantumKernelRegressor(QuantumKernel):
    """Quantum Kernel Regressor using SVR."""
    
    def __init__(self, feature_map: FeatureMap, 
                 kernel_type: str = 'fidelity',
                 svr_params: Optional[dict] = None,
                 shots: int = 1024,
                 backend: str = 'statevector'):
        """Initialize Quantum Kernel Regressor.
        
        Args:
            feature_map: Quantum feature map
            kernel_type: Type of quantum kernel
            svr_params: Parameters for SVR
            shots: Number of measurement shots
            backend: Quantum backend
        """
        super().__init__(feature_map, kernel_type, None, shots, backend)
        
        # Replace SVC with SVR
        from sklearn.svm import SVR
        
        if svr_params is None:
            svr_params = {'C': 1.0, 'kernel': 'precomputed'}
        else:
            svr_params['kernel'] = 'precomputed'
            
        self.svm = SVR(**svr_params)
        
    def forward_probs(self, x: np.ndarray) -> np.ndarray:
        """Predict regression value.
        
        Args:
            x: Input sample
            
        Returns:
            Predicted value (as single-element array)
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Compute kernel with training data
        K_test = self.compute_kernel_matrix(np.array([x]), self.X_train)
        
        # Get SVR prediction
        prediction = self.svm.predict(K_test)
        
        return np.array([prediction[0]])
        
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean squared error.
        
        Args:
            X: Input samples
            y: True values
            
        Returns:
            MSE loss
        """
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)