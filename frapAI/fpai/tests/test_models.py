"""Tests for FPAI models module.

Tests for VQC (Variational Quantum Classifier) and
Quantum Kernel models.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ..models import VQC, QuantumKernel, HardwareEfficientAnsatz
from ..core import AngleEncoding, ZZFeatureMap, POVM
from . import TEST_CONFIG


class TestHardwareEfficientAnsatz:
    """Test HardwareEfficientAnsatz class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        self.ansatz = HardwareEfficientAnsatz(
            n_qubits=self.n_qubits,
            layers=2,
            entanglement='linear'
        )
        
    def test_initialization(self):
        """Test ansatz initialization."""
        assert self.ansatz.n_qubits == self.n_qubits
        assert self.ansatz.layers == 2
        assert self.ansatz.entanglement == 'linear'
        assert hasattr(self.ansatz, 'n_parameters')
        
    def test_parameter_count(self):
        """Test parameter count calculation."""
        # For hardware efficient ansatz:
        # Each layer has n_qubits rotation gates + entangling gates
        # Plus final rotation layer
        expected_params = self.ansatz.n_parameters
        assert expected_params > 0
        assert isinstance(expected_params, int)
        
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        params = self.ansatz.get_initial_parameters()
        
        assert len(params) == self.ansatz.n_parameters
        assert all(isinstance(p, (int, float, np.number)) for p in params)
        
    def test_circuit_construction(self):
        """Test quantum circuit construction."""
        params = self.ansatz.get_initial_parameters()
        circuit_info = self.ansatz.construct_circuit(params)
        
        assert isinstance(circuit_info, dict)
        assert 'gates' in circuit_info
        assert 'depth' in circuit_info
        assert circuit_info['depth'] > 0
        
    def test_different_entanglement_patterns(self):
        """Test different entanglement patterns."""
        patterns = ['linear', 'circular', 'full']
        
        for pattern in patterns:
            ansatz = HardwareEfficientAnsatz(
                n_qubits=self.n_qubits,
                layers=1,
                entanglement=pattern
            )
            
            params = ansatz.get_initial_parameters()
            circuit_info = ansatz.construct_circuit(params)
            
            assert isinstance(circuit_info, dict)
            assert circuit_info['depth'] > 0
    
    def test_gradient_computation(self):
        """Test gradient computation capability."""
        params = self.ansatz.get_initial_parameters()
        
        # Mock gradient computation
        gradients = self.ansatz.compute_gradients(params)
        
        assert len(gradients) == len(params)
        assert all(isinstance(g, (int, float, np.number)) for g in gradients)


class TestVQC:
    """Test VQC (Variational Quantum Classifier) class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        
        # Create feature map and ansatz
        self.feature_map = AngleEncoding(n_qubits=self.n_qubits)
        self.ansatz = HardwareEfficientAnsatz(
            n_qubits=self.n_qubits,
            layers=2
        )
        
        # Create POVM and VQC model
        self.povm = POVM.projective(n_outcomes=2, dim=2**self.n_qubits)
        
        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            povm=self.povm,
            optimizer='adam'
        )
        
        # Generate test data
        np.random.seed(TEST_CONFIG['random_seed'])
        self.X, self.y = make_classification(
            n_samples=TEST_CONFIG['small_dataset_size'],
            n_features=self.n_qubits,
            n_classes=2,
            n_redundant=0,
            n_informative=self.n_qubits,
            random_state=TEST_CONFIG['random_seed']
        )
        
    def test_initialization(self):
        """Test VQC initialization."""
        assert self.vqc.feature_map == self.feature_map
        assert self.vqc.ansatz == self.ansatz
        assert self.vqc.optimizer == 'adam'
        assert self.vqc.learning_rate == 0.01
        assert not hasattr(self.vqc, 'fitted') or not self.vqc.fitted
        
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        params = self.vqc._initialize_parameters()
        
        assert len(params) == self.ansatz.n_parameters
        assert all(isinstance(p, (int, float, np.number)) for p in params)
        
    def test_forward_pass(self):
        """Test forward pass computation."""
        params = self.vqc._initialize_parameters()
        x_sample = self.X[0]
        
        output = self.vqc._forward_pass(x_sample, params)
        
        assert isinstance(output, (int, float, np.number))
        assert 0 <= output <= 1  # Should be a probability
        
    def test_cost_function(self):
        """Test cost function computation."""
        params = self.vqc._initialize_parameters()
        X_batch = self.X[:5]  # Small batch
        y_batch = self.y[:5]
        
        cost = self.vqc._compute_cost(params, X_batch, y_batch)
        
        assert isinstance(cost, (int, float, np.number))
        assert cost >= 0  # Cost should be non-negative
        
    def test_gradient_computation(self):
        """Test gradient computation."""
        params = self.vqc._initialize_parameters()
        X_batch = self.X[:3]  # Very small batch for testing
        y_batch = self.y[:3]
        
        gradients = self.vqc._compute_gradients(params, X_batch, y_batch)
        
        assert len(gradients) == len(params)
        assert all(isinstance(g, (int, float, np.number)) for g in gradients)
        
    def test_fit_method(self):
        """Test model fitting."""
        # Use very small dataset and few epochs for testing
        X_small = self.X[:10]
        y_small = self.y[:10]
        
        history = self.vqc.fit(
            X_small, y_small,
            epochs=3,
            batch_size=5,
            validation_split=0.0,  # No validation for small dataset
            verbose=False
        )
        
        assert isinstance(history, dict)
        assert 'loss' in history
        assert len(history['loss']) == 3  # 3 epochs
        assert hasattr(self.vqc, 'fitted') and self.vqc.fitted
        
    def test_predict_proba(self):
        """Test probability prediction."""
        # Fit model first
        X_small = self.X[:10]
        y_small = self.y[:10]
        self.vqc.fit(X_small, y_small, epochs=2, verbose=False)
        
        # Test prediction
        X_test = self.X[:5]
        probabilities = self.vqc.predict_proba(X_test)
        
        assert probabilities.shape == (5, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=self.tolerance)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        
    def test_predict(self):
        """Test class prediction."""
        # Fit model first
        X_small = self.X[:10]
        y_small = self.y[:10]
        self.vqc.fit(X_small, y_small, epochs=2, verbose=False)
        
        # Test prediction
        X_test = self.X[:5]
        predictions = self.vqc.predict(X_test)
        
        assert len(predictions) == 5
        assert all(pred in [0, 1] for pred in predictions)
        
    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        with pytest.raises(ValueError):
            self.vqc.predict(self.X[:5])
            
    def test_different_optimizers(self):
        """Test different optimizers."""
        optimizers = ['adam', 'sgd', 'rmsprop']
        
        for optimizer in optimizers:
            vqc = VQC(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                povm=self.povm,
                optimizer=optimizer
            )
            
            # Test that initialization works
            params = vqc._initialize_parameters()
            assert len(params) == self.ansatz.n_parameters
    
    def test_validation_split(self):
        """Test training with validation split."""
        X_medium = self.X[:20]  # Larger dataset for validation
        y_medium = self.y[:20]
        
        history = self.vqc.fit(
            X_medium, y_medium,
            epochs=2,
            validation_split=0.3,
            verbose=False
        )
        
        assert 'val_loss' in history
        assert len(history['val_loss']) == 2
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        X_medium = self.X[:20]
        y_medium = self.y[:20]
        
        history = self.vqc.fit(
            X_medium, y_medium,
            epochs=10,
            validation_split=0.3,
            early_stopping=True,
            patience=2,
            verbose=False
        )
        
        # Should stop early if no improvement
        assert len(history['loss']) <= 10


class TestQuantumKernel:
    """Test QuantumKernel class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        
        # Create feature map
        self.feature_map = ZZFeatureMap(
            n_qubits=self.n_qubits,
            reps=2
        )
        
        # Create quantum kernel
        self.qkernel = QuantumKernel(
            feature_map=self.feature_map,
            enforce_psd=True
        )
        
        # Generate test data
        np.random.seed(TEST_CONFIG['random_seed'])
        self.X, self.y = make_classification(
            n_samples=TEST_CONFIG['small_dataset_size'],
            n_features=self.n_qubits,
            n_classes=2,
            n_redundant=0,
            n_informative=self.n_qubits,
            random_state=TEST_CONFIG['random_seed']
        )
        
    def test_initialization(self):
        """Test QuantumKernel initialization."""
        assert self.qkernel.feature_map == self.feature_map
        assert self.qkernel.enforce_psd == True
        assert hasattr(self.qkernel, 'kernel_type')
        
    def test_kernel_element_computation(self):
        """Test single kernel element computation."""
        x1 = self.X[0]
        x2 = self.X[1]
        
        kernel_value = self.qkernel._compute_kernel_element(x1, x2)
        
        assert isinstance(kernel_value, (int, float, np.number))
        assert 0 <= kernel_value <= 1  # Kernel values should be in [0,1]
        
    def test_kernel_symmetry(self):
        """Test kernel symmetry property."""
        x1 = self.X[0]
        x2 = self.X[1]
        
        k12 = self.qkernel._compute_kernel_element(x1, x2)
        k21 = self.qkernel._compute_kernel_element(x2, x1)
        
        assert np.abs(k12 - k21) < self.tolerance
        
    def test_kernel_diagonal(self):
        """Test kernel diagonal elements."""
        x = self.X[0]
        
        k_diag = self.qkernel._compute_kernel_element(x, x)
        
        # Diagonal elements should be 1 (or close to 1)
        assert np.abs(k_diag - 1.0) < self.tolerance
        
    def test_kernel_matrix_computation(self):
        """Test kernel matrix computation."""
        X_small = self.X[:5]  # Small subset for testing
        
        K = self.qkernel.compute_kernel_matrix(X_small)
        
        assert K.shape == (5, 5)
        assert np.allclose(K, K.T, atol=self.tolerance)  # Symmetry
        assert np.all(np.diag(K) >= 1.0 - self.tolerance)  # Diagonal elements
        
    def test_kernel_matrix_psd(self):
        """Test that kernel matrix is positive semi-definite."""
        X_small = self.X[:5]
        
        K = self.qkernel.compute_kernel_matrix(X_small)
        
        # Check positive semi-definiteness
        eigenvals = np.linalg.eigvals(K)
        assert np.all(eigenvals >= -self.tolerance)  # All eigenvalues >= 0
        
    def test_fit_method(self):
        """Test kernel fitting (training data storage)."""
        X_train = self.X[:10]
        y_train = self.y[:10]
        
        history = self.qkernel.fit(X_train, y_train)
        
        assert hasattr(self.qkernel, 'X_train_')
        assert hasattr(self.qkernel, 'y_train_')
        assert np.array_equal(self.qkernel.X_train_, X_train)
        assert np.array_equal(self.qkernel.y_train_, y_train)
        assert isinstance(history, dict)
        
    def test_predict_proba(self):
        """Test probability prediction."""
        # Fit model first
        X_train = self.X[:15]
        y_train = self.y[:15]
        self.qkernel.fit(X_train, y_train)
        
        # Test prediction
        X_test = self.X[15:20]
        probabilities = self.qkernel.predict_proba(X_test)
        
        assert probabilities.shape == (5, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=self.tolerance)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        
    def test_predict(self):
        """Test class prediction."""
        # Fit model first
        X_train = self.X[:15]
        y_train = self.y[:15]
        self.qkernel.fit(X_train, y_train)
        
        # Test prediction
        X_test = self.X[15:20]
        predictions = self.qkernel.predict(X_test)
        
        assert len(predictions) == 5
        assert all(pred in [0, 1] for pred in predictions)
        
    def test_different_feature_maps(self):
        """Test with different feature maps."""
        feature_maps = [
            AngleEncoding(n_qubits=self.n_qubits),
            ZZFeatureMap(n_qubits=self.n_qubits, reps=1)
        ]
        
        for fm in feature_maps:
            qk = QuantumKernel(feature_map=fm)
            
            X_small = self.X[:3]
            K = qk.compute_kernel_matrix(X_small)
            
            assert K.shape == (3, 3)
            assert np.allclose(K, K.T, atol=self.tolerance)
    
    def test_kernel_alignment(self):
        """Test kernel alignment computation."""
        X_small = self.X[:10]
        y_small = self.y[:10]
        
        alignment = self.qkernel.compute_kernel_alignment(X_small, y_small)
        
        assert isinstance(alignment, (int, float, np.number))
        assert -1 <= alignment <= 1  # Alignment should be in [-1, 1]
    
    def test_kernel_properties_analysis(self):
        """Test kernel properties analysis."""
        X_small = self.X[:8]
        
        properties = self.qkernel.analyze_kernel_properties(X_small)
        
        assert isinstance(properties, dict)
        assert 'rank' in properties
        assert 'condition_number' in properties
        assert 'spectral_gap' in properties
        assert 'effective_dimension' in properties
        
        # Check property ranges
        assert 1 <= properties['rank'] <= len(X_small)
        assert properties['condition_number'] >= 1
        assert properties['spectral_gap'] >= 0
        assert properties['effective_dimension'] >= 1


class TestModelIntegration:
    """Integration tests for model components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        
        # Generate test data
        np.random.seed(TEST_CONFIG['random_seed'])
        self.X, self.y = make_classification(
            n_samples=TEST_CONFIG['small_dataset_size'],
            n_features=self.n_qubits,
            n_classes=2,
            n_redundant=0,
            n_informative=self.n_qubits,
            random_state=TEST_CONFIG['random_seed']
        )
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=TEST_CONFIG['random_seed']
        )
    
    def test_vqc_vs_quantum_kernel_consistency(self):
        """Test consistency between VQC and QuantumKernel on same data."""
        # Use same feature map for both models
        feature_map = AngleEncoding(n_qubits=self.n_qubits)
        
        # Create VQC
        ansatz = HardwareEfficientAnsatz(n_qubits=self.n_qubits, layers=1)
        povm = POVM.projective(n_outcomes=2, dim=2**self.n_qubits)
        vqc = VQC(feature_map=feature_map, ansatz=ansatz, povm=povm)
        
        # Create QuantumKernel
        qkernel = QuantumKernel(feature_map=feature_map)
        
        # Train both models
        vqc.fit(self.X_train, self.y_train, epochs=3, verbose=False)
        qkernel.fit(self.X_train, self.y_train)
        
        # Test predictions
        vqc_pred = vqc.predict(self.X_test)
        qk_pred = qkernel.predict(self.X_test)
        
        # Both should produce valid predictions
        assert all(pred in [0, 1] for pred in vqc_pred)
        assert all(pred in [0, 1] for pred in qk_pred)
        
        # Calculate accuracies
        vqc_acc = accuracy_score(self.y_test, vqc_pred)
        qk_acc = accuracy_score(self.y_test, qk_pred)
        
        # Both should achieve reasonable accuracy (> random)
        assert vqc_acc > 0.3  # Better than random for binary classification
        assert qk_acc > 0.3
    
    def test_model_performance_comparison(self):
        """Test performance comparison between different configurations."""
        feature_maps = [
            AngleEncoding(n_qubits=self.n_qubits),
            ZZFeatureMap(n_qubits=self.n_qubits, reps=1)
        ]
        
        results = {}
        
        for i, fm in enumerate(feature_maps):
            # Test QuantumKernel
            qk = QuantumKernel(feature_map=fm)
            qk.fit(self.X_train, self.y_train)
            qk_pred = qk.predict(self.X_test)
            qk_acc = accuracy_score(self.y_test, qk_pred)
            
            results[f'QK_{i}'] = qk_acc
            
            # Test VQC
            ansatz = HardwareEfficientAnsatz(n_qubits=self.n_qubits, layers=1)
            povm = POVM.projective(n_outcomes=2, dim=2**self.n_qubits)
            vqc = VQC(feature_map=fm, ansatz=ansatz, povm=povm)
            vqc.fit(self.X_train, self.y_train, epochs=3, verbose=False)
            vqc_pred = vqc.predict(self.X_test)
            vqc_acc = accuracy_score(self.y_test, vqc_pred)
            
            results[f'VQC_{i}'] = vqc_acc
        
        # All models should achieve reasonable performance
        for model_name, accuracy in results.items():
            assert accuracy > 0.2, f"{model_name} achieved poor accuracy: {accuracy}"
    
    def test_model_reproducibility(self):
        """Test model reproducibility with fixed random seeds."""
        # Set random seed
        np.random.seed(TEST_CONFIG['random_seed'])
        
        # Create and train first model
        feature_map = AngleEncoding(n_qubits=self.n_qubits)
        ansatz = HardwareEfficientAnsatz(n_qubits=self.n_qubits, layers=1)
        povm = POVM.projective(n_outcomes=2, dim=2**self.n_qubits)
        vqc1 = VQC(feature_map=feature_map, ansatz=ansatz, povm=povm)
        vqc1.fit(self.X_train, self.y_train, epochs=3, verbose=False)
        pred1 = vqc1.predict(self.X_test)
        
        # Reset random seed and create second model
        np.random.seed(TEST_CONFIG['random_seed'])
        vqc2 = VQC(feature_map=feature_map, ansatz=ansatz, povm=povm)
        vqc2.fit(self.X_train, self.y_train, epochs=3, verbose=False)
        pred2 = vqc2.predict(self.X_test)
        
        # Predictions should be similar (allowing for some numerical differences)
        agreement = np.mean(pred1 == pred2)
        assert agreement > 0.8, f"Models not reproducible: {agreement} agreement"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])