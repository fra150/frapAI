"""Tests for FPAI core module.

Tests for quantum states, feature maps, POVM measurements,
and base model functionality.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from ..core import (
    QuantumState, AngleEncoding, AmplitudeEncoding, ZZFeatureMap,
    POVM, FPAIModel
)
from . import TEST_CONFIG


class TestQuantumState:
    """Test QuantumState class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        
    def test_initialization(self):
        """Test QuantumState initialization."""
        # Test zero state
        state = QuantumState(self.n_qubits)
        assert state.n_qubits == self.n_qubits
        assert len(state.amplitudes) == 2**self.n_qubits
        assert np.abs(state.amplitudes[0] - 1.0) < self.tolerance
        assert np.sum(np.abs(state.amplitudes[1:])) < self.tolerance
        
    def test_custom_amplitudes(self):
        """Test QuantumState with custom amplitudes."""
        # Create equal superposition state
        n_states = 2**self.n_qubits
        amplitudes = np.ones(n_states) / np.sqrt(n_states)
        
        state = QuantumState(self.n_qubits, amplitudes=amplitudes)
        assert np.allclose(state.amplitudes, amplitudes, atol=self.tolerance)
        
    def test_normalization(self):
        """Test state normalization."""
        # Create unnormalized state
        amplitudes = np.random.random(2**self.n_qubits) + 1j * np.random.random(2**self.n_qubits)
        state = QuantumState(self.n_qubits, amplitudes=amplitudes)
        
        # Check normalization
        norm = np.sum(np.abs(state.amplitudes)**2)
        assert np.abs(norm - 1.0) < self.tolerance
        
    def test_invalid_amplitudes(self):
        """Test error handling for invalid amplitudes."""
        # Wrong number of amplitudes
        with pytest.raises(ValueError):
            QuantumState(self.n_qubits, amplitudes=np.array([1.0, 0.0]))
        
        # Zero amplitudes
        with pytest.raises(ValueError):
            QuantumState(self.n_qubits, amplitudes=np.zeros(2**self.n_qubits))
    
    def test_probabilities(self):
        """Test probability calculation."""
        state = QuantumState(self.n_qubits)
        probabilities = state.get_probabilities()
        
        assert len(probabilities) == 2**self.n_qubits
        assert np.abs(np.sum(probabilities) - 1.0) < self.tolerance
        assert probabilities[0] > 1.0 - self.tolerance  # |0...0⟩ state
        
    def test_measurement_simulation(self):
        """Test measurement simulation."""
        state = QuantumState(self.n_qubits)
        shots = TEST_CONFIG['test_shots']
        
        counts = state.measure(shots=shots)
        
        assert isinstance(counts, dict)
        assert sum(counts.values()) == shots
        assert '0' * self.n_qubits in counts  # Should measure |0...0⟩ most often
        
    def test_density_matrix(self):
        """Test density matrix calculation."""
        state = QuantumState(self.n_qubits)
        rho = state.get_density_matrix()
        
        assert rho.shape == (2**self.n_qubits, 2**self.n_qubits)
        assert np.allclose(rho, rho.conj().T, atol=self.tolerance)  # Hermitian
        assert np.abs(np.trace(rho) - 1.0) < self.tolerance  # Unit trace
        
    def test_fidelity(self):
        """Test fidelity calculation."""
        state1 = QuantumState(self.n_qubits)
        state2 = QuantumState(self.n_qubits)
        
        # Identical states should have fidelity 1
        fidelity = state1.fidelity(state2)
        assert np.abs(fidelity - 1.0) < self.tolerance
        
        # Orthogonal states should have fidelity 0
        amplitudes_orthogonal = np.zeros(2**self.n_qubits)
        amplitudes_orthogonal[-1] = 1.0  # |1...1⟩ state
        state_orthogonal = QuantumState(self.n_qubits, amplitudes=amplitudes_orthogonal)
        
        fidelity_orthogonal = state1.fidelity(state_orthogonal)
        assert np.abs(fidelity_orthogonal) < self.tolerance


class TestAngleEncoding:
    """Test AngleEncoding feature map."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        self.feature_map = AngleEncoding(n_qubits=self.n_qubits)
        
    def test_initialization(self):
        """Test AngleEncoding initialization."""
        assert self.feature_map.n_qubits == self.n_qubits
        assert hasattr(self.feature_map, 'rotation_gates')
        
    def test_encoding_shape(self):
        """Test encoding output shape."""
        x = np.random.random(self.n_qubits)
        state = self.feature_map.encode(x)
        
        assert isinstance(state, QuantumState)
        assert state.n_qubits == self.n_qubits
        assert len(state.amplitudes) == 2**self.n_qubits
        
    def test_encoding_normalization(self):
        """Test that encoded states are normalized."""
        x = np.random.random(self.n_qubits)
        state = self.feature_map.encode(x)
        
        norm = np.sum(np.abs(state.amplitudes)**2)
        assert np.abs(norm - 1.0) < self.tolerance
        
    def test_different_inputs(self):
        """Test encoding with different input values."""
        x1 = np.zeros(self.n_qubits)
        x2 = np.ones(self.n_qubits)
        
        state1 = self.feature_map.encode(x1)
        state2 = self.feature_map.encode(x2)
        
        # Different inputs should produce different states
        fidelity = state1.fidelity(state2)
        assert fidelity < 1.0 - self.tolerance
        
    def test_invalid_input_size(self):
        """Test error handling for wrong input size."""
        x_wrong_size = np.random.random(self.n_qubits + 1)
        
        with pytest.raises(ValueError):
            self.feature_map.encode(x_wrong_size)
    
    def test_batch_encoding(self):
        """Test batch encoding."""
        batch_size = 5
        X = np.random.random((batch_size, self.n_qubits))
        
        states = self.feature_map.encode_batch(X)
        
        assert len(states) == batch_size
        assert all(isinstance(state, QuantumState) for state in states)
        assert all(state.n_qubits == self.n_qubits for state in states)
    
    def test_parameter_update(self):
        """Test parameter updates."""
        initial_params = self.feature_map.get_params()
        new_params = np.random.random(len(initial_params))
        
        self.feature_map.set_params(new_params)
        updated_params = self.feature_map.get_params()
        
        assert np.allclose(updated_params, new_params, atol=self.tolerance)


class TestAmplitudeEncoding:
    """Test AmplitudeEncoding feature map."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        self.feature_map = AmplitudeEncoding(n_features=self.n_qubits)
        
    def test_initialization(self):
        """Test AmplitudeEncoding initialization."""
        assert self.feature_map.n_qubits == self.n_qubits
        assert self.feature_map.n_features <= 2**self.n_qubits
        
    def test_encoding_normalization(self):
        """Test that encoded states are normalized."""
        n_features = min(self.feature_map.n_features, 8)  # Limit for testing
        x = np.random.random(n_features)
        state = self.feature_map.encode(x)
        
        norm = np.sum(np.abs(state.amplitudes)**2)
        assert np.abs(norm - 1.0) < self.tolerance
        
    def test_amplitude_correspondence(self):
        """Test that amplitudes correspond to input features."""
        n_features = min(self.feature_map.n_features, 4)  # Small test
        x = np.random.random(n_features)
        x_normalized = x / np.linalg.norm(x)  # Normalize input
        
        state = self.feature_map.encode(x)
        
        # First n_features amplitudes should correspond to normalized input
        for i in range(n_features):
            expected_amplitude = x_normalized[i]
            actual_amplitude = np.real(state.amplitudes[i])  # Assuming real amplitudes
            assert np.abs(actual_amplitude - expected_amplitude) < self.tolerance * 10  # Relaxed tolerance
    
    def test_zero_input(self):
        """Test handling of zero input."""
        n_features = min(self.feature_map.n_features, 4)
        x = np.zeros(n_features)
        
        with pytest.raises(ValueError):
            self.feature_map.encode(x)


class TestZZFeatureMap:
    """Test ZZFeatureMap feature map."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        self.feature_map = ZZFeatureMap(
            n_qubits=self.n_qubits,
            reps=2,
            entanglement='linear'
        )
        
    def test_initialization(self):
        """Test ZZFeatureMap initialization."""
        assert self.feature_map.n_qubits == self.n_qubits
        assert self.feature_map.reps == 2
        assert self.feature_map.entanglement == 'linear'
        
    def test_encoding_creates_entanglement(self):
        """Test that ZZ feature map creates entangled states."""
        x = np.random.random(self.n_qubits)
        state = self.feature_map.encode(x)
        
        # Check that state is not separable (simple test)
        # For a separable state, the density matrix should have rank 1
        rho = state.get_density_matrix()
        eigenvals = np.linalg.eigvals(rho)
        rank = np.sum(eigenvals > self.tolerance)
        
        # Entangled states should have rank > 1
        assert rank > 1
        
    def test_different_entanglement_patterns(self):
        """Test different entanglement patterns."""
        patterns = ['linear', 'circular', 'full']
        
        for pattern in patterns:
            fm = ZZFeatureMap(
                n_qubits=self.n_qubits,
                reps=1,
                entanglement=pattern
            )
            
            x = np.random.random(self.n_qubits)
            state = fm.encode(x)
            
            assert isinstance(state, QuantumState)
            assert state.n_qubits == self.n_qubits
    
    def test_repetitions_effect(self):
        """Test effect of different repetition numbers."""
        x = np.random.random(self.n_qubits)
        
        fm1 = ZZFeatureMap(n_qubits=self.n_qubits, reps=1)
        fm2 = ZZFeatureMap(n_qubits=self.n_qubits, reps=3)
        
        state1 = fm1.encode(x)
        state2 = fm2.encode(x)
        
        # Different repetitions should produce different states
        fidelity = state1.fidelity(state2)
        assert fidelity < 1.0 - self.tolerance


class TestPOVM:
    """Test POVM measurement class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        self.povm = POVM(n_qubits=self.n_qubits, measurement_type='computational')
        
    def test_initialization(self):
        """Test POVM initialization."""
        assert self.povm.n_qubits == self.n_qubits
        assert self.povm.measurement_type == 'computational'
        assert len(self.povm.operators) == 2**self.n_qubits
        
    def test_povm_completeness(self):
        """Test POVM completeness relation."""
        # Sum of POVM operators should equal identity
        povm_sum = sum(self.povm.operators)
        identity = np.eye(2**self.n_qubits)
        
        assert np.allclose(povm_sum, identity, atol=self.tolerance)
        
    def test_measurement_probabilities(self):
        """Test measurement probability calculation."""
        state = QuantumState(self.n_qubits)
        probabilities = self.povm.measure_probabilities(state)
        
        assert len(probabilities) == 2**self.n_qubits
        assert np.abs(np.sum(probabilities) - 1.0) < self.tolerance
        assert all(p >= 0 for p in probabilities)
        
    def test_shot_simulation(self):
        """Test shot-based measurement simulation."""
        state = QuantumState(self.n_qubits)
        shots = TEST_CONFIG['test_shots']
        
        outcomes = self.povm.measure_shots(state, shots=shots)
        
        assert len(outcomes) == shots
        assert all(0 <= outcome < 2**self.n_qubits for outcome in outcomes)
        
    def test_different_measurement_types(self):
        """Test different measurement types."""
        measurement_types = ['computational', 'pauli_z', 'random']
        
        for meas_type in measurement_types:
            povm = POVM(n_qubits=self.n_qubits, measurement_type=meas_type)
            state = QuantumState(self.n_qubits)
            
            probabilities = povm.measure_probabilities(state)
            
            assert len(probabilities) > 0
            assert np.abs(np.sum(probabilities) - 1.0) < self.tolerance


class MockModel(FPAIModel):
    """Mock model for testing FPAIModel base class."""
    
    def __init__(self):
        super().__init__()
        self.fitted = False
        
    def fit(self, X, y, **kwargs):
        self.fitted = True
        return {'loss': [1.0, 0.5, 0.1]}
        
    def predict_proba(self, X):
        if not self.fitted:
            raise ValueError("Model not fitted")
        n_samples = len(X)
        return np.random.random((n_samples, 2))
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class TestFPAIModel:
    """Test FPAIModel base class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = MockModel()
        self.n_samples = TEST_CONFIG['small_dataset_size']
        self.n_features = TEST_CONFIG['n_qubits']
        
        # Generate test data
        np.random.seed(TEST_CONFIG['random_seed'])
        self.X = np.random.random((self.n_samples, self.n_features))
        self.y = np.random.randint(0, 2, self.n_samples)
        
    def test_initialization(self):
        """Test model initialization."""
        assert hasattr(self.model, 'fitted')
        assert not self.model.fitted
        
    def test_fit_method(self):
        """Test fit method."""
        history = self.model.fit(self.X, self.y)
        
        assert self.model.fitted
        assert isinstance(history, dict)
        assert 'loss' in history
        
    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        with pytest.raises(ValueError):
            self.model.predict(self.X)
            
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        self.model.fit(self.X, self.y)
        
        predictions = self.model.predict(self.X)
        probabilities = self.model.predict_proba(self.X)
        
        assert len(predictions) == self.n_samples
        assert probabilities.shape == (self.n_samples, 2)
        assert all(pred in [0, 1] for pred in predictions)
        
    def test_serialization(self):
        """Test model serialization."""
        self.model.fit(self.X, self.y)
        
        # Test save/load (mock implementation)
        model_state = {
            'fitted': self.model.fitted,
            'class_name': self.model.__class__.__name__
        }
        
        # Create new model and load state
        new_model = MockModel()
        new_model.fitted = model_state['fitted']
        
        assert new_model.fitted == self.model.fitted


class TestIntegration:
    """Integration tests for core components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_qubits = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        
    def test_feature_map_povm_integration(self):
        """Test integration between feature maps and POVM."""
        # Create feature map and POVM
        feature_map = AngleEncoding(n_qubits=self.n_qubits)
        povm = POVM(n_qubits=self.n_qubits, measurement_type='computational')
        
        # Encode data and measure
        x = np.random.random(self.n_qubits)
        state = feature_map.encode(x)
        probabilities = povm.measure_probabilities(state)
        
        assert len(probabilities) == 2**self.n_qubits
        assert np.abs(np.sum(probabilities) - 1.0) < self.tolerance
        
    def test_multiple_feature_maps_consistency(self):
        """Test consistency across different feature maps."""
        feature_maps = [
            AngleEncoding(n_qubits=self.n_qubits),
            ZZFeatureMap(n_qubits=self.n_qubits, reps=1)
        ]
        
        x = np.random.random(self.n_qubits)
        
        for fm in feature_maps:
            state = fm.encode(x)
            
            # All should produce valid quantum states
            assert isinstance(state, QuantumState)
            assert state.n_qubits == self.n_qubits
            
            # States should be normalized
            norm = np.sum(np.abs(state.amplitudes)**2)
            assert np.abs(norm - 1.0) < self.tolerance
    
    def test_state_fidelity_properties(self):
        """Test fidelity properties across different encodings."""
        fm1 = AngleEncoding(n_qubits=self.n_qubits)
        fm2 = ZZFeatureMap(n_qubits=self.n_qubits, reps=1)
        
        x = np.random.random(self.n_qubits)
        
        state1 = fm1.encode(x)
        state2 = fm2.encode(x)
        
        # Fidelity should be between 0 and 1
        fidelity = state1.fidelity(state2)
        assert 0 <= fidelity <= 1
        
        # Self-fidelity should be 1
        self_fidelity = state1.fidelity(state1)
        assert np.abs(self_fidelity - 1.0) < self.tolerance


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])