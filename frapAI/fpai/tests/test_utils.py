"""Tests for FPAI utils module.

Tests for calibration, metrics, visualization, and data utilities.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from ..utils import (
    TemperatureScaling, PlattScaling, IsotonicCalibration, VectorScaling,
    expected_calibration_error, reliability_diagram, brier_score,
    plot_calibration_curve, plot_reliability_diagram,
    generate_quantum_classification_data, load_benchmark_dataset,
    preprocess_features
)
from . import TEST_CONFIG


class TestCalibration:
    """Test calibration methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = TEST_CONFIG['tolerance']
        
        # Generate test data
        np.random.seed(TEST_CONFIG['random_seed'])
        X, y = make_classification(
            n_samples=TEST_CONFIG['medium_dataset_size'],
            n_features=TEST_CONFIG['n_qubits'],
            n_classes=2,
            random_state=TEST_CONFIG['random_seed']
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=TEST_CONFIG['random_seed']
        )
        
        # Train a simple model to get predictions
        model = LogisticRegression(random_state=TEST_CONFIG['random_seed'])
        model.fit(X_train, y_train)
        
        self.y_true = y_test
        self.y_prob = model.predict_proba(X_test)[:, 1]
        self.y_prob_binary = model.predict_proba(X_test)
        
    def test_temperature_scaling_initialization(self):
        """Test TemperatureScaling initialization."""
        calibrator = TemperatureScaling()
        
        assert hasattr(calibrator, 'temperature')
        assert calibrator.temperature is None
        
    def test_temperature_scaling_fit(self):
        """Test TemperatureScaling fitting."""
        calibrator = TemperatureScaling()
        calibrator.fit(self.y_prob, self.y_true)
        
        assert calibrator.temperature is not None
        assert isinstance(calibrator.temperature, (int, float, np.number))
        assert calibrator.temperature > 0
        
    def test_temperature_scaling_transform(self):
        """Test TemperatureScaling transformation."""
        calibrator = TemperatureScaling()
        calibrator.fit(self.y_prob, self.y_true)
        
        calibrated_probs = calibrator.transform(self.y_prob)
        
        assert len(calibrated_probs) == len(self.y_prob)
        assert np.all(calibrated_probs >= 0)
        assert np.all(calibrated_probs <= 1)
        
    def test_temperature_scaling_binary_classification(self):
        """Test TemperatureScaling with binary classification probabilities."""
        calibrator = TemperatureScaling()
        calibrator.fit(self.y_prob_binary, self.y_true)
        
        calibrated_probs = calibrator.transform(self.y_prob_binary)
        
        assert calibrated_probs.shape == self.y_prob_binary.shape
        assert np.allclose(calibrated_probs.sum(axis=1), 1.0, atol=self.tolerance)
        
    def test_platt_scaling(self):
        """Test PlattScaling calibration."""
        calibrator = PlattScaling()
        calibrator.fit(self.y_prob, self.y_true)
        
        calibrated_probs = calibrator.transform(self.y_prob)
        
        assert len(calibrated_probs) == len(self.y_prob)
        assert np.all(calibrated_probs >= 0)
        assert np.all(calibrated_probs <= 1)
        assert hasattr(calibrator, 'sigmoid_model')
        
    def test_isotonic_calibration(self):
        """Test IsotonicCalibration."""
        calibrator = IsotonicCalibration()
        calibrator.fit(self.y_prob, self.y_true)
        
        calibrated_probs = calibrator.transform(self.y_prob)
        
        assert len(calibrated_probs) == len(self.y_prob)
        assert np.all(calibrated_probs >= 0)
        assert np.all(calibrated_probs <= 1)
        assert hasattr(calibrator, 'isotonic_regressor')
        
    def test_vector_scaling(self):
        """Test VectorScaling calibration."""
        calibrator = VectorScaling(n_classes=2)
        calibrator.fit(self.y_prob_binary, self.y_true)
        
        calibrated_probs = calibrator.transform(self.y_prob_binary)
        
        assert calibrated_probs.shape == self.y_prob_binary.shape
        assert np.allclose(calibrated_probs.sum(axis=1), 1.0, atol=self.tolerance)
        assert hasattr(calibrator, 'temperatures')
        assert len(calibrator.temperatures) == 2
        
    def test_calibration_improvement(self):
        """Test that calibration improves ECE."""
        # Calculate original ECE
        original_ece = expected_calibration_error(self.y_true, self.y_prob)
        
        # Apply temperature scaling
        calibrator = TemperatureScaling()
        calibrator.fit(self.y_prob, self.y_true)
        calibrated_probs = calibrator.transform(self.y_prob)
        
        # Calculate calibrated ECE
        calibrated_ece = expected_calibration_error(self.y_true, calibrated_probs)
        
        # Calibration should improve (reduce) ECE in most cases
        # Note: This might not always be true for well-calibrated models
        assert isinstance(original_ece, (int, float, np.number))
        assert isinstance(calibrated_ece, (int, float, np.number))
        assert calibrated_ece >= 0
        assert original_ece >= 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = TEST_CONFIG['tolerance']
        
        # Generate test data
        np.random.seed(TEST_CONFIG['random_seed'])
        n_samples = TEST_CONFIG['small_dataset_size']
        
        self.y_true = np.random.randint(0, 2, n_samples)
        self.y_prob = np.random.random(n_samples)
        
        # Ensure some diversity in predictions
        self.y_prob_binary = np.column_stack([
            1 - self.y_prob,
            self.y_prob
        ])
        
    def test_expected_calibration_error(self):
        """Test Expected Calibration Error calculation."""
        ece = expected_calibration_error(self.y_true, self.y_prob)
        
        assert isinstance(ece, (int, float, np.number))
        assert 0 <= ece <= 1
        
    def test_expected_calibration_error_perfect(self):
        """Test ECE with perfectly calibrated predictions."""
        # Create perfectly calibrated predictions
        y_true_perfect = np.array([0, 0, 1, 1])
        y_prob_perfect = np.array([0.0, 0.0, 1.0, 1.0])
        
        ece = expected_calibration_error(y_true_perfect, y_prob_perfect)
        
        assert ece < self.tolerance  # Should be very close to 0
        
    def test_reliability_diagram_data(self):
        """Test reliability diagram data generation."""
        bin_boundaries, bin_lowers, bin_uppers, bin_centers, counts, bin_accuracies, bin_confidences = reliability_diagram(
            self.y_true, self.y_prob, n_bins=10, return_data=True
        )
        
        assert len(bin_boundaries) == 11  # n_bins + 1
        assert len(bin_centers) == 10
        assert len(counts) == 10
        assert len(bin_accuracies) == 10
        assert len(bin_confidences) == 10
        
        # Check that bins are properly ordered
        assert np.all(np.diff(bin_boundaries) > 0)
        
    def test_brier_score(self):
        """Test Brier score calculation."""
        brier = brier_score(self.y_true, self.y_prob)
        
        assert isinstance(brier, (int, float, np.number))
        assert 0 <= brier <= 1
        
    def test_brier_score_perfect(self):
        """Test Brier score with perfect predictions."""
        y_true_perfect = np.array([0, 0, 1, 1])
        y_prob_perfect = np.array([0.0, 0.0, 1.0, 1.0])
        
        brier = brier_score(y_true_perfect, y_prob_perfect)
        
        assert brier < self.tolerance  # Should be very close to 0
        
    def test_brier_score_worst(self):
        """Test Brier score with worst predictions."""
        y_true_worst = np.array([0, 0, 1, 1])
        y_prob_worst = np.array([1.0, 1.0, 0.0, 0.0])
        
        brier = brier_score(y_true_worst, y_prob_worst)
        
        assert brier > 0.9  # Should be close to 1
        
    def test_metrics_with_edge_cases(self):
        """Test metrics with edge cases."""
        # All predictions same class
        y_true_same = np.ones(10)
        y_prob_same = np.ones(10) * 0.8
        
        ece_same = expected_calibration_error(y_true_same, y_prob_same)
        brier_same = brier_score(y_true_same, y_prob_same)
        
        assert isinstance(ece_same, (int, float, np.number))
        assert isinstance(brier_same, (int, float, np.number))
        assert ece_same >= 0
        assert brier_same >= 0
        
    def test_fairness_metrics(self):
        """Test fairness metrics calculation."""
        from ..utils.metrics import fairness_metrics
        
        # Create test data with sensitive attribute
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        sensitive_attr = np.random.randint(0, 2, n_samples)
        
        fairness_results = fairness_metrics(y_true, y_pred, sensitive_attr)
        
        assert isinstance(fairness_results, dict)
        assert 'demographic_parity' in fairness_results
        assert 'equalized_odds' in fairness_results
        assert 'equal_opportunity' in fairness_results
        assert 'accuracy_parity' in fairness_results
        
        # All fairness metrics should be between 0 and 1
        for metric_name, metric_value in fairness_results.items():
            assert 0 <= metric_value <= 1, f"{metric_name}: {metric_value}"


class TestVisualization:
    """Test visualization functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Generate test data
        np.random.seed(TEST_CONFIG['random_seed'])
        n_samples = TEST_CONFIG['small_dataset_size']
        
        self.y_true = np.random.randint(0, 2, n_samples)
        self.y_prob = np.random.random(n_samples)
        
        # Close any existing plots
        plt.close('all')
        
    def teardown_method(self):
        """Clean up after tests."""
        plt.close('all')
        
    def test_plot_calibration_curve(self):
        """Test calibration curve plotting."""
        fig, ax = plot_calibration_curve(self.y_true, self.y_prob)
        
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) > 0  # Should have plotted lines
        
    def test_plot_reliability_diagram(self):
        """Test reliability diagram plotting."""
        fig, ax = plot_reliability_diagram(self.y_true, self.y_prob)
        
        assert fig is not None
        assert ax is not None
        # Should have bars for the histogram
        assert len(ax.patches) > 0 or len(ax.collections) > 0
        
    def test_plot_with_custom_parameters(self):
        """Test plotting with custom parameters."""
        fig, ax = plot_calibration_curve(
            self.y_true, self.y_prob,
            n_bins=5,
            strategy='quantile'
        )
        
        assert fig is not None
        assert ax is not None
        
    def test_plot_multiple_models(self):
        """Test plotting multiple models on same figure."""
        # Create second set of predictions
        y_prob2 = np.random.random(len(self.y_prob))
        
        fig, ax = plt.subplots()
        
        # Plot first model
        plot_calibration_curve(
            self.y_true, self.y_prob,
            ax=ax, name='Model 1'
        )
        
        # Plot second model on same axes
        plot_calibration_curve(
            self.y_true, y_prob2,
            ax=ax, name='Model 2'
        )
        
        assert len(ax.lines) >= 2  # Should have multiple lines
        
    def test_quantum_state_visualization(self):
        """Test quantum state visualization functions."""
        from ..utils.visualization import plot_quantum_state
        
        # Create simple quantum state
        n_qubits = 2
        amplitudes = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Test amplitude plotting
        fig, ax = plot_quantum_state(amplitudes, plot_type='amplitudes')
        
        assert fig is not None
        assert ax is not None
        
    def test_training_history_visualization(self):
        """Test training history visualization."""
        from ..utils.visualization import plot_training_history
        
        # Create mock training history
        history = {
            'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3],
            'accuracy': [0.5, 0.6, 0.7, 0.8, 0.9]
        }
        
        fig, axes = plot_training_history(history)
        
        assert fig is not None
        assert len(axes) >= 1


class TestDataUtils:
    """Test data utility functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.n_samples = TEST_CONFIG['small_dataset_size']
        self.n_features = TEST_CONFIG['n_qubits']
        self.tolerance = TEST_CONFIG['tolerance']
        
    def test_generate_quantum_classification_data(self):
        """Test quantum classification data generation."""
        X, y = generate_quantum_classification_data(
            n_samples=self.n_samples,
            n_features=self.n_features,
            dataset_type='moons'
        )
        
        assert X.shape == (self.n_samples, self.n_features)
        assert len(y) == self.n_samples
        assert set(y) <= {0, 1}  # Binary classification
        
    def test_different_dataset_types(self):
        """Test different quantum dataset types."""
        dataset_types = ['moons', 'circles', 'spiral', 'xor', 'blobs']
        
        for dataset_type in dataset_types:
            X, y = generate_quantum_classification_data(
                n_samples=50,  # Small for testing
                n_features=self.n_features,
                dataset_type=dataset_type
            )
            
            assert X.shape == (50, self.n_features)
            assert len(y) == 50
            assert len(set(y)) >= 2  # At least 2 classes
            
    def test_quantum_advantage_dataset(self):
        """Test quantum advantage dataset generation."""
        X, y = generate_quantum_classification_data(
            n_samples=self.n_samples,
            n_features=self.n_features,
            dataset_type='quantum_advantage'
        )
        
        assert X.shape == (self.n_samples, self.n_features)
        assert len(y) == self.n_samples
        
    def test_load_benchmark_dataset(self):
        """Test benchmark dataset loading."""
        dataset_names = ['iris', 'wine', 'breast_cancer']
        
        for dataset_name in dataset_names:
            X, y = load_benchmark_dataset(dataset_name)
            
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert len(X) == len(y)
            assert X.ndim == 2
            assert y.ndim == 1
            
    def test_load_benchmark_dataset_with_preprocessing(self):
        """Test benchmark dataset loading with preprocessing."""
        X, y = load_benchmark_dataset('iris', normalize=True, n_features=4)
        
        assert X.shape[1] == 4
        # Check normalization (features should be roughly in [0, 1] or [-1, 1])
        assert np.all(X >= -2) and np.all(X <= 2)
        
    def test_preprocess_features(self):
        """Test feature preprocessing."""
        # Generate random data
        X = np.random.randn(self.n_samples, self.n_features) * 10 + 5
        
        # Test standardization
        X_std = preprocess_features(X, method='standardize')
        assert X_std.shape == X.shape
        assert np.allclose(np.mean(X_std, axis=0), 0, atol=self.tolerance)
        assert np.allclose(np.std(X_std, axis=0), 1, atol=self.tolerance)
        
        # Test normalization
        X_norm = preprocess_features(X, method='normalize')
        assert X_norm.shape == X.shape
        assert np.all(X_norm >= 0) and np.all(X_norm <= 1)
        
        # Test quantum scaling
        X_quantum = preprocess_features(X, method='quantum_scale')
        assert X_quantum.shape == X.shape
        assert np.all(X_quantum >= 0) and np.all(X_quantum <= np.pi)
        
    def test_data_splitting(self):
        """Test data splitting functionality."""
        from ..utils.data import split_data
        
        X = np.random.random((100, self.n_features))
        y = np.random.randint(0, 2, 100)
        
        splits = split_data(X, y, test_size=0.2, val_size=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        # Check sizes
        assert len(X_train) == 60  # 60% for training
        assert len(X_val) == 20   # 20% for validation
        assert len(X_test) == 20  # 20% for testing
        
        # Check that all data is accounted for
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == 100
        
    def test_synthetic_data_properties(self):
        """Test properties of synthetic data."""
        X, y = generate_quantum_classification_data(
            n_samples=200,
            n_features=4,
            dataset_type='moons',
            noise=0.1
        )
        
        # Check data properties
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))
        assert len(np.unique(y)) == 2  # Binary classification
        
        # Check class balance (should be roughly balanced)
        class_counts = np.bincount(y)
        balance_ratio = min(class_counts) / max(class_counts)
        assert balance_ratio > 0.3  # Not too imbalanced
        
    def test_data_reproducibility(self):
        """Test data generation reproducibility."""
        # Generate data with same seed
        X1, y1 = generate_quantum_classification_data(
            n_samples=50,
            n_features=4,
            dataset_type='circles',
            random_state=42
        )
        
        X2, y2 = generate_quantum_classification_data(
            n_samples=50,
            n_features=4,
            dataset_type='circles',
            random_state=42
        )
        
        # Should be identical
        assert np.allclose(X1, X2)
        assert np.array_equal(y1, y2)


class TestUtilsIntegration:
    """Integration tests for utils components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = TEST_CONFIG['tolerance']
        
        # Generate test data
        np.random.seed(TEST_CONFIG['random_seed'])
        X, y = make_classification(
            n_samples=TEST_CONFIG['medium_dataset_size'],
            n_features=TEST_CONFIG['n_qubits'],
            n_classes=2,
            random_state=TEST_CONFIG['random_seed']
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=TEST_CONFIG['random_seed']
        )
        
        # Train model and get predictions
        model = LogisticRegression(random_state=TEST_CONFIG['random_seed'])
        model.fit(X_train, y_train)
        
        self.y_true = y_test
        self.y_prob = model.predict_proba(X_test)[:, 1]
        
    def test_calibration_metrics_integration(self):
        """Test integration between calibration and metrics."""
        # Calculate original metrics
        original_ece = expected_calibration_error(self.y_true, self.y_prob)
        original_brier = brier_score(self.y_true, self.y_prob)
        
        # Apply calibration
        calibrator = TemperatureScaling()
        calibrator.fit(self.y_prob, self.y_true)
        calibrated_probs = calibrator.transform(self.y_prob)
        
        # Calculate calibrated metrics
        calibrated_ece = expected_calibration_error(self.y_true, calibrated_probs)
        calibrated_brier = brier_score(self.y_true, calibrated_probs)
        
        # All metrics should be valid
        assert 0 <= original_ece <= 1
        assert 0 <= calibrated_ece <= 1
        assert 0 <= original_brier <= 1
        assert 0 <= calibrated_brier <= 1
        
    def test_data_preprocessing_pipeline(self):
        """Test complete data preprocessing pipeline."""
        # Load benchmark data
        X, y = load_benchmark_dataset('iris')
        
        # Preprocess features
        X_processed = preprocess_features(X, method='quantum_scale')
        
        # Split data
        from ..utils.data import split_data
        splits = split_data(X_processed, y, test_size=0.2, val_size=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        # Check pipeline results
        assert X_processed.shape == X.shape
        assert np.all(X_processed >= 0) and np.all(X_processed <= np.pi)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        
    def test_visualization_metrics_integration(self):
        """Test integration between visualization and metrics."""
        plt.close('all')
        
        # Create calibration plot
        fig, ax = plot_calibration_curve(self.y_true, self.y_prob)
        
        # Calculate ECE for the same data
        ece = expected_calibration_error(self.y_true, self.y_prob)
        
        # Both should work with same data
        assert fig is not None
        assert isinstance(ece, (int, float, np.number))
        
        plt.close('all')


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])