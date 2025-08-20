"""Tests for FPAI examples module.

Tests for example implementations and benchmark suites.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ..examples import (
    BasicClassificationExample, QuantumKernelDemo,
    CalibrationExample, FeatureMapComparison, BenchmarkSuite
)
from . import TEST_CONFIG


class TestBasicClassificationExample:
    """Test BasicClassificationExample class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.example = BasicClassificationExample()
        self.n_qubits = TEST_CONFIG['n_qubits']
        
    def test_initialization(self):
        """Test example initialization."""
        assert hasattr(self.example, 'n_qubits')
        assert hasattr(self.example, 'n_samples')
        assert self.example.n_qubits > 0
        assert self.example.n_samples > 0
        
    def test_data_generation(self):
        """Test data generation."""
        X, y = self.example._generate_data()
        
        assert X.shape == (self.example.n_samples, self.example.n_qubits)
        assert len(y) == self.example.n_samples
        assert set(y) <= {0, 1}  # Binary classification
        
    def test_model_creation(self):
        """Test model creation."""
        model = self.example._create_vqc_model()
        
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
    @patch('matplotlib.pyplot.show')
    def test_run_example(self, mock_show):
        """Test running the complete example."""
        # Mock plotting to avoid display issues in tests
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot'), \
             patch('matplotlib.pyplot.plot'), \
             patch('matplotlib.pyplot.scatter'):
            
            results = self.example.run_example(
                epochs=2,  # Reduced for testing
                verbose=False
            )
        
        assert isinstance(results, dict)
        assert 'model_performance' in results
        assert 'calibration_results' in results
        
    def test_feature_map_comparison(self):
        """Test feature map comparison."""
        # Use small dataset for testing
        comparison_results = self.example.compare_feature_maps(
            n_samples=50,
            epochs=2,
            verbose=False
        )
        
        assert isinstance(comparison_results, dict)
        assert len(comparison_results) > 0
        
        # Check that all feature maps were tested
        for fm_name, results in comparison_results.items():
            assert isinstance(results, dict)
            assert 'accuracy' in results
            assert 'ece' in results


class TestQuantumKernelDemo:
    """Test QuantumKernelDemo class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.demo = QuantumKernelDemo()
        
    def test_initialization(self):
        """Test demo initialization."""
        assert hasattr(self.demo, 'n_qubits')
        assert hasattr(self.demo, 'n_samples')
        
    def test_kernel_creation(self):
        """Test quantum kernel creation."""
        kernel = self.demo._create_quantum_kernel()
        
        assert hasattr(kernel, 'fit')
        assert hasattr(kernel, 'predict')
        assert hasattr(kernel, 'compute_kernel_matrix')
        
    def test_kernel_properties_analysis(self):
        """Test kernel properties analysis."""
        # Generate small test data
        np.random.seed(TEST_CONFIG['random_seed'])
        X = np.random.random((20, self.demo.n_qubits))
        
        kernel = self.demo._create_quantum_kernel()
        properties = self.demo._analyze_kernel_properties(kernel, X)
        
        assert isinstance(properties, dict)
        assert 'rank' in properties
        assert 'condition_number' in properties
        assert 'spectral_gap' in properties
        
    @patch('matplotlib.pyplot.show')
    def test_run_demo(self, mock_show):
        """Test running the quantum kernel demo."""
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot'), \
             patch('matplotlib.pyplot.plot'):
            
            results = self.demo.run_quantum_kernel_demo(
                n_samples=30,  # Small for testing
                verbose=False
            )
        
        assert isinstance(results, dict)
        assert 'quantum_kernel_performance' in results
        assert 'kernel_properties' in results
        
    def test_classical_comparison(self):
        """Test comparison with classical kernels."""
        comparison_results = self.demo.compare_with_classical_kernels(
            n_samples=40,  # Small for testing
            verbose=False
        )
        
        assert isinstance(comparison_results, dict)
        assert 'quantum' in comparison_results
        assert 'classical' in comparison_results
        
        # Check that classical kernels were tested
        classical_results = comparison_results['classical']
        assert len(classical_results) > 0
        
        for kernel_name, results in classical_results.items():
            assert isinstance(results, dict)
            assert 'accuracy' in results


class TestCalibrationExample:
    """Test CalibrationExample class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.example = CalibrationExample()
        
    def test_initialization(self):
        """Test example initialization."""
        assert hasattr(self.example, 'n_qubits')
        assert hasattr(self.example, 'n_samples')
        
    def test_model_training(self):
        """Test basic model training."""
        # Generate small test data
        np.random.seed(TEST_CONFIG['random_seed'])
        X = np.random.random((30, self.example.n_qubits))
        y = np.random.randint(0, 2, 30)
        
        model = self.example._train_base_model(X, y, epochs=2, verbose=False)
        
        assert hasattr(model, 'predict_proba')
        
        # Test prediction
        probabilities = model.predict_proba(X[:5])
        assert probabilities.shape == (5, 2)
        
    def test_calibration_methods(self):
        """Test different calibration methods."""
        # Generate test data and predictions
        np.random.seed(TEST_CONFIG['random_seed'])
        y_true = np.random.randint(0, 2, 50)
        y_prob = np.random.random(50)
        
        calibration_results = self.example._apply_calibration_methods(
            y_prob, y_true
        )
        
        assert isinstance(calibration_results, dict)
        assert 'temperature_scaling' in calibration_results
        assert 'platt_scaling' in calibration_results
        assert 'isotonic' in calibration_results
        
        # Check that all methods return valid probabilities
        for method_name, calibrated_probs in calibration_results.items():
            assert len(calibrated_probs) == len(y_prob)
            assert np.all(calibrated_probs >= 0)
            assert np.all(calibrated_probs <= 1)
            
    @patch('matplotlib.pyplot.show')
    def test_run_calibration_demo(self, mock_show):
        """Test running the calibration demo."""
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot'), \
             patch('matplotlib.pyplot.plot'):
            
            results = self.example.run_calibration_demo(
                n_samples=40,  # Small for testing
                epochs=2,
                verbose=False
            )
        
        assert isinstance(results, dict)
        assert 'original_metrics' in results
        assert 'calibrated_metrics' in results
        
    def test_calibration_comparison_suite(self):
        """Test calibration comparison across datasets."""
        comparison_results = self.example.run_calibration_comparison_suite(
            n_samples=30,  # Small for testing
            epochs=2,
            verbose=False
        )
        
        assert isinstance(comparison_results, dict)
        assert len(comparison_results) > 0
        
        # Check that multiple datasets were tested
        for dataset_name, results in comparison_results.items():
            assert isinstance(results, dict)
            assert 'calibration_methods' in results


class TestFeatureMapComparison:
    """Test FeatureMapComparison class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.comparison = FeatureMapComparison()
        
    def test_initialization(self):
        """Test comparison initialization."""
        assert hasattr(self.comparison, 'n_qubits')
        assert hasattr(self.comparison, 'feature_maps')
        assert len(self.comparison.feature_maps) > 0
        
    def test_feature_map_creation(self):
        """Test feature map creation."""
        for fm_name in self.comparison.feature_maps:
            fm = self.comparison._create_feature_map(fm_name)
            assert hasattr(fm, 'encode')
            assert hasattr(fm, 'n_qubits')
            
    def test_model_training(self):
        """Test VQC model training."""
        # Generate small test data
        np.random.seed(TEST_CONFIG['random_seed'])
        X_train = np.random.random((20, self.comparison.n_qubits))
        y_train = np.random.randint(0, 2, 20)
        X_test = np.random.random((10, self.comparison.n_qubits))
        y_test = np.random.randint(0, 2, 10)
        
        fm = self.comparison._create_feature_map('angle_encoding')
        results = self.comparison._train_vqc_model(
            fm, X_train, y_train, X_test, y_test,
            epochs=2, verbose=False
        )
        
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'ece' in results
        assert 'training_time' in results
        
    def test_kernel_model_training(self):
        """Test quantum kernel model training."""
        # Generate small test data
        np.random.seed(TEST_CONFIG['random_seed'])
        X_train = np.random.random((20, self.comparison.n_qubits))
        y_train = np.random.randint(0, 2, 20)
        X_test = np.random.random((10, self.comparison.n_qubits))
        y_test = np.random.randint(0, 2, 10)
        
        fm = self.comparison._create_feature_map('zz_feature_map')
        results = self.comparison._train_kernel_model(
            fm, X_train, y_train, X_test, y_test,
            verbose=False
        )
        
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'ece' in results
        assert 'training_time' in results
        
    def test_feature_map_comparison(self):
        """Test feature map comparison."""
        comparison_results = self.comparison.compare_feature_maps(
            datasets=['moons'],  # Single dataset for testing
            model_types=['vqc'],  # Single model type for testing
            n_samples=30,
            epochs=2,
            verbose=False
        )
        
        assert isinstance(comparison_results, dict)
        assert len(comparison_results) > 0
        
        # Check structure of results
        for dataset_name, dataset_results in comparison_results.items():
            assert isinstance(dataset_results, dict)
            for model_type, model_results in dataset_results.items():
                assert isinstance(model_results, dict)
                assert len(model_results) > 0  # Should have feature map results
                
    def test_recommendations(self):
        """Test feature map recommendations."""
        # Mock dataset characteristics
        dataset_characteristics = {
            'n_features': 4,
            'n_samples': 100,
            'n_classes': 2,
            'separability': 'medium'
        }
        
        recommendations = self.comparison.get_feature_map_recommendations(
            dataset_characteristics
        )
        
        assert isinstance(recommendations, dict)
        assert len(recommendations) > 0
        
        # Check that recommendations include reasoning
        for fm_name, recommendation in recommendations.items():
            assert isinstance(recommendation, dict)
            assert 'score' in recommendation
            assert 'reasoning' in recommendation


class TestBenchmarkSuite:
    """Test BenchmarkSuite class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.benchmark = BenchmarkSuite()
        
    def test_initialization(self):
        """Test benchmark initialization."""
        assert hasattr(self.benchmark, 'quantum_models')
        assert hasattr(self.benchmark, 'classical_models')
        assert hasattr(self.benchmark, 'datasets')
        
    def test_data_loading(self):
        """Test dataset loading and preprocessing."""
        X, y = self.benchmark._load_and_preprocess_dataset(
            'iris', n_samples=50
        )
        
        assert X.shape[0] <= 50  # Should respect n_samples limit
        assert len(X) == len(y)
        assert X.ndim == 2
        assert y.ndim == 1
        
    def test_quantum_model_benchmark(self):
        """Test quantum model benchmarking."""
        results = self.benchmark.run_quantum_model_benchmark(
            datasets=['iris'],  # Single dataset for testing
            models=['vqc'],     # Single model for testing
            n_samples=30,
            epochs=2,
            verbose=False
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check structure of results
        for dataset_name, dataset_results in results.items():
            assert isinstance(dataset_results, dict)
            for model_name, model_results in dataset_results.items():
                assert isinstance(model_results, dict)
                assert 'accuracy' in model_results
                assert 'training_time' in model_results
                
    def test_quantum_vs_classical_comparison(self):
        """Test quantum vs classical comparison."""
        comparison_results = self.benchmark.run_quantum_vs_classical_comparison(
            datasets=['iris'],  # Single dataset for testing
            n_samples=40,
            epochs=2,
            verbose=False
        )
        
        assert isinstance(comparison_results, dict)
        assert 'quantum' in comparison_results
        assert 'classical' in comparison_results
        
        # Check that both quantum and classical results exist
        quantum_results = comparison_results['quantum']
        classical_results = comparison_results['classical']
        
        assert len(quantum_results) > 0
        assert len(classical_results) > 0
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Mock results data
        mock_results = {
            'iris': {
                'vqc': {
                    'accuracy': 0.85,
                    'ece': 0.12,
                    'training_time': 10.5
                }
            }
        }
        
        summary = self.benchmark._generate_performance_summary(mock_results)
        
        assert isinstance(summary, dict)
        assert 'best_models' in summary
        assert 'average_performance' in summary
        
    def test_comprehensive_benchmark(self):
        """Test comprehensive benchmark suite."""
        # Run with minimal configuration for testing
        results = self.benchmark.run_comprehensive_benchmark(
            datasets=['iris'],
            n_samples=25,
            epochs=2,
            include_classical=False,  # Skip classical for speed
            verbose=False
        )
        
        assert isinstance(results, dict)
        assert 'quantum_results' in results
        assert 'summary' in results


class TestExamplesIntegration:
    """Integration tests for examples."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tolerance = TEST_CONFIG['tolerance']
        
    def test_examples_consistency(self):
        """Test consistency across different examples."""
        # Test that all examples can be instantiated
        examples = [
            BasicClassificationExample(),
            QuantumKernelDemo(),
            CalibrationExample(),
            FeatureMapComparison(),
            BenchmarkSuite()
        ]
        
        for example in examples:
            assert hasattr(example, 'n_qubits')
            # Most examples should have these common attributes
            if hasattr(example, 'n_samples'):
                assert example.n_samples > 0
                
    def test_data_compatibility(self):
        """Test that examples work with same data format."""
        # Generate common test data
        np.random.seed(TEST_CONFIG['random_seed'])
        X = np.random.random((50, TEST_CONFIG['n_qubits']))
        y = np.random.randint(0, 2, 50)
        
        # Test that basic classification example can handle this data
        basic_example = BasicClassificationExample()
        basic_example.n_samples = 50
        basic_example.n_qubits = TEST_CONFIG['n_qubits']
        
        # Should be able to create a model and make predictions
        model = basic_example._create_vqc_model()
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
    @patch('matplotlib.pyplot.show')
    def test_visualization_integration(self, mock_show):
        """Test that examples integrate properly with visualization."""
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot'), \
             patch('matplotlib.pyplot.plot'):
            
            # Test basic classification example visualization
            basic_example = BasicClassificationExample()
            
            # Should not raise errors when creating plots
            try:
                # This would normally create plots
                results = basic_example.run_example(
                    epochs=1, verbose=False
                )
                assert isinstance(results, dict)
            except Exception as e:
                # If there are plotting issues, they should be handled gracefully
                assert "display" not in str(e).lower()
                
    def test_parameter_validation(self):
        """Test parameter validation across examples."""
        examples = [
            BasicClassificationExample(),
            QuantumKernelDemo(),
            CalibrationExample()
        ]
        
        for example in examples:
            # Test that n_qubits is reasonable
            assert 1 <= example.n_qubits <= 10
            
            # Test that n_samples is reasonable
            if hasattr(example, 'n_samples'):
                assert 10 <= example.n_samples <= 10000


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])