"""Test suite for FPAI framework.

This module contains comprehensive tests for all components
of the FPAI quantum machine learning framework.
"""

# Test configuration
TEST_CONFIG = {
    'random_seed': 42,
    'n_qubits': 4,
    'test_shots': 1024,
    'tolerance': 1e-6,
    'small_dataset_size': 100,
    'medium_dataset_size': 500
}

__all__ = [
    'TEST_CONFIG'
]