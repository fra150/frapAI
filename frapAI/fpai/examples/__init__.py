"""Examples module for FPAI framework.

This module contains practical examples and demonstrations
of the FPAI quantum machine learning framework.
"""

from .basic_classification import BasicClassificationExample
from .quantum_kernel_demo import QuantumKernelDemo, run_quantum_kernel_demo, run_kernel_comparison
from .calibration_example import CalibrationExample, run_calibration_demo, run_calibration_suite
from .benchmark_suite import BenchmarkSuite, run_quantum_benchmark, run_quantum_vs_classical_benchmark, run_full_benchmark_suite
from .feature_map_comparison import FeatureMapComparison, run_feature_map_demo, run_comprehensive_feature_map_comparison

__all__ = [
    'BasicClassificationExample',
    'QuantumKernelDemo', 
    'run_quantum_kernel_demo',
    'run_kernel_comparison',
    'CalibrationExample',
    'run_calibration_demo',
    'run_calibration_suite',
    'BenchmarkSuite',
    'run_quantum_benchmark',
    'run_quantum_vs_classical_benchmark', 
    'run_full_benchmark_suite',
    'FeatureMapComparison',
    'run_feature_map_demo',
    'run_comprehensive_feature_map_comparison'
]