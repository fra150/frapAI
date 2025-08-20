"""Utility modules for FPAI framework.

Includes calibration, evaluation metrics, and helper functions.
"""

from .calibration import TemperatureScaling, PlattScaling, IsotonicCalibration, VectorScaling, Calibrator
from .metrics import (
    expected_calibration_error,
    reliability_diagram,
    brier_score,
    negative_log_likelihood,
    fairness_metrics,
    calibration_curve,
    comprehensive_evaluation
)
from .visualization import (
    plot_calibration_curve,
    plot_reliability_diagram,
    plot_quantum_state,
    plot_training_history,
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_model_comparison,
    plot_feature_importance
)
from .data import (
    generate_synthetic_classification,
    generate_quantum_dataset,
    generate_quantum_classification_data,
    load_benchmark_dataset,
    create_train_val_test_split,
    split_data,
    get_dataset_info,
    preprocess_features
)

__all__ = [
    # Calibration
    "Calibrator",
    "TemperatureScaling",
    "PlattScaling", 
    "IsotonicCalibration",
    "VectorScaling",
    
    # Metrics
    "expected_calibration_error",
    "reliability_diagram",
    "brier_score",
    "negative_log_likelihood",
    "fairness_metrics",
    "calibration_curve",
    "comprehensive_evaluation",
    
    # Visualization
    "plot_calibration_curve",
    "plot_reliability_diagram",
    "plot_quantum_state",
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_decision_boundary",
    "plot_model_comparison",
    "plot_feature_importance",
    
    # Data utilities
    "generate_synthetic_classification",
    "generate_quantum_dataset",
    "generate_quantum_classification_data",
    "load_benchmark_dataset",
    "create_train_val_test_split",
    "split_data",
    "preprocess_features",
    "get_dataset_info"
]