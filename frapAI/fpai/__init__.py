"""FPAI - Fuzzy-Probabilistic AI Framework

A quantum-probabilistic framework for machine learning that integrates:
- Quantum states on Hilbert spaces
- Quantum feature maps
- Parametrized quantum circuits
- POVM measurements
- Born rule for calibrated, interpretable predictions

Compatible with NISQ scenarios and classical simulation.
"""

__version__ = "2.0.0"
__author__ = "FPAI Team"
__email__ = "fpai@example.com"

# Core imports
from .core import QuantumState, POVM, FeatureMap
from .models import VQC, QuantumKernel
from .utils import Calibrator

__all__ = [
    "QuantumState",
    "POVM", 
    "FeatureMap",
    "VQC",
    "QuantumKernel",
    "Calibrator"
]