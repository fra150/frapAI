#!/usr/bin/env python3
"""Simple test to check if the framework works without infinite loops."""

import numpy as np
from fpai.core import AngleEncoding, POVM
from fpai.models import VQC, HardwareEfficientAnsatz
from fpai.utils import generate_quantum_dataset

def test_simple_vqc():
    """Test VQC with minimal configuration."""
    print("Testing simple VQC...")
    
    # Generate small dataset
    X, y = generate_quantum_dataset('moons', n_samples=20, n_features=2, noise=0.1)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Create components
    feature_map = AngleEncoding(n_features=2)
    ansatz = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
    povm = POVM.projective(n_outcomes=2, dim=4)
    
    # Create VQC
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        povm=povm,
        shots=100
    )
    
    print("Testing forward pass...")
    # Test single forward pass
    probs = vqc.forward_probs(X[0])
    print(f"Forward pass successful: {probs}")
    
    print("Testing loss computation...")
    # Test loss computation
    loss = vqc.loss(X[:5], y[:5])  # Use only 5 samples
    print(f"Loss computation successful: {loss}")
    
    print("Testing training with minimal epochs...")
    # Test training with very few epochs
    vqc.fit(X, y, epochs=2, learning_rate=0.1, verbose=True)
    print("Training successful!")
    
    # Test prediction
    pred_probs = vqc.predict_proba(X[:3])
    print(f"Prediction successful: {pred_probs.shape}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_simple_vqc()