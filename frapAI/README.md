# FPAI - Fair and Private AI Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](#testing)

FPAI (Fair and Private AI) is a comprehensive quantum machine learning framework that combines quantum computing with fairness and privacy-preserving techniques. The framework provides tools for quantum feature mapping, variational quantum classifiers, quantum kernels, and advanced calibration methods.

## 🚀 Features

### Core Quantum Components
- **Quantum States**: Advanced quantum state representation and manipulation
- **Feature Maps**: Multiple quantum feature encoding strategies
  - Angle Encoding
  - Amplitude Encoding
  - ZZ Feature Map
- **POVM Measurements**: Positive Operator-Valued Measures for quantum measurements

### Machine Learning Models
- **Variational Quantum Classifier (VQC)**: Parameterized quantum circuits for classification
- **Quantum Kernel Methods**: Kernel-based quantum machine learning
- **Hardware-Efficient Ansatz**: Optimized quantum circuit architectures

### Advanced Utilities
- **Calibration Methods**: Multiple probability calibration techniques
  - Temperature Scaling
  - Platt Scaling
  - Isotonic Calibration
  - Vector Scaling
- **Metrics & Evaluation**: Comprehensive evaluation metrics including fairness measures
- **Visualization**: Rich plotting capabilities for quantum states and model performance
- **Data Management**: Quantum dataset generation and preprocessing tools

## 📦 Installation

### Prerequisites
- Python 3.8+
- DeepSeek API key

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/fra150/frapAI.git
cd frapAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your DeepSeek API key:
```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

4. Start the web interface:
```bash
cd web
python server.py  # Frontend server (port 8000)
python api_server.py  # Backend API (port 5000)
```

5. Open your browser and go to `http://localhost:8000`

### 🔧 Configuration

1. Copy the environment file:
```bash
cp web/.env.example web/.env
```

2. Edit `web/.env` and add your API keys:
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# OpenAI API Configuration (fallback)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

3. Start the server:
```bash
cd web
python api_server.py
```

4. Open your browser and go to `http://localhost:5000`

### Optional GPU Support

For GPU acceleration (requires CUDA-capable GPU):

```bash
pip install torch-gpu tensorflow-gpu
```

## 🎯 Quick Start

### Basic Classification Example

```python
from fpai.examples import BasicClassificationExample
from fpai.utils.data import generate_quantum_classification_data

# Generate quantum dataset
X, y = generate_quantum_classification_data(n_samples=200, n_features=4, n_classes=2)

# Create and run classification example
example = BasicClassificationExample(n_qubits=4, n_layers=2)
results = example.run_example(X, y)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Calibration Error: {results['calibration_error']:.3f}")
```

### Quantum Kernel Demo

```python
from fpai.examples import run_quantum_kernel_demo

# Run quantum kernel classification demo
results = run_quantum_kernel_demo(n_samples=100, n_features=4)
print(f"Quantum Kernel Accuracy: {results['accuracy']:.3f}")
```

### Feature Map Comparison

```python
from fpai.examples import FeatureMapComparison

# Compare different feature maps
comparison = FeatureMapComparison()
results = comparison.run_comprehensive_comparison(
    datasets=['iris', 'wine', 'breast_cancer'],
    feature_maps=['angle', 'amplitude', 'zz'],
    model_types=['vqc', 'kernel']
)

# Get recommendations
recommendations = comparison.get_feature_map_recommendations(X, y)
print(f"Recommended feature map: {recommendations['best_feature_map']}")
```

## 📚 Documentation

### Core Modules

#### Quantum States
```python
from fpai.core import QuantumState

# Create quantum state
state = QuantumState(n_qubits=2)
state.initialize_random()

# Measure probabilities
probs = state.get_probabilities()
print(f"State probabilities: {probs}")
```

#### Feature Maps
```python
from fpai.core import AngleEncoding, AmplitudeEncoding, ZZFeatureMap
import numpy as np

# Angle encoding
angle_map = AngleEncoding(n_qubits=4)
encoded = angle_map.encode(np.array([0.1, 0.2, 0.3, 0.4]))

# Amplitude encoding
amp_map = AmplitudeEncoding(n_features=2)
encoded = amp_map.encode(np.array([0.6, 0.8]))

# ZZ Feature Map
zz_map = ZZFeatureMap(n_qubits=4, depth=2)
encoded = zz_map.encode(np.array([0.1, 0.2, 0.3, 0.4]))
```

#### Models
```python
from fpai.models import VQC, QuantumKernel
from fpai.core import AngleEncoding
from fpai.models import HardwareEfficientAnsatz

# Variational Quantum Classifier
feature_map = AngleEncoding(n_qubits=4)
ansatz = HardwareEfficientAnsatz(n_qubits=4, n_layers=2)
vqc = VQC(feature_map=feature_map, ansatz=ansatz)

# Quantum Kernel
kernel = QuantumKernel(feature_map=feature_map)
```

### Calibration
```python
from fpai.utils.calibration import TemperatureScaling, PlattScaling

# Temperature scaling
temp_cal = TemperatureScaling()
temp_cal.fit(logits_val, y_val)
calibrated_probs = temp_cal.predict_proba(logits_test)

# Platt scaling
platt_cal = PlattScaling()
platt_cal.fit(scores_val, y_val)
calibrated_probs = platt_cal.predict_proba(scores_test)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest fpai/tests/

# Run specific test modules
pytest fpai/tests/test_core.py
pytest fpai/tests/test_models.py
pytest fpai/tests/test_utils.py
pytest fpai/tests/test_examples.py

# Run with coverage
pytest fpai/tests/ --cov=fpai --cov-report=html
```

## 📊 Benchmarks

Run comprehensive benchmarks:

```python
from fpai.examples import BenchmarkSuite

# Create benchmark suite
benchmark = BenchmarkSuite()

# Run quantum model benchmarks
results = benchmark.run_quantum_model_benchmark(
    datasets=['iris', 'wine', 'breast_cancer'],
    models=['vqc', 'quantum_kernel'],
    feature_maps=['angle', 'amplitude', 'zz']
)

# Compare with classical models
comparison = benchmark.run_quantum_vs_classical_comparison(
    dataset_name='iris',
    quantum_models=['vqc', 'quantum_kernel'],
    classical_models=['svm', 'random_forest', 'logistic_regression']
)
```

## 🔧 Configuration

### Environment Variables

```bash
# Set quantum backend
export FPAI_BACKEND="qiskit_aer"  # or "pennylane", "cirq"

# Set number of shots for quantum measurements
export FPAI_SHOTS=1024

# Enable GPU acceleration (if available)
export FPAI_USE_GPU=true

# Set random seed for reproducibility
export FPAI_RANDOM_SEED=42
```

### Custom Configuration

```python
from fpai.utils.config import set_config

# Configure global settings
set_config({
    'backend': 'qiskit_aer',
    'shots': 2048,
    'optimization_level': 3,
    'random_seed': 42
})
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run code formatting
black fpai/
isort fpai/

# Run linting
flake8 fpai/
mypy fpai/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [https://fpai.readthedocs.io](https://fpai.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-username/fpai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/fpai/discussions)

## 🙏 Acknowledgments

- [Qiskit](https://qiskit.org/) for quantum computing framework
- [PennyLane](https://pennylane.ai/) for quantum machine learning
- [scikit-learn](https://scikit-learn.org/) for classical ML utilities
- The quantum computing and machine learning communities

## 📈 Roadmap

- [ ] Support for more quantum backends (IonQ, Rigetti)
- [ ] Advanced privacy-preserving techniques (differential privacy)
- [ ] Federated quantum learning capabilities
- [ ] Enhanced fairness metrics and bias detection
- [ ] Quantum advantage benchmarking suite
- [ ] Integration with quantum cloud services

---

**FPAI** - Bridging Quantum Computing, Fairness, and Privacy in AI 🚀