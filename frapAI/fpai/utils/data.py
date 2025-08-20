"""Data utilities for FPAI framework.

Provides functions for generating synthetic datasets, loading benchmarks,
and preprocessing features for quantum machine learning.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

try:
    from sklearn.datasets import fetch_openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False


def generate_quantum_dataset(dataset_type: str = 'classification',
                           n_samples: int = 1000,
                           n_features: int = 4,
                           n_classes: int = 2,
                           noise: float = 0.1,
                           random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate quantum-friendly datasets.
    
    Args:
        dataset_type: Type of dataset ('classification', 'moons', 'circles', 'spiral', 'xor')
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        noise: Noise level
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    if dataset_type == 'classification':
        return generate_synthetic_classification(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes,
            random_state=random_state
        )
    elif dataset_type == 'moons':
        from sklearn.datasets import make_moons
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == 'circles':
        from sklearn.datasets import make_circles
        return make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == 'spiral':
        return _generate_spiral_dataset(n_samples, noise, random_state)
    elif dataset_type == 'xor':
        return _generate_xor_dataset(n_samples, n_features, noise, random_state)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def _generate_spiral_dataset(n_samples: int, noise: float, random_state: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate spiral dataset."""
    if random_state is not None:
        np.random.seed(random_state)
        
    n_per_class = n_samples // 2
    t = np.linspace(0, 2*np.pi, n_per_class)
    
    # Class 0: inner spiral
    r1 = 0.5 * t / (2*np.pi)
    x1 = r1 * np.cos(t) + noise * np.random.randn(n_per_class)
    y1 = r1 * np.sin(t) + noise * np.random.randn(n_per_class)
    
    # Class 1: outer spiral
    r2 = 1.0 * t / (2*np.pi) + 0.5
    x2 = r2 * np.cos(t + np.pi) + noise * np.random.randn(n_per_class)
    y2 = r2 * np.sin(t + np.pi) + noise * np.random.randn(n_per_class)
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    return X, y.astype(int)


def _generate_xor_dataset(n_samples: int, n_features: int, noise: float, random_state: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate XOR-like dataset."""
    if random_state is not None:
        np.random.seed(random_state)
        
    X = np.random.randn(n_samples, n_features)
    # XOR pattern on first two features
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    
    # Add noise
    flip_mask = np.random.rand(n_samples) < noise
    y[flip_mask] = 1 - y[flip_mask]
    
    return X, y


def create_train_val_test_split(X: np.ndarray, y: np.ndarray,
                               train_size: float = 0.6,
                               val_size: float = 0.2,
                               test_size: float = 0.2,
                               random_state: Optional[int] = None,
                               stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/validation/test split.
    
    Args:
        X: Features
        y: Labels
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0")
        
    stratify_y = y if stratify else None
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), 
        random_state=random_state, stratify=stratify_y
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    stratify_temp = y_temp if stratify else None
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio),
        random_state=random_state, stratify=stratify_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: Optional[int] = None, stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple train/test split.
    
    Args:
        X: Features
        y: Labels
        test_size: Test set proportion
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_y = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_y)


def generate_synthetic_classification(n_samples: int = 1000,
                                    n_features: int = 4,
                                    n_classes: int = 2,
                                    n_informative: Optional[int] = None,
                                    n_redundant: int = 0,
                                    n_clusters_per_class: int = 1,
                                    class_sep: float = 1.0,
                                    flip_y: float = 0.01,
                                    random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_clusters_per_class: Number of clusters per class
        class_sep: Class separation factor
        flip_y: Fraction of samples with flipped labels
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    if n_informative is None:
        n_informative = min(n_features, max(1, n_features // 2))
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state
    )
    
    return X, y


def generate_synthetic_regression(n_samples: int = 1000,
                                n_features: int = 4,
                                n_informative: Optional[int] = None,
                                noise: float = 0.1,
                                bias: float = 0.0,
                                random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        noise: Standard deviation of noise
        bias: Bias term
        random_state: Random seed
        
    Returns:
        Tuple of (features, targets)
    """
    if n_informative is None:
        n_informative = min(n_features, max(1, n_features // 2))
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        bias=bias,
        random_state=random_state
    )
    
    return X, y


# Alias for backward compatibility
def generate_quantum_classification_data(n_samples: int = 1000,
                                        n_features: int = 2,
                                        dataset_type: str = 'moons',
                                        noise: float = 0.1,
                                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate quantum classification data (alias for generate_quantum_dataset).
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        dataset_type: Type of dataset ('moons', 'circles', 'blobs', 'xor', 'spiral')
        noise: Noise level
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    return generate_quantum_dataset(
        dataset_type=dataset_type,
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
def generate_quantum_classification_data(dataset_type: str = 'moons',
                                        n_samples: int = 1000,
                                        n_features: int = 2,
                                        noise: float = 0.1,
                                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate quantum classification datasets (alias for generate_quantum_dataset).
    
    Args:
        dataset_type: Type of dataset ('moons', 'circles', 'blobs', 'xor', 'spiral', 'quantum_advantage')
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    if dataset_type == 'quantum_advantage':
        return generate_quantum_advantage_dataset(
            n_samples=n_samples, n_features=n_features, 
            noise=noise, random_state=random_state
        )
    else:
        return generate_quantum_dataset(
            dataset_type=dataset_type, n_samples=n_samples, 
            n_features=n_features, noise=noise, random_state=random_state
        )


def generate_quantum_dataset(dataset_type: str = 'moons',
                            n_samples: int = 1000,
                            n_features: int = 2,
                            noise: float = 0.1,
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate datasets commonly used in quantum ML benchmarks.
    
    Args:
        dataset_type: Type of dataset ('moons', 'circles', 'blobs', 'xor', 'spiral')
        n_samples: Number of samples
        n_features: Number of features (for 'blobs' and 'xor')
        noise: Noise level
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_state)
    
    if dataset_type == 'moons':
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        
    elif dataset_type == 'circles':
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.6, random_state=random_state)
        
    elif dataset_type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=n_features,
                         cluster_std=noise, random_state=random_state)
        
    elif dataset_type == 'xor':
        # Generate XOR-like dataset
        X = np.random.uniform(-1, 1, (n_samples, n_features))
        if n_features >= 2:
            y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        else:
            y = (X[:, 0] > 0).astype(int)
        
        # Add noise
        if noise > 0:
            flip_mask = np.random.random(n_samples) < noise
            y[flip_mask] = 1 - y[flip_mask]
            
    elif dataset_type == 'spiral':
        # Generate spiral dataset
        n_per_class = n_samples // 2
        theta = np.linspace(0, 4*np.pi, n_per_class)
        r = np.linspace(0.1, 1, n_per_class)
        
        # Class 0
        X0 = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta)
        ])
        
        # Class 1 (shifted by pi)
        X1 = np.column_stack([
            r * np.cos(theta + np.pi),
            r * np.sin(theta + np.pi)
        ])
        
        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
        
        # Add noise
        if noise > 0:
            X += np.random.normal(0, noise, X.shape)
        
        # Pad with zeros if more features needed
        if n_features > 2:
            padding = np.zeros((n_samples, n_features - 2))
            X = np.hstack([X, padding])
            
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return X, y


def load_benchmark_dataset(name: str, 
                          test_size: float = 0.2,
                          random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load benchmark datasets for quantum ML.
    
    Args:
        name: Dataset name ('iris', 'wine', 'breast_cancer', 'digits')
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Dictionary with train/test splits
    """
    if name == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        
    elif name == 'wine':
        from sklearn.datasets import load_wine
        data = load_wine()
        
    elif name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        
    elif name == 'digits':
        from sklearn.datasets import load_digits
        data = load_digits()
        
    elif name == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        
    else:
        raise ValueError(f"Unknown benchmark dataset: {name}")
    
    X, y = data.data, data.target
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': data.feature_names if hasattr(data, 'feature_names') else None,
        'target_names': data.target_names if hasattr(data, 'target_names') else None
    }


def preprocess_features(X: np.ndarray,
                       scaler_type: str = 'standard',
                       feature_range: Tuple[float, float] = (0, 1),
                       fit_scaler: bool = True,
                       scaler: Optional[Any] = None) -> Tuple[np.ndarray, Any]:
    """Preprocess features for quantum ML.
    
    Args:
        X: Input features
        scaler_type: Type of scaler ('standard', 'minmax', 'robust', 'none')
        feature_range: Range for MinMaxScaler
        fit_scaler: Whether to fit the scaler
        scaler: Pre-fitted scaler to use
        
    Returns:
        Tuple of (scaled_features, fitted_scaler)
    """
    if scaler_type == 'none':
        return X, None
    
    if scaler is None:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def quantum_feature_preprocessing(X: np.ndarray,
                                encoding_type: str = 'angle',
                                n_qubits: Optional[int] = None,
                                normalize: bool = True) -> np.ndarray:
    """Preprocess features specifically for quantum encoding.
    
    Args:
        X: Input features
        encoding_type: Type of encoding ('angle', 'amplitude')
        n_qubits: Number of qubits (for amplitude encoding)
        normalize: Whether to normalize features
        
    Returns:
        Preprocessed features
    """
    X_processed = X.copy()
    
    if normalize:
        # Normalize to [0, 1] for angle encoding or [-1, 1] for amplitude
        if encoding_type == 'angle':
            X_processed = (X_processed - X_processed.min(axis=0)) / (X_processed.max(axis=0) - X_processed.min(axis=0))
            # Scale to [0, π] for angle encoding
            X_processed = X_processed * np.pi
        elif encoding_type == 'amplitude':
            # Normalize to unit vector for amplitude encoding
            if n_qubits is not None:
                # Pad or truncate to fit 2^n_qubits dimensions
                target_dim = 2 ** n_qubits
                if X_processed.shape[1] < target_dim:
                    padding = np.zeros((X_processed.shape[0], target_dim - X_processed.shape[1]))
                    X_processed = np.hstack([X_processed, padding])
                elif X_processed.shape[1] > target_dim:
                    X_processed = X_processed[:, :target_dim]
            
            # Normalize each sample to unit vector
            norms = np.linalg.norm(X_processed, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            X_processed = X_processed / norms
    
    return X_processed


def create_train_val_test_split(X: np.ndarray, y: np.ndarray,
                               train_size: float = 0.6,
                               val_size: float = 0.2,
                               test_size: float = 0.2,
                               random_state: Optional[int] = None,
                               stratify: bool = True) -> Dict[str, np.ndarray]:
    """Create train/validation/test splits.
    
    Args:
        X: Features
        y: Labels
        train_size: Fraction for training
        val_size: Fraction for validation
        test_size: Fraction for testing
        random_state: Random seed
        stratify: Whether to stratify splits
        
    Returns:
        Dictionary with splits
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    stratify_y = y if stratify and len(np.unique(y)) > 1 else None
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state, stratify=stratify_y
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    stratify_temp = y_temp if stratify and len(np.unique(y_temp)) > 1 else None
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=stratify_temp
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def add_noise_to_data(X: np.ndarray, y: np.ndarray,
                     feature_noise: float = 0.0,
                     label_noise: float = 0.0,
                     random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Add noise to features and labels.
    
    Args:
        X: Features
        y: Labels
        feature_noise: Standard deviation of Gaussian noise for features
        label_noise: Probability of flipping labels
        random_state: Random seed
        
    Returns:
        Tuple of (noisy_features, noisy_labels)
    """
    np.random.seed(random_state)
    
    X_noisy = X.copy()
    y_noisy = y.copy()
    
    # Add feature noise
    if feature_noise > 0:
        noise = np.random.normal(0, feature_noise, X.shape)
        X_noisy = X + noise
    
    # Add label noise
    if label_noise > 0 and len(np.unique(y)) == 2:  # Binary classification
        flip_mask = np.random.random(len(y)) < label_noise
        y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
    
    return X_noisy, y_noisy


def generate_quantum_advantage_dataset(n_samples: int = 1000,
                                     n_features: int = 4,
                                     entanglement_strength: float = 1.0,
                                     noise: float = 0.1,
                                     random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dataset designed to showcase quantum advantage.
    
    Creates a dataset where quantum feature maps should outperform classical ones
    due to the entangled structure of the decision boundary.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features (should be even)
        entanglement_strength: Strength of entanglement in decision boundary
        noise: Noise level
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_state)
    
    # Ensure even number of features for pairing
    if n_features % 2 != 0:
        n_features += 1
    
    # Generate random features
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    
    # Create entangled decision boundary
    # Pair features and create XOR-like interactions
    y = np.zeros(n_samples)
    
    for i in range(0, n_features, 2):
        if i + 1 < n_features:
            # Create entangled interaction between feature pairs
            interaction = np.cos(entanglement_strength * X[:, i]) * np.cos(entanglement_strength * X[:, i + 1])
            y += interaction
    
    # Convert to binary labels
    y = (y > 0).astype(int)
    
    # Add noise
    if noise > 0:
        flip_mask = np.random.random(n_samples) < noise
        y[flip_mask] = 1 - y[flip_mask]
    
    return X, y


def create_quantum_dataset_suite() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create a suite of datasets for quantum ML benchmarking.
    
    Returns:
        Dictionary of dataset name to (X, y) tuples
    """
    datasets = {}
    
    # Standard quantum ML datasets
    datasets['moons'] = generate_quantum_dataset('moons', n_samples=500, noise=0.1, random_state=42)
    datasets['circles'] = generate_quantum_dataset('circles', n_samples=500, noise=0.1, random_state=42)
    datasets['spiral'] = generate_quantum_dataset('spiral', n_samples=500, noise=0.05, random_state=42)
    datasets['xor'] = generate_quantum_dataset('xor', n_samples=500, n_features=2, noise=0.1, random_state=42)
    
    # Quantum advantage dataset
    datasets['quantum_advantage'] = generate_quantum_advantage_dataset(
        n_samples=500, n_features=4, entanglement_strength=2.0, noise=0.1, random_state=42
    )
    
    # High-dimensional dataset
    datasets['high_dim'] = generate_synthetic_classification(
        n_samples=500, n_features=8, n_classes=2, class_sep=0.8, random_state=42
    )
    
    return datasets


def get_dataset_info(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Get information about a dataset.
    
    Args:
        X: Features
        y: Labels
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': {str(cls): np.sum(y == cls) for cls in np.unique(y)},
        'feature_ranges': {
            'min': X.min(axis=0).tolist(),
            'max': X.max(axis=0).tolist(),
            'mean': X.mean(axis=0).tolist(),
            'std': X.std(axis=0).tolist()
        },
        'class_balance': np.min(np.bincount(y.astype(int))) / np.max(np.bincount(y.astype(int))),
        'has_missing_values': np.any(np.isnan(X)),
        'sparsity': np.sum(X == 0) / X.size
    }
    
    return info