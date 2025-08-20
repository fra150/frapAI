"""Quantum kernel demonstration using FPAI framework.

Shows how to use quantum kernels for classification and compares
them with classical kernels.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import time

from ..models import QuantumKernel
from ..core import AngleEncoding, ZZFeatureMap
from ..utils import (
    generate_quantum_dataset, preprocess_features, create_train_val_test_split,
    comprehensive_evaluation, plot_calibration_curve
)

try:
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class QuantumKernelDemo:
    """Demonstration of quantum kernel methods."""
    
    def __init__(self, n_qubits: int = 4, random_state: int = 42):
        """
        Initialize the demo.
        
        Args:
            n_qubits: Number of qubits
            random_state: Random seed
        """
        self.n_qubits = n_qubits
        self.random_state = random_state
        self.results = {}

    def _prepare_data(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Generate, preprocess, and split the dataset."""
        X, y = generate_quantum_dataset(
            dataset_type=dataset_name,
            n_samples=400,
            n_features=self.n_qubits,
            noise=0.1,
            random_state=self.random_state
        )
        X_scaled, _ = preprocess_features(X, scaler_type='minmax')
        return create_train_val_test_split(
            X_scaled, y, train_size=0.7, val_size=0.0, test_size=0.3,
            random_state=self.random_state
        )

    def run_quantum_kernel_demo(self, dataset_name: str = 'moons',
                               feature_map_type: str = 'angle',
                               kernel_type: str = 'fidelity',
                               shots: int = 1024,
                               verbose: bool = True) -> Dict[str, Any]:
        """
        Run quantum kernel classification demo.
        
        Args:
            dataset_name: Dataset to use
            feature_map_type: Feature map type ('angle', 'zz')
            kernel_type: Kernel type ('fidelity', 'projected')
            shots: Number of measurement shots
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results
        """
        if verbose:
            print(f"\n=== Quantum Kernel Demo ===")
            print(f"Dataset: {dataset_name}, Feature Map: {feature_map_type}, Kernel: {kernel_type}")
            print(f"Qubits: {self.n_qubits}, Shots: {shots}")
            print("=" * 40)

        # Step 1: Prepare data
        if verbose:
            print("\n1. Preparing data...")
        splits = self._prepare_data(dataset_name)

        # Step 2: Create feature map
        if verbose:
            print(f"2. Creating {feature_map_type} feature map...")
        
        if feature_map_type == 'angle':
            feature_map = AngleEncoding(n_features=self.n_qubits)
        elif feature_map_type == 'zz':
            feature_map = ZZFeatureMap(n_qubits=self.n_qubits, depth=2)
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")
        
        # Step 3: Create and train quantum kernel
        if verbose:
            print("3. Training quantum kernel classifier...")
        
        qkernel = QuantumKernel(
            feature_map=feature_map,
            kernel_type=kernel_type,
            shots=shots,
            random_state=self.random_state
        )
        
        start_time = time.time()
        qkernel.fit(splits['X_train'], splits['y_train'])
        training_time = time.time() - start_time
        
        # Step 4: Make predictions
        if verbose:
            print("4. Making predictions...")
        
        y_pred_proba = qkernel.predict_proba(splits['X_test'])
        y_pred = qkernel.predict(splits['X_test'])
        
        # Step 5: Evaluate results
        if verbose:
            print("5. Evaluating results...")
        
        results = comprehensive_evaluation(
            splits['y_test'], y_pred_proba, y_pred
        )
        
        # Step 6: Analyze kernel properties
        if verbose:
            print("6. Analyzing kernel properties...")
        
        kernel_analysis = self._analyze_kernel_properties(
            qkernel, splits['X_train'], splits['y_train']
        )
        
        # Step 7: Compile results
        experiment_results = {
            'dataset': dataset_name,
            'feature_map': feature_map_type,
            'kernel_type': kernel_type,
            'n_qubits': self.n_qubits,
            'shots': shots,
            'training_time': training_time,
            'n_train_samples': len(splits['X_train']),
            'n_test_samples': len(splits['X_test']),
            'results': results,
            'kernel_analysis': kernel_analysis,
            'predictions': {
                'y_true': splits['y_test'],
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            },
            'model_info': {
                'n_support_vectors': len(qkernel.support_vectors_),
                'support_vector_indices': qkernel.support_vector_indices_,
                'dual_coefficients': qkernel.dual_coef_
            }
        }
        
        # Store results
        key = f"{dataset_name}_{feature_map_type}_{kernel_type}"
        self.results[key] = experiment_results
        
        # Print summary
        if verbose:
            self._print_results_summary(experiment_results)
        
        return experiment_results
    
    def _analyze_kernel_properties(self, qkernel, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Analyze properties of the quantum kernel."""
        # Compute kernel matrix
        K_train = qkernel.compute_kernel_matrix(X_train, X_train)
        
        # Kernel alignment
        alignment = qkernel.kernel_alignment(X_train, y_train)
        
        # Kernel matrix analysis
        matrix_analysis = qkernel.kernel_matrix_analysis(K_train)
        
        return {
            'kernel_alignment': alignment,
            'matrix_analysis': matrix_analysis,
            'kernel_matrix_shape': K_train.shape,
            'kernel_matrix_stats': {
                'mean': np.mean(K_train),
                'std': np.std(K_train),
                'min': np.min(K_train),
                'max': np.max(K_train)
            }
        }
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of the results."""
        print("\n" + "="*50)
        print("QUANTUM KERNEL RESULTS SUMMARY")
        print("="*50)
        
        res = results['results']
        kernel_analysis = results['kernel_analysis']
        model_info = results['model_info']
        
        print(f"Training time: {results['training_time']:.2f}s")
        print(f"Training samples: {results['n_train_samples']}")
        print(f"Test samples: {results['n_test_samples']}")
        print(f"Support vectors: {model_info['n_support_vectors']}")
        
        print("\nPerformance:")
        print(f"  Accuracy: {res['accuracy']:.3f}")
        print(f"  AUROC: {res['auroc']:.3f}")
        print(f"  ECE: {res['ece']:.3f}")
        print(f"  Brier Score: {res['brier_score']:.3f}")
        
        print("\nKernel Analysis:")
        print(f"  Kernel Alignment: {kernel_analysis['kernel_alignment']:.3f}")
        
        matrix_stats = kernel_analysis['kernel_matrix_stats']
        print(f"  Kernel Matrix Mean: {matrix_stats['mean']:.3f}")
        print(f"  Kernel Matrix Std: {matrix_stats['std']:.3f}")
        
        matrix_analysis = kernel_analysis['matrix_analysis']
        print(f"  Condition Number: {matrix_analysis['condition_number']:.2e}")
        print(f"  Rank: {matrix_analysis['rank']}")
        print(f"  Trace: {matrix_analysis['trace']:.3f}")
    
    def compare_with_classical_kernels(self, dataset_name: str = 'moons',
                                     classical_kernels: List[str] = ['rbf', 'poly', 'linear'],
                                     verbose: bool = True) -> Dict[str, Any]:
        """Compare quantum kernel with classical kernels.
        
        Args:
            dataset_name: Dataset to use
            classical_kernels: List of classical kernel types
            verbose: Whether to print progress
            
        Returns:
            Dictionary with comparison results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for classical kernel comparison")
        
        if verbose:
            print(f"\n=== Quantum vs Classical Kernel Comparison ===")
            print(f"Dataset: {dataset_name}")

        # Prepare data once
        splits = self._prepare_data(dataset_name)

        # Run quantum kernel
        quantum_result = self.run_quantum_kernel_demo(
            dataset_name=dataset_name,
            feature_map_type='angle',
            kernel_type='fidelity',
            verbose=False
        )
        
        # Run classical kernels
        classical_results = {}
        
        for kernel in classical_kernels:
            if verbose:
                print(f"\nTesting {kernel} kernel...")
            
            start_time = time.time()
            
            # Train classical SVM
            svm = SVC(kernel=kernel, probability=True, random_state=self.random_state)
            svm.fit(splits['X_train'], splits['y_train'])
            
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred_proba = svm.predict_proba(splits['X_test'])
            y_pred = svm.predict(splits['X_test'])
            
            # Evaluate
            results = comprehensive_evaluation(
                splits['y_test'], y_pred_proba, y_pred
            )
            
            classical_results[kernel] = {
                'results': results,
                'training_time': training_time,
                'n_support_vectors': len(svm.support_vectors_)
            }
            
            if verbose:
                print(f"  Accuracy: {results['accuracy']:.3f}, Time: {training_time:.2f}s")
        
        # Compile comparison
        comparison_results = {
            'quantum': quantum_result,
            'classical': classical_results,
            'dataset': dataset_name
        }
        
        # Print comparison summary
        if verbose:
            self._print_comparison_summary(comparison_results)
        
        return comparison_results
    
    def _print_comparison_summary(self, comparison_results: Dict[str, Any]):
        """Print comparison summary."""
        print("\n" + "="*70)
        print("QUANTUM vs CLASSICAL KERNEL COMPARISON")
        print("="*70)
        
        quantum = comparison_results['quantum']
        classical = comparison_results['classical']
        
        print(f"{'Kernel':<15} {'Accuracy':<10} {'AUROC':<8} {'ECE':<8} {'Time(s)':<8} {'SVs':<6}")
        print("-" * 70)
        
        # Quantum result
        q_res = quantum['results']
        print(f"{'Quantum':<15} {q_res['accuracy']:<10.3f} {q_res['auroc']:<8.3f} "
              f"{q_res['ece']:<8.3f} {quantum['training_time']:<8.1f} "
              f"{quantum['model_info']['n_support_vectors']:<6}")
        
        # Classical results
        for kernel_name, result in classical.items():
            c_res = result['results']
            print(f"{kernel_name:<15} {c_res['accuracy']:<10.3f} {c_res['auroc']:<8.3f} "
                  f"{c_res['ece']:<8.3f} {result['training_time']:<8.1f} "
                  f"{result['n_support_vectors']:<6}")
    
    def visualize_kernel_matrix(self, experiment_key: str, save_plot: bool = False, plot_dir: str = "plots"):
        """Visualize the kernel matrix.
        
        Args:
            experiment_key: Key of experiment to visualize
            save_plot: Whether to save plot
            plot_dir: Directory to save plot
        """
        if experiment_key not in self.results:
            raise ValueError(f"No results found for experiment: {experiment_key}")
        
        # This would require recreating the quantum kernel and computing the matrix
        # For now, we'll just print the analysis
        result = self.results[experiment_key]
        kernel_analysis = result['kernel_analysis']
        
        print(f"\nKernel Matrix Analysis for {experiment_key}:")
        print(f"Shape: {kernel_analysis['kernel_matrix_shape']}")
        print(f"Alignment: {kernel_analysis['kernel_alignment']:.3f}")
        
        matrix_analysis = kernel_analysis['matrix_analysis']
        for key, value in matrix_analysis.items():
            print(f"{key}: {value}")
    
    def run_kernel_comparison_suite(self, datasets: List[str] = ['moons', 'circles'],
                                  feature_maps: List[str] = ['angle', 'zz'],
                                  verbose: bool = True) -> Dict[str, Any]:
        """Run comprehensive kernel comparison suite.
        
        Args:
            datasets: List of datasets to test
            feature_maps: List of feature maps to test
            verbose: Whether to print progress
            
        Returns:
            Dictionary with all results
        """
        if verbose:
            print("\n" + "="*60)
            print("QUANTUM KERNEL COMPARISON SUITE")
            print("="*60)
        
        suite_results = {}
        
        for dataset in datasets:
            for feature_map in feature_maps:
                if verbose:
                    print(f"\nTesting {feature_map} on {dataset}...")
                
                try:
                    result = self.run_quantum_kernel_demo(
                        dataset_name=dataset,
                        feature_map_type=feature_map,
                        kernel_type='fidelity',
                        verbose=False
                    )
                    
                    key = f"{dataset}_{feature_map}"
                    suite_results[key] = result
                    
                    if verbose:
                        acc = result['results']['accuracy']
                        ece = result['results']['ece']
                        time_taken = result['training_time']
                        print(f"  Accuracy: {acc:.3f}, ECE: {ece:.3f}, Time: {time_taken:.1f}s")
                        
                except Exception as e:
                    if verbose:
                        print(f"  Error: {str(e)}")
                    suite_results[f"{dataset}_{feature_map}"] = None
        
        return suite_results


def run_quantum_kernel_demo():
    """Run a basic quantum kernel demo."""
    demo = QuantumKernelDemo(n_qubits=4)
    
    # Run single experiment
    result = demo.run_quantum_kernel_demo(
        dataset_name='moons',
        feature_map_type='angle',
        kernel_type='fidelity',
        shots=1024,
        verbose=True
    )
    
    return result


def run_kernel_comparison():
    """Run quantum vs classical kernel comparison."""
    demo = QuantumKernelDemo(n_qubits=4)
    
    try:
        comparison = demo.compare_with_classical_kernels(
            dataset_name='moons',
            classical_kernels=['rbf', 'poly'],
            verbose=True
        )
        return comparison
    except ImportError:
        print("Scikit-learn not available for classical kernel comparison")
        return None


if __name__ == "__main__":
    print("="*70)
    print("### Running basic quantum kernel demo ###")
    print("="*70)
    run_quantum_kernel_demo()
    
    print("\n" + "="*70)
    print("### Running quantum vs classical comparison ###")
    print("="*70)
    run_kernel_comparison()
