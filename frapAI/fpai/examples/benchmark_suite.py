"""Benchmark suite for FPAI framework.

Comprehensive benchmarking of quantum machine learning models
with performance analysis and comparison with classical methods.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import json
from datetime import datetime

from ..models import VQC, QuantumKernel, HardwareEfficientAnsatz
from ..core import AngleEncoding, AmplitudeEncoding, ZZFeatureMap, POVM
from ..utils import (
    generate_quantum_dataset, load_benchmark_dataset, preprocess_features,
    create_train_val_test_split, comprehensive_evaluation, 
    TemperatureScaling, expected_calibration_error
)

# Classical ML imports for comparison
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BenchmarkSuite:
    """Comprehensive benchmark suite for FPAI framework."""
    
    def __init__(self, n_qubits: int = 4, random_state: int = 42):
        """
        Initialize benchmark suite.
        
        Args:
            n_qubits: Number of qubits for quantum models
            random_state: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.random_state = random_state
        self.results = {}
        self.benchmark_config = {
            'n_qubits': n_qubits,
            'random_state': random_state,
            'timestamp': datetime.now().isoformat(),
            'sklearn_available': SKLEARN_AVAILABLE
        }
        
    def run_quantum_model_benchmark(self, 
                                  datasets: List[str] = ['moons', 'circles', 'iris'],
                                  models: List[str] = ['vqc', 'quantum_kernel'],
                                  feature_maps: List[str] = ['angle', 'amplitude'],
                                  verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive quantum model benchmark.
        
        Args:
            datasets: List of datasets to test
            models: List of quantum models to test
            feature_maps: List of feature maps to test
            verbose: Whether to print progress
            
        Returns:
            Dictionary with benchmark results
        """
        if verbose:
            print("\n" + "="*70)
            print("QUANTUM MODEL BENCHMARK SUITE")
            print("="*70)
            print(f"Datasets: {datasets}")
            print(f"Models: {models}")
            print(f"Feature Maps: {feature_maps}")
            print(f"Qubits: {self.n_qubits}")
        
        benchmark_results = {
            'config': self.benchmark_config.copy(),
            'datasets': {},
            'summary': {}
        }
        
        total_experiments = len(datasets) * len(models) * len(feature_maps)
        experiment_count = 0
        
        for dataset_name in datasets:
            if verbose:
                print(f"\n{'='*20} Dataset: {dataset_name} {'='*20}")
            
            benchmark_results['datasets'][dataset_name] = {}
            
            # Load dataset
            try:
                X, y = self._load_dataset(dataset_name)
                
                # Preprocess
                X_scaled, scaler = preprocess_features(
                    X, scaler_type='minmax', feature_range=(0, 1)
                )
                
                # Create splits
                splits = create_train_val_test_split(
                    X_scaled, y, train_size=0.6, val_size=0.2, test_size=0.2,
                    random_state=self.random_state
                )
                
                dataset_info = {
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'n_classes': len(np.unique(y)),
                    'class_distribution': {str(cls): int(count) for cls, count in 
                                         zip(*np.unique(y, return_counts=True))}
                }
                
                benchmark_results['datasets'][dataset_name]['info'] = dataset_info
                benchmark_results['datasets'][dataset_name]['models'] = {}
                
                for model_name in models:
                    benchmark_results['datasets'][dataset_name]['models'][model_name] = {}
                    
                    for fm_name in feature_maps:
                        experiment_count += 1
                        
                        if verbose:
                            print(f"\n[{experiment_count}/{total_experiments}] "
                                  f"Testing {model_name} with {fm_name} encoding...")
                        
                        try:
                            result = self._run_single_experiment(
                                dataset_name, model_name, fm_name, splits, verbose=False
                            )
                            
                            benchmark_results['datasets'][dataset_name]['models'][model_name][fm_name] = result
                            
                            if verbose:
                                eval_res = result['evaluation']
                                print(f"   Accuracy: {eval_res['accuracy']:.3f}, "
                                      f"AUROC: {eval_res['auroc']:.3f}, "
                                      f"ECE: {eval_res['ece']:.3f}, "
                                      f"Time: {result['total_time']:.2f}s")
                                
                        except Exception as e:
                            if verbose:
                                print(f"   Error: {str(e)}")
                            benchmark_results['datasets'][dataset_name]['models'][model_name][fm_name] = {
                                'error': str(e)
                            }
                            
            except Exception as e:
                if verbose:
                    print(f"Error loading dataset {dataset_name}: {str(e)}")
                benchmark_results['datasets'][dataset_name] = {'error': str(e)}
        
        # Generate summary
        benchmark_results['summary'] = self._generate_benchmark_summary(benchmark_results)
        
        # Store results
        self.results['quantum_benchmark'] = benchmark_results
        
        if verbose:
            self._print_benchmark_summary(benchmark_results)
        
        return benchmark_results
    
    def run_quantum_vs_classical_comparison(self,
                                          datasets: List[str] = ['moons', 'iris'],
                                          quantum_configs: List[Dict[str, str]] = None,
                                          classical_models: List[str] = None,
                                          verbose: bool = True) -> Dict[str, Any]:
        """
        Compare quantum models with classical ML models.
        
        Args:
            datasets: List of datasets to test
            quantum_configs: List of quantum model configurations
            classical_models: List of classical models to test
            verbose: Whether to print progress
            
        Returns:
            Dictionary with comparison results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for classical model comparison")
        
        if quantum_configs is None:
            quantum_configs = [
                {'model': 'vqc', 'feature_map': 'angle'},
                {'model': 'quantum_kernel', 'feature_map': 'amplitude'}
            ]
        
        if classical_models is None:
            classical_models = ['random_forest', 'svm', 'logistic_regression']
        
        if verbose:
            print("\n" + "="*70)
            print("QUANTUM VS CLASSICAL COMPARISON")
            print("="*70)
        
        comparison_results = {
            'config': self.benchmark_config.copy(),
            'datasets': {},
            'summary': {}
        }
        
        for dataset_name in datasets:
            if verbose:
                print(f"\n{'='*20} Dataset: {dataset_name} {'='*20}")
            
            try:
                # Load and preprocess dataset
                X, y = self._load_dataset(dataset_name)
                X_scaled, scaler = preprocess_features(
                    X, scaler_type='standard'  # Use standard scaling for classical models
                )
                
                splits = create_train_val_test_split(
                    X_scaled, y, train_size=0.6, val_size=0.2, test_size=0.2,
                    random_state=self.random_state
                )
                
                dataset_results = {
                    'quantum_models': {},
                    'classical_models': {}
                }
                
                # Test quantum models
                if verbose:
                    print("\nTesting quantum models...")
                
                for config in quantum_configs:
                    model_name = config['model']
                    fm_name = config['feature_map']
                    config_key = f"{model_name}_{fm_name}"
                    
                    if verbose:
                        print(f"   {config_key}...")
                    
                    try:
                        result = self._run_single_experiment(
                            dataset_name, model_name, fm_name, splits, verbose=False
                        )
                        dataset_results['quantum_models'][config_key] = result
                        
                        if verbose:
                            eval_res = result['evaluation']
                            print(f"     Accuracy: {eval_res['accuracy']:.3f}, "
                                  f"Time: {result['total_time']:.2f}s")
                            
                    except Exception as e:
                        if verbose:
                            print(f"     Error: {str(e)}")
                        dataset_results['quantum_models'][config_key] = {'error': str(e)}
                
                # Test classical models
                if verbose:
                    print("\nTesting classical models...")
                
                for model_name in classical_models:
                    if verbose:
                        print(f"   {model_name}...")
                    
                    try:
                        result = self._run_classical_experiment(
                            model_name, splits, verbose=False
                        )
                        dataset_results['classical_models'][model_name] = result
                        
                        if verbose:
                            eval_res = result['evaluation']
                            print(f"     Accuracy: {eval_res['accuracy']:.3f}, "
                                  f"Time: {result['total_time']:.2f}s")
                            
                    except Exception as e:
                        if verbose:
                            print(f"     Error: {str(e)}")
                        dataset_results['classical_models'][model_name] = {'error': str(e)}
                
                comparison_results['datasets'][dataset_name] = dataset_results
                
            except Exception as e:
                if verbose:
                    print(f"Error processing dataset {dataset_name}: {str(e)}")
                comparison_results['datasets'][dataset_name] = {'error': str(e)}
        
        # Generate comparison summary
        comparison_results['summary'] = self._generate_comparison_summary(comparison_results)
        
        # Store results
        self.results['quantum_vs_classical'] = comparison_results
        
        if verbose:
            self._print_comparison_summary(comparison_results)
        
        return comparison_results
    
    def _load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset by name."""
        if dataset_name in ['moons', 'circles', 'xor', 'spiral', 'blobs']:
            return generate_quantum_dataset(
                dataset_type=dataset_name,
                n_samples=800,
                n_features=min(self.n_qubits, 4),
                noise=0.1,
                random_state=self.random_state
            )
        else:
            return load_benchmark_dataset(
                dataset_name, 
                n_features=min(self.n_qubits, 4),
                random_state=self.random_state
            )
    
    def _run_single_experiment(self, dataset_name: str, model_name: str, 
                             fm_name: str, splits: Dict[str, np.ndarray],
                             verbose: bool = False) -> Dict[str, Any]:
        """Run single quantum model experiment."""
        start_time = time.time()
        
        # Create feature map
        if fm_name == 'angle':
            feature_map = AngleEncoding(n_features=self.n_qubits)
        elif fm_name == 'amplitude':
            feature_map = AmplitudeEncoding(n_features=self.n_qubits)
        elif fm_name == 'zz':
            feature_map = ZZFeatureMap(
                n_qubits=self.n_qubits, reps=2, entanglement='linear'
            )
        else:
            raise ValueError(f"Unknown feature map: {fm_name}")
        
        # Create and train model
        if model_name == 'vqc':
            ansatz = HardwareEfficientAnsatz(
                n_qubits=self.n_qubits,
                n_layers=2,
                entangling_pattern='linear'
            )
            
            povm = POVM.projective(n_outcomes=2, dim=2**self.n_qubits)
            
            model = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                povm=povm,
                shots=1024
            )
            
            training_start = time.time()
            history = model.fit(
                splits['X_train'], splits['y_train'],
                X_val=splits['X_val'], y_val=splits['y_val'],
                epochs=20,
                learning_rate=0.1,
                batch_size=32,
                early_stopping_patience=5,
                verbose=verbose
            )
            training_time = time.time() - training_start
            
        elif model_name == 'quantum_kernel':
            model = QuantumKernel(
                feature_map=feature_map,
                kernel_type='fidelity',
                shots=1024,
                random_state=self.random_state
            )
            
            training_start = time.time()
            model.fit(splits['X_train'], splits['y_train'])
            training_time = time.time() - training_start
            history = {}
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Evaluate model
        inference_start = time.time()
        y_pred_proba = model.predict_proba(splits['X_test'])
        y_pred = model.predict(splits['X_test'])
        inference_time = time.time() - inference_start
        
        evaluation = comprehensive_evaluation(
            splits['y_test'], y_pred_proba, y_pred
        )
        
        # Test calibration
        calibration_start = time.time()
        calibrator = TemperatureScaling()
        calibrator.fit(splits['X_val'], splits['y_val'], model)
        y_pred_proba_cal = calibrator.predict_proba(splits['X_test'], model)
        calibrated_ece = expected_calibration_error(y_pred_proba_cal[:, 1], splits['y_test'])
        calibration_time = time.time() - calibration_start
        
        total_time = time.time() - start_time
        
        return {
            'model_name': model_name,
            'feature_map': fm_name,
            'evaluation': evaluation,
            'calibrated_ece': calibrated_ece,
            'training_time': training_time,
            'inference_time': inference_time,
            'calibration_time': calibration_time,
            'total_time': total_time,
            'history': history,
            'model_params': {
                'n_qubits': self.n_qubits,
                'shots': 1024
            }
        }
    
    def _run_classical_experiment(self, model_name: str, 
                                splits: Dict[str, np.ndarray],
                                verbose: bool = False) -> Dict[str, Any]:
        """Run single classical model experiment."""
        start_time = time.time()
        
        # Create classical model
        if model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            )
        elif model_name == 'svm':
            model = SVC(
                probability=True, random_state=self.random_state
            )
        elif model_name == 'logistic_regression':
            model = LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
        elif model_name == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=500
            )
        else:
            raise ValueError(f"Unknown classical model: {model_name}")
        
        # Train model
        training_start = time.time()
        model.fit(splits['X_train'], splits['y_train'])
        training_time = time.time() - training_start
        
        # Evaluate model
        inference_start = time.time()
        y_pred_proba = model.predict_proba(splits['X_test'])
        y_pred = model.predict(splits['X_test'])
        inference_time = time.time() - inference_start
        
        evaluation = comprehensive_evaluation(
            splits['y_test'], y_pred_proba, y_pred
        )
        
        total_time = time.time() - start_time
        
        return {
            'model_name': model_name,
            'evaluation': evaluation,
            'training_time': training_time,
            'inference_time': inference_time,
            'total_time': total_time,
            'model_params': model.get_params() if hasattr(model, 'get_params') else {}
        }
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of benchmark results."""
        summary = {
            'total_experiments': 0,
            'successful_experiments': 0,
            'failed_experiments': 0,
            'best_performers': {},
            'average_metrics': {},
            'timing_analysis': {}
        }
        
        all_results = []
        
        for dataset_name, dataset_data in results['datasets'].items():
            if 'error' in dataset_data:
                continue
                
            for model_name, model_data in dataset_data.get('models', {}).items():
                for fm_name, result in model_data.items():
                    summary['total_experiments'] += 1
                    
                    if 'error' in result:
                        summary['failed_experiments'] += 1
                    else:
                        summary['successful_experiments'] += 1
                        all_results.append({
                            'dataset': dataset_name,
                            'model': model_name,
                            'feature_map': fm_name,
                            **result
                        })
        
        if all_results:
            # Find best performers
            metrics = ['accuracy', 'auroc', 'f1_score']
            for metric in metrics:
                best_result = max(all_results, key=lambda x: x['evaluation'][metric])
                summary['best_performers'][metric] = {
                    'dataset': best_result['dataset'],
                    'model': best_result['model'],
                    'feature_map': best_result['feature_map'],
                    'score': best_result['evaluation'][metric]
                }
            
            # Calculate average metrics
            for metric in metrics + ['ece', 'brier_score']:
                scores = [r['evaluation'][metric] for r in all_results]
                summary['average_metrics'][metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            
            # Timing analysis
            timing_metrics = ['training_time', 'inference_time', 'total_time']
            for metric in timing_metrics:
                times = [r[metric] for r in all_results]
                summary['timing_analysis'][metric] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times)
                }
        
        return summary
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of quantum vs classical comparison."""
        summary = {
            'quantum_wins': 0,
            'classical_wins': 0,
            'ties': 0,
            'average_performance': {
                'quantum': {},
                'classical': {}
            },
            'timing_comparison': {
                'quantum': {},
                'classical': {}
            }
        }
        
        quantum_results = []
        classical_results = []
        
        for dataset_name, dataset_data in results['datasets'].items():
            if 'error' in dataset_data:
                continue
            
            # Collect quantum results
            for model_name, result in dataset_data.get('quantum_models', {}).items():
                if 'error' not in result:
                    quantum_results.append(result)
            
            # Collect classical results
            for model_name, result in dataset_data.get('classical_models', {}).items():
                if 'error' not in result:
                    classical_results.append(result)
            
            # Compare best performers on this dataset
            if dataset_data.get('quantum_models') and dataset_data.get('classical_models'):
                quantum_best = max(
                    [r for r in dataset_data['quantum_models'].values() if 'error' not in r],
                    key=lambda x: x['evaluation']['accuracy'],
                    default=None
                )
                classical_best = max(
                    [r for r in dataset_data['classical_models'].values() if 'error' not in r],
                    key=lambda x: x['evaluation']['accuracy'],
                    default=None
                )
                
                if quantum_best and classical_best:
                    q_acc = quantum_best['evaluation']['accuracy']
                    c_acc = classical_best['evaluation']['accuracy']
                    
                    if q_acc > c_acc + 0.01:  # 1% threshold
                        summary['quantum_wins'] += 1
                    elif c_acc > q_acc + 0.01:
                        summary['classical_wins'] += 1
                    else:
                        summary['ties'] += 1
        
        # Calculate average performance
        if quantum_results:
            metrics = ['accuracy', 'auroc', 'f1_score']
            for metric in metrics:
                scores = [r['evaluation'][metric] for r in quantum_results]
                summary['average_performance']['quantum'][metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
            
            times = [r['total_time'] for r in quantum_results]
            summary['timing_comparison']['quantum'] = {
                'mean': np.mean(times),
                'std': np.std(times)
            }
        
        if classical_results:
            for metric in metrics:
                scores = [r['evaluation'][metric] for r in classical_results]
                summary['average_performance']['classical'][metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
            
            times = [r['total_time'] for r in classical_results]
            summary['timing_comparison']['classical'] = {
                'mean': np.mean(times),
                'std': np.std(times)
            }
        
        return summary
    
    def _print_benchmark_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        summary = results['summary']
        
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful_experiments']}")
        print(f"Failed: {summary['failed_experiments']}")
        
        if summary['best_performers']:
            print("\nBest Performers:")
            for metric, best in summary['best_performers'].items():
                print(f"  {metric.title()}: {best['model']}_{best['feature_map']} "
                      f"on {best['dataset']} ({best['score']:.3f})")
        
        if summary['average_metrics']:
            print("\nAverage Performance:")
            for metric, stats in summary['average_metrics'].items():
                print(f"  {metric.title()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        if summary['timing_analysis']:
            print("\nTiming Analysis:")
            for metric, stats in summary['timing_analysis'].items():
                print(f"  {metric.replace('_', ' ').title()}: {stats['mean']:.2f}s ± {stats['std']:.2f}s")
    
    def _print_comparison_summary(self, results: Dict[str, Any]):
        """Print quantum vs classical comparison summary."""
        summary = results['summary']
        
        print("\n" + "="*70)
        print("QUANTUM VS CLASSICAL COMPARISON SUMMARY")
        print("="*70)
        
        total_comparisons = summary['quantum_wins'] + summary['classical_wins'] + summary['ties']
        
        if total_comparisons > 0:
            print(f"Dataset comparisons: {total_comparisons}")
            print(f"Quantum wins: {summary['quantum_wins']} ({summary['quantum_wins']/total_comparisons*100:.1f}%)")
            print(f"Classical wins: {summary['classical_wins']} ({summary['classical_wins']/total_comparisons*100:.1f}%)")
            print(f"Ties: {summary['ties']} ({summary['ties']/total_comparisons*100:.1f}%)")
        
        if summary['average_performance']:
            print("\nAverage Performance Comparison:")
            metrics = ['accuracy', 'auroc', 'f1_score']
            
            print(f"{'Metric':<12} {'Quantum':<15} {'Classical':<15} {'Difference':<12}")
            print("-" * 60)
            
            for metric in metrics:
                q_perf = summary['average_performance'].get('quantum', {}).get(metric, {})
                c_perf = summary['average_performance'].get('classical', {}).get(metric, {})
                
                if q_perf and c_perf:
                    q_mean = q_perf['mean']
                    c_mean = c_perf['mean']
                    diff = q_mean - c_mean
                    
                    print(f"{metric.title():<12} {q_mean:.3f} ± {q_perf['std']:.3f}   "
                          f"{c_mean:.3f} ± {c_perf['std']:.3f}   {diff:+.3f}")
        
        if summary['timing_comparison']:
            print("\nTiming Comparison:")
            q_time = summary['timing_comparison'].get('quantum', {})
            c_time = summary['timing_comparison'].get('classical', {})
            
            if q_time and c_time:
                print(f"Quantum: {q_time['mean']:.2f}s ± {q_time['std']:.2f}s")
                print(f"Classical: {c_time['mean']:.2f}s ± {c_time['std']:.2f}s")
                print(f"Ratio (Q/C): {q_time['mean']/c_time['mean']:.2f}x")
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file.
        
        Args:
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fpai_benchmark_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def run_quantum_benchmark():
    """Run quantum model benchmark."""
    benchmark = BenchmarkSuite(n_qubits=4)
    
    results = benchmark.run_quantum_model_benchmark(
        datasets=['moons', 'circles'],
        models=['vqc', 'quantum_kernel'],
        feature_maps=['angle', 'amplitude'],
        verbose=True
    )
    
    return results


def run_quantum_vs_classical_benchmark():
    """Run quantum vs classical comparison."""
    if not SKLEARN_AVAILABLE:
        print("Scikit-learn not available. Skipping classical comparison.")
        return None
    
    benchmark = BenchmarkSuite(n_qubits=4)
    
    results = benchmark.run_quantum_vs_classical_comparison(
        datasets=['moons', 'iris'],
        quantum_configs=[
            {'model': 'vqc', 'feature_map': 'angle'},
            {'model': 'quantum_kernel', 'feature_map': 'amplitude'}
        ],
        classical_models=['random_forest', 'svm'],
        verbose=True
    )
    
    return results


def run_full_benchmark_suite():
    """Run complete benchmark suite."""
    benchmark = BenchmarkSuite(n_qubits=4)
    
    print("Running quantum model benchmark...")
    quantum_results = benchmark.run_quantum_model_benchmark(
        datasets=['moons', 'circles'],
        models=['vqc'],
        feature_maps=['angle', 'amplitude'],
        verbose=True
    )
    
    if SKLEARN_AVAILABLE:
        print("\n" + "="*70)
        print("Running quantum vs classical comparison...")
        comparison_results = benchmark.run_quantum_vs_classical_comparison(
            datasets=['moons'],
            verbose=True
        )
    
    # Save results
    benchmark.save_results()
    
    return benchmark.results


if __name__ == "__main__":
    # Run full benchmark suite
    print("Running FPAI benchmark suite...")
    results = run_full_benchmark_suite()
    print("\nBenchmark completed!")