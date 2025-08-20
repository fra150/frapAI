"""Calibration example using FPAI framework.

Demonstrates different calibration techniques and their impact
on prediction reliability and uncertainty quantification.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import time

from ..models import VQC, HardwareEfficientAnsatz
from ..core import AngleEncoding, POVM
from ..utils import (
    generate_quantum_dataset, preprocess_features, create_train_val_test_split,
    TemperatureScaling, PlattScaling, IsotonicCalibration, VectorScaling,
    comprehensive_evaluation, expected_calibration_error, reliability_diagram,
    plot_calibration_curve, plot_reliability_diagram
)


class CalibrationExample:
    """Demonstration of calibration techniques."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, random_state: int = 42):
        """
        Initialize the example.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of ansatz layers
            random_state: Random seed
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.random_state = random_state
        self.results = {}
        
    def run_calibration_demo(self, dataset_name: str = 'moons',
                           calibration_methods: List[str] = ['temperature', 'platt', 'isotonic'],
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Run calibration demonstration.
        
        Args:
            dataset_name: Dataset to use
            calibration_methods: List of calibration methods to test
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results
        """
        if verbose:
            print(f"\n=== Calibration Demo ===")
            print(f"Dataset: {dataset_name}")
            print(f"Calibration Methods: {calibration_methods}")
            print(f"Qubits: {self.n_qubits}, Layers: {self.n_layers}")
            print("=" * 40)
        
        # Step 1: Generate dataset
        if verbose:
            print("\n1. Generating dataset...")
        
        X, y = generate_quantum_dataset(
            dataset_type=dataset_name,
            n_samples=1000,
            n_features=self.n_qubits,
            noise=0.15,  # More noise to create miscalibration
            random_state=self.random_state
        )
        
        # Step 2: Preprocess data
        if verbose:
            print("2. Preprocessing data...")
        
        X_scaled, scaler = preprocess_features(
            X, scaler_type='minmax', feature_range=(0, 1)
        )
        
        # Create train/val/test splits
        splits = create_train_val_test_split(
            X_scaled, y, train_size=0.5, val_size=0.25, test_size=0.25,
            random_state=self.random_state
        )
        
        # Step 3: Train base model
        if verbose:
            print("3. Training base VQC model...")
        
        feature_map = AngleEncoding(n_features=self.n_qubits)
        ansatz = HardwareEfficientAnsatz(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            entangling_pattern='linear'
        )
        
        povm = POVM.projective(n_outcomes=2, dim=2**n_qubits)
        
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            povm=povm,
            shots=1024
        )
        
        # Train with early stopping to potentially create miscalibration
        history = vqc.fit(
            splits['X_train'], splits['y_train'],
            X_val=splits['X_val'], y_val=splits['y_val'],
            epochs=30,
            learning_rate=0.1,
            batch_size=32,
            early_stopping_patience=5,
            verbose=False
        )
        
        # Step 4: Get uncalibrated predictions
        if verbose:
            print("4. Getting uncalibrated predictions...")
        
        y_pred_proba_uncal = vqc.predict_proba(splits['X_test'])
        y_pred_uncal = vqc.predict(splits['X_test'])
        
        # Evaluate uncalibrated model
        uncalibrated_results = comprehensive_evaluation(
            splits['y_test'], y_pred_proba_uncal, y_pred_uncal
        )
        
        # Step 5: Apply calibration methods
        if verbose:
            print("5. Applying calibration methods...")
        
        calibration_results = {}
        calibrated_predictions = {}
        
        for method in calibration_methods:
            if verbose:
                print(f"   Testing {method} calibration...")
            
            start_time = time.time()
            
            # Create calibrator
            if method == 'temperature':
                calibrator = TemperatureScaling()
            elif method == 'platt':
                calibrator = PlattScaling()
            elif method == 'isotonic':
                calibrator = IsotonicCalibration()
            elif method == 'vector':
                calibrator = VectorScaling()
            else:
                raise ValueError(f"Unknown calibration method: {method}")
            
            # Fit calibrator on validation set
            calibrator.fit(splits['X_val'], splits['y_val'], vqc)
            
            # Apply calibration to test set
            y_pred_proba_cal = calibrator.predict_proba(splits['X_test'], vqc)
            y_pred_cal = (y_pred_proba_cal[:, 1] > 0.5).astype(int)
            
            calibration_time = time.time() - start_time
            
            # Evaluate calibrated predictions
            cal_results = comprehensive_evaluation(
                splits['y_test'], y_pred_proba_cal, y_pred_cal
            )
            
            calibration_results[method] = {
                'results': cal_results,
                'calibration_time': calibration_time,
                'calibrator_params': calibrator.get_params() if hasattr(calibrator, 'get_params') else {}
            }
            
            calibrated_predictions[method] = {
                'y_pred_proba': y_pred_proba_cal,
                'y_pred': y_pred_cal
            }
        
        # Step 6: Analyze calibration improvement
        if verbose:
            print("6. Analyzing calibration improvement...")
        
        calibration_analysis = self._analyze_calibration_improvement(
            splits['y_test'], y_pred_proba_uncal, calibrated_predictions
        )
        
        # Compile results
        experiment_results = {
            'dataset': dataset_name,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'data_splits': {
                'n_train': len(splits['X_train']),
                'n_val': len(splits['X_val']),
                'n_test': len(splits['X_test'])
            },
            'uncalibrated_results': uncalibrated_results,
            'calibration_results': calibration_results,
            'calibration_analysis': calibration_analysis,
            'predictions': {
                'y_true': splits['y_test'],
                'uncalibrated': {
                    'y_pred_proba': y_pred_proba_uncal,
                    'y_pred': y_pred_uncal
                },
                'calibrated': calibrated_predictions
            },
            'training_history': history
        }
        
        # Store results
        key = f"{dataset_name}_calibration"
        self.results[key] = experiment_results
        
        # Print summary
        if verbose:
            self._print_calibration_summary(experiment_results)
        
        return experiment_results
    
    def _analyze_calibration_improvement(self, y_true: np.ndarray, 
                                       y_pred_proba_uncal: np.ndarray,
                                       calibrated_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Analyze calibration improvement across methods."""
        uncal_ece = expected_calibration_error(y_pred_proba_uncal[:, 1], y_true)
        
        improvements = {}
        
        for method, preds in calibrated_predictions.items():
            cal_ece = expected_calibration_error(preds['y_pred_proba'][:, 1], y_true)
            
            improvements[method] = {
                'ece_improvement': uncal_ece - cal_ece,
                'ece_relative_improvement': (uncal_ece - cal_ece) / uncal_ece if uncal_ece > 0 else 0,
                'uncalibrated_ece': uncal_ece,
                'calibrated_ece': cal_ece
            }
        
        # Find best method
        best_method = min(improvements.keys(), 
                         key=lambda x: improvements[x]['calibrated_ece'])
        
        return {
            'improvements': improvements,
            'best_method': best_method,
            'best_ece': improvements[best_method]['calibrated_ece']
        }
    
    def _print_calibration_summary(self, results: Dict[str, Any]):
        """Print calibration results summary."""
        print("\n" + "="*60)
        print("CALIBRATION RESULTS SUMMARY")
        print("="*60)
        
        uncal = results['uncalibrated_results']
        cal_results = results['calibration_results']
        analysis = results['calibration_analysis']
        
        print(f"Data splits: {results['data_splits']['n_train']}/"
              f"{results['data_splits']['n_val']}/"
              f"{results['data_splits']['n_test']} (train/val/test)")
        
        print("\nUncalibrated Performance:")
        print(f"  Accuracy: {uncal['accuracy']:.3f}")
        print(f"  AUROC: {uncal['auroc']:.3f}")
        print(f"  ECE: {uncal['ece']:.3f}")
        print(f"  Brier Score: {uncal['brier_score']:.3f}")
        print(f"  NLL: {uncal['nll']:.3f}")
        
        print("\nCalibration Results:")
        print(f"{'Method':<12} {'Accuracy':<9} {'ECE':<8} {'Brier':<8} {'NLL':<8} {'Time(s)':<8}")
        print("-" * 60)
        
        for method, result in cal_results.items():
            res = result['results']
            time_taken = result['calibration_time']
            print(f"{method:<12} {res['accuracy']:<9.3f} {res['ece']:<8.3f} "
                  f"{res['brier_score']:<8.3f} {res['nll']:<8.3f} {time_taken:<8.3f}")
        
        print("\nCalibration Improvements:")
        improvements = analysis['improvements']
        print(f"{'Method':<12} {'ΔECE':<8} {'Rel. Imp.':<10} {'Final ECE':<10}")
        print("-" * 45)
        
        for method, imp in improvements.items():
            rel_imp = imp['ece_relative_improvement'] * 100
            print(f"{method:<12} {imp['ece_improvement']:<8.3f} {rel_imp:<10.1f}% "
                  f"{imp['calibrated_ece']:<10.3f}")
        
        print(f"\nBest calibration method: {analysis['best_method']} "
              f"(ECE = {analysis['best_ece']:.3f})")
    
    def visualize_calibration_results(self, experiment_key: str, 
                                    save_plots: bool = False, 
                                    plot_dir: str = "plots"):
        """Visualize calibration results.
        
        Args:
            experiment_key: Key of experiment to visualize
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
        """
        if experiment_key not in self.results:
            raise ValueError(f"No results found for experiment: {experiment_key}")
        
        results = self.results[experiment_key]
        preds = results['predictions']
        
        # Plot uncalibrated reliability diagram
        plot_reliability_diagram(
            preds['uncalibrated']['y_pred_proba'][:, 1], 
            preds['y_true'],
            title="Reliability Diagram (Uncalibrated)",
            save_path=f"{plot_dir}/reliability_uncalibrated_{experiment_key}.png" if save_plots else None
        )
        
        # Plot calibrated reliability diagrams
        for method, cal_preds in preds['calibrated'].items():
            plot_reliability_diagram(
                cal_preds['y_pred_proba'][:, 1],
                preds['y_true'],
                title=f"Reliability Diagram ({method.title()} Calibrated)",
                save_path=f"{plot_dir}/reliability_{method}_{experiment_key}.png" if save_plots else None
            )
        
        # Plot calibration curves comparison
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            # Uncalibrated
            from ..utils.metrics import calibration_curve
            fraction_pos, mean_pred = calibration_curve(
                preds['uncalibrated']['y_pred_proba'][:, 1], preds['y_true']
            )
            axes[0].plot(mean_pred, fraction_pos, 'o-', label='Uncalibrated')
            axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.7)
            axes[0].set_title('Uncalibrated')
            axes[0].set_xlabel('Mean Predicted Probability')
            axes[0].set_ylabel('Fraction of Positives')
            axes[0].grid(True, alpha=0.3)
            
            # Calibrated methods
            for i, (method, cal_preds) in enumerate(preds['calibrated'].items()):
                if i < 3:  # Only plot first 3 methods
                    fraction_pos, mean_pred = calibration_curve(
                        cal_preds['y_pred_proba'][:, 1], preds['y_true']
                    )
                    axes[i+1].plot(mean_pred, fraction_pos, 'o-', label=method.title())
                    axes[i+1].plot([0, 1], [0, 1], 'k--', alpha=0.7)
                    axes[i+1].set_title(f'{method.title()} Calibrated')
                    axes[i+1].set_xlabel('Mean Predicted Probability')
                    axes[i+1].set_ylabel('Fraction of Positives')
                    axes[i+1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f"{plot_dir}/calibration_comparison_{experiment_key}.png", 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
    
    def run_calibration_comparison_suite(self, 
                                       datasets: List[str] = ['moons', 'circles'],
                                       methods: List[str] = ['temperature', 'platt', 'isotonic'],
                                       verbose: bool = True) -> Dict[str, Any]:
        """Run calibration comparison across multiple datasets.
        
        Args:
            datasets: List of datasets to test
            methods: List of calibration methods
            verbose: Whether to print progress
            
        Returns:
            Dictionary with all results
        """
        if verbose:
            print("\n" + "="*60)
            print("CALIBRATION COMPARISON SUITE")
            print("="*60)
        
        suite_results = {}
        
        for dataset in datasets:
            if verbose:
                print(f"\nTesting calibration on {dataset}...")
            
            try:
                result = self.run_calibration_demo(
                    dataset_name=dataset,
                    calibration_methods=methods,
                    verbose=False
                )
                
                suite_results[dataset] = result
                
                if verbose:
                    uncal_ece = result['uncalibrated_results']['ece']
                    best_method = result['calibration_analysis']['best_method']
                    best_ece = result['calibration_analysis']['best_ece']
                    improvement = uncal_ece - best_ece
                    
                    print(f"  Uncalibrated ECE: {uncal_ece:.3f}")
                    print(f"  Best method: {best_method} (ECE: {best_ece:.3f})")
                    print(f"  Improvement: {improvement:.3f}")
                    
            except Exception as e:
                if verbose:
                    print(f"  Error: {str(e)}")
                suite_results[dataset] = None
        
        return suite_results
    
    def analyze_calibration_reliability(self, experiment_key: str) -> Dict[str, Any]:
        """Analyze calibration reliability in detail.
        
        Args:
            experiment_key: Key of experiment to analyze
            
        Returns:
            Dictionary with detailed reliability analysis
        """
        if experiment_key not in self.results:
            raise ValueError(f"No results found for experiment: {experiment_key}")
        
        results = self.results[experiment_key]
        preds = results['predictions']
        y_true = preds['y_true']
        
        reliability_analysis = {}
        
        # Analyze uncalibrated
        uncal_proba = preds['uncalibrated']['y_pred_proba'][:, 1]
        bin_centers, accuracies, confidences = reliability_diagram(uncal_proba, y_true)
        
        reliability_analysis['uncalibrated'] = {
            'bin_centers': bin_centers,
            'accuracies': accuracies,
            'confidences': confidences,
            'reliability_score': np.mean(np.abs(accuracies - confidences))
        }
        
        # Analyze calibrated methods
        for method, cal_preds in preds['calibrated'].items():
            cal_proba = cal_preds['y_pred_proba'][:, 1]
            bin_centers, accuracies, confidences = reliability_diagram(cal_proba, y_true)
            
            reliability_analysis[method] = {
                'bin_centers': bin_centers,
                'accuracies': accuracies,
                'confidences': confidences,
                'reliability_score': np.mean(np.abs(accuracies - confidences))
            }
        
        return reliability_analysis


def run_calibration_demo():
    """Run a basic calibration demo."""
    demo = CalibrationExample(n_qubits=4, n_layers=2)
    
    # Run calibration demo
    result = demo.run_calibration_demo(
        dataset_name='moons',
        calibration_methods=['temperature', 'platt', 'isotonic'],
        verbose=True
    )
    
    # Visualize results
    try:
        demo.visualize_calibration_results('moons_calibration')
    except ImportError:
        print("\nVisualization skipped (matplotlib not available)")
    
    return result


def run_calibration_suite():
    """Run calibration comparison suite."""
    demo = CalibrationExample(n_qubits=4, n_layers=2)
    
    # Run suite
    suite_results = demo.run_calibration_comparison_suite(
        datasets=['moons', 'circles'],
        methods=['temperature', 'platt'],
        verbose=True
    )
    
    return suite_results


if __name__ == "__main__":
    # Run basic demo
    print("Running calibration demo...")
    run_calibration_demo()
    
    print("\n" + "="*70)
    
    # Run suite
    print("Running calibration suite...")
    run_calibration_suite()