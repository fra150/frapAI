"""Basic classification example using FPAI framework.

Demonstrates how to use VQC for binary classification with different
feature maps and calibration techniques.
"""

import numpy as np
from typing import Dict, Any, Optional
import time

from ..models import VQC, HardwareEfficientAnsatz
from ..core import AngleEncoding, AmplitudeEncoding, ZZFeatureMap, POVM
from ..utils import (
    generate_quantum_dataset, preprocess_features, create_train_val_test_split,
    TemperatureScaling, comprehensive_evaluation, plot_calibration_curve,
    plot_training_history, plot_confusion_matrix
)


class BasicClassificationExample:
    """Basic classification example with VQC."""
    
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
        
    def run_example(self, dataset_name: str = 'moons', 
                   feature_map_type: str = 'angle',
                   use_calibration: bool = True,
                   verbose: bool = True) -> Dict[str, Any]:
        """
        Run the classification example.
        
        Args:
            dataset_name: Dataset to use ('moons', 'circles', 'spiral', 'xor')
            feature_map_type: Feature map type ('angle', 'amplitude', 'zz')
            use_calibration: Whether to use probability calibration
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results
        """
        if verbose:
            print(f"\n=== FPAI Basic Classification Example ===")
            print(f"Dataset: {dataset_name}")
            print(f"Feature Map: {feature_map_type}")
            print(f"Qubits: {self.n_qubits}, Layers: {self.n_layers}")
            print(f"Calibration: {use_calibration}")
            print("=" * 50)
        
        # Step 1: Generate dataset
        if verbose:
            print("\n1. Generating dataset...")
        
        # Determine number of features based on dataset
        if dataset_name in ['moons', 'circles']:
            n_features = 2
        else:
            n_features = self.n_qubits
            
        X, y = generate_quantum_dataset(
            dataset_type=dataset_name,
            n_samples=800,
            n_features=n_features,
            noise=0.1,
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
            X_scaled, y, train_size=0.6, val_size=0.2, test_size=0.2,
            random_state=self.random_state
        )
        
        # Step 3: Create feature map
        if verbose:
            print("3. Creating {} feature map...".format(feature_map_type))
        
        # Use number of qubits based on features
        n_qubits_needed = max(n_features, 2)  # At least 2 qubits
        
        if feature_map_type == 'angle':
            feature_map = AngleEncoding(n_features=n_features)
        elif feature_map_type == 'amplitude':
            feature_map = AmplitudeEncoding(n_features=n_features)
        elif feature_map_type == 'zz':
            feature_map = ZZFeatureMap(n_qubits=n_qubits_needed, reps=2)
        else:
            raise ValueError(f"Unknown feature map type: {feature_map_type}")
        
        # Step 4: Create variational ansatz
        if verbose:
            print("4. Creating variational ansatz...")
        
        ansatz = HardwareEfficientAnsatz(
            n_qubits=n_qubits_needed,
            n_layers=self.n_layers
        )
        
        # Step 5: Create POVM
        if verbose:
            print("5. Creating POVM...")
        
        povm = POVM.projective(n_outcomes=2, dim=2**n_qubits_needed)
        
        # Step 6: Create and train VQC
        if verbose:
            print("6. Training VQC...")
        
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            povm=povm,
            shots=1024
        )
        
        start_time = time.time()
        
        history = vqc.fit(
            splits['X_train'], splits['y_train'],
            epochs=50,
            learning_rate=0.1,
            batch_size=32,
            validation_split=0.2,
            early_stopping=True,
            patience=10,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        
        # Step 7: Make predictions
        if verbose:
            print("7. Making predictions...")
        
        y_pred_proba = vqc.predict_proba(splits['X_test'])
        y_pred = vqc.predict(splits['X_test'])
        
        # Step 8: Calibration (optional)
        calibrator = None
        y_pred_proba_cal = y_pred_proba.copy()
        
        if use_calibration:
            if verbose:
                print("8. Calibrating predictions...")
            
            # Get validation predictions for calibration
            y_val_proba = vqc.predict_proba(splits['X_val'])
            
            calibrator = TemperatureScaling()
            calibrator.fit(y_val_proba, splits['y_val'])
            y_pred_proba_cal = calibrator.calibrate(y_pred_proba)
        
        # Step 9: Evaluate results
        if verbose:
            print("9. Evaluating results...")
        
        # Uncalibrated results
        results_uncal = comprehensive_evaluation(
            splits['y_test'], y_pred_proba, y_pred
        )
        
        # Calibrated results (if applicable)
        results_cal = None
        if use_calibration:
            y_pred_cal = (y_pred_proba_cal[:, 1] > 0.5).astype(int)
            results_cal = comprehensive_evaluation(
                splits['y_test'], y_pred_proba_cal, y_pred_cal
            )
        
        # Compile results
        experiment_results = {
            'dataset': dataset_name,
            'feature_map': feature_map_type,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'training_time': training_time,
            'n_train_samples': len(splits['X_train']),
            'n_test_samples': len(splits['X_test']),
            'training_history': history,
            'uncalibrated_results': results_uncal,
            'calibrated_results': results_cal,
            'predictions': {
                'y_true': splits['y_test'],
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_pred_proba_calibrated': y_pred_proba_cal if use_calibration else None
            }
        }
        
        # Store results
        self.results[f"{dataset_name}_{feature_map_type}"] = experiment_results
        
        # Print summary
        if verbose:
            self._print_results_summary(experiment_results)
        
        return experiment_results
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of the results."""
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        
        uncal = results['uncalibrated_results']
        cal = results['calibrated_results']
        
        print(f"Training time: {results['training_time']:.2f}s")
        print(f"Training samples: {results['n_train_samples']}")
        print(f"Test samples: {results['n_test_samples']}")
        
        print("\nUncalibrated Performance:")
        print(f"  Accuracy: {uncal['accuracy']:.3f}")
        print(f"  AUROC: {uncal['auroc']:.3f}")
        print(f"  ECE: {uncal['ece']:.3f}")
        print(f"  Brier Score: {uncal['brier_score']:.3f}")
        print(f"  NLL: {uncal['nll']:.3f}")
        
        if cal is not None:
            print("\nCalibrated Performance:")
            print(f"  Accuracy: {cal['accuracy']:.3f}")
            print(f"  AUROC: {cal['auroc']:.3f}")
            print(f"  ECE: {cal['ece']:.3f}")
            print(f"  Brier Score: {cal['brier_score']:.3f}")
            print(f"  NLL: {cal['nll']:.3f}")
            
            print("\nCalibration Improvement:")
            print(f"  ΔECE: {uncal['ece'] - cal['ece']:.3f}")
            print(f"  ΔBrier: {uncal['brier_score'] - cal['brier_score']:.3f}")
            print(f"  ΔNLL: {uncal['nll'] - cal['nll']:.3f}")
    
    def visualize_results(self, experiment_key: str, save_plots: bool = False, plot_dir: str = "plots"):
        """Visualize results from an experiment.
        
        Args:
            experiment_key: Key of experiment to visualize
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
        """
        if experiment_key not in self.results:
            raise ValueError(f"No results found for experiment: {experiment_key}")
        
        results = self.results[experiment_key]
        preds = results['predictions']
        
        # Plot training history
        if results['training_history']:
            plot_training_history(
                results['training_history'],
                title=f"Training History - {experiment_key}",
                save_path=f"{plot_dir}/training_history_{experiment_key}.png" if save_plots else None
            )
        
        # Plot calibration curves
        plot_calibration_curve(
            preds['y_pred_proba'][:, 1], preds['y_true'],
            title=f"Calibration Curve (Uncalibrated) - {experiment_key}",
            save_path=f"{plot_dir}/calibration_uncal_{experiment_key}.png" if save_plots else None
        )
        
        if preds['y_pred_proba_calibrated'] is not None:
            plot_calibration_curve(
                preds['y_pred_proba_calibrated'][:, 1], preds['y_true'],
                title=f"Calibration Curve (Calibrated) - {experiment_key}",
                save_path=f"{plot_dir}/calibration_cal_{experiment_key}.png" if save_plots else None
            )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            preds['y_true'], preds['y_pred'],
            title=f"Confusion Matrix - {experiment_key}",
            save_path=f"{plot_dir}/confusion_matrix_{experiment_key}.png" if save_plots else None
        )
    
    def compare_feature_maps(self, dataset_name: str = 'moons', 
                           feature_maps: list = ['angle', 'amplitude', 'zz'],
                           verbose: bool = True) -> Dict[str, Any]:
        """Compare different feature maps on the same dataset.
        
        Args:
            dataset_name: Dataset to use
            feature_maps: List of feature map types to compare
            verbose: Whether to print progress
            
        Returns:
            Dictionary with comparison results
        """
        if verbose:
            print(f"\n=== Feature Map Comparison on {dataset_name} ===")
        
        comparison_results = {}
        
        for fm_type in feature_maps:
            if verbose:
                print(f"\nTesting {fm_type} feature map...")
            
            try:
                result = self.run_example(
                    dataset_name=dataset_name,
                    feature_map_type=fm_type,
                    use_calibration=True,
                    verbose=False
                )
                comparison_results[fm_type] = result
                
                if verbose:
                    uncal = result['uncalibrated_results']
                    print(f"  Accuracy: {uncal['accuracy']:.3f}, ECE: {uncal['ece']:.3f}")
                    
            except Exception as e:
                if verbose:
                    print(f"  Error with {fm_type}: {str(e)}")
                comparison_results[fm_type] = None
        
        # Print comparison summary
        if verbose:
            self._print_comparison_summary(comparison_results)
        
        return comparison_results
    
    def _print_comparison_summary(self, comparison_results: Dict[str, Any]):
        """Print comparison summary."""
        print("\n" + "="*60)
        print("FEATURE MAP COMPARISON SUMMARY")
        print("="*60)
        
        print(f"{'Feature Map':<15} {'Accuracy':<10} {'AUROC':<8} {'ECE':<8} {'Time(s)':<8}")
        print("-" * 60)
        
        for fm_type, result in comparison_results.items():
            if result is not None:
                uncal = result['uncalibrated_results']
                print(f"{fm_type:<15} {uncal['accuracy']:<10.3f} {uncal['auroc']:<8.3f} "
                      f"{uncal['ece']:<8.3f} {result['training_time']:<8.1f}")
            else:
                print(f"{fm_type:<15} {'ERROR':<10} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8}")


def run_basic_example():
    """Run a basic example with default settings."""
    example = BasicClassificationExample(n_qubits=4, n_layers=2)
    
    # Run single experiment
    result = example.run_example(
        dataset_name='moons',
        feature_map_type='angle',
        use_calibration=True,
        verbose=True
    )
    
    # Visualize results
    try:
        example.visualize_results('moons_angle')
    except ImportError:
        print("\nVisualization skipped (matplotlib not available)")
    
    return result


def run_feature_map_comparison():
    """Run feature map comparison."""
    example = BasicClassificationExample(n_qubits=4, n_layers=2)
    
    # Compare feature maps
    comparison = example.compare_feature_maps(
        dataset_name='moons',
        feature_maps=['angle', 'zz'],  # Skip amplitude for speed
        verbose=True
    )
    
    return comparison


if __name__ == "__main__":
    # Run basic example
    print("Running basic classification example...")
    run_basic_example()
    
    print("\n" + "="*70)
    
    # Run feature map comparison
    print("Running feature map comparison...")
    run_feature_map_comparison()