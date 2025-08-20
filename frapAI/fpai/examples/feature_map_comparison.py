"""Feature map comparison example using FPAI framework.

Demonstrates different quantum feature maps and their impact
on classification performance and quantum advantage.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time

from ..models import VQC, QuantumKernel, HardwareEfficientAnsatz
from ..core import AngleEncoding, AmplitudeEncoding, ZZFeatureMap, POVM
from ..utils import (
    generate_quantum_dataset, preprocess_features, create_train_val_test_split,
    comprehensive_evaluation, plot_model_comparison, plot_feature_importance
)


class FeatureMapComparison:
    """Comparison of different quantum feature maps."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, random_state: int = 42):
        """
        Initialize the comparison.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of ansatz layers
            random_state: Random seed
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.random_state = random_state
        self.results = {}
        
    def compare_feature_maps(self, dataset_name: str = 'moons',
                           feature_maps: List[str] = ['angle', 'amplitude', 'zz'],
                           model_type: str = 'vqc',
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Compare different feature maps.
        
        Args:
            dataset_name: Dataset to use
            feature_maps: List of feature map types to test
            model_type: Type of model ('vqc' or 'kernel')
            verbose: Whether to print progress
            
        Returns:
            Dictionary with comparison results
        """
        if verbose:
            print(f"\n=== Feature Map Comparison ===")
            print(f"Dataset: {dataset_name}")
            print(f"Feature Maps: {feature_maps}")
            print(f"Model Type: {model_type.upper()}")
            print(f"Qubits: {self.n_qubits}, Layers: {self.n_layers}")
            print("=" * 40)
        
        # Step 1: Generate dataset
        if verbose:
            print("\n1. Generating dataset...")
        
        X, y = generate_quantum_dataset(
            dataset_type=dataset_name,
            n_samples=800,
            n_features=self.n_qubits,
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
        
        # Step 3: Test each feature map
        if verbose:
            print("3. Testing feature maps...")
        
        feature_map_results = {}
        
        for fm_name in feature_maps:
            if verbose:
                print(f"   Testing {fm_name} encoding...")
            
            start_time = time.time()
            
            try:
                # Create feature map
                feature_map = self._create_feature_map(fm_name)
                
                # Train model
                if model_type == 'vqc':
                    model, history = self._train_vqc_model(
                        feature_map, splits, verbose=False
                    )
                elif model_type == 'kernel':
                    model, history = self._train_kernel_model(
                        feature_map, splits, verbose=False
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Evaluate model
                y_pred_proba = model.predict_proba(splits['X_test'])
                y_pred = model.predict(splits['X_test'])
                
                evaluation = comprehensive_evaluation(
                    splits['y_test'], y_pred_proba, y_pred
                )
                
                training_time = time.time() - start_time
                
                # Get feature importance if available
                feature_importance = None
                if hasattr(model, 'get_feature_importance'):
                    try:
                        feature_importance = model.get_feature_importance(
                            splits['X_test'], splits['y_test']
                        )
                    except:
                        feature_importance = None
                
                # Store results
                feature_map_results[fm_name] = {
                    'evaluation': evaluation,
                    'training_time': training_time,
                    'feature_importance': feature_importance,
                    'model': model,
                    'history': history,
                    'predictions': {
                        'y_pred_proba': y_pred_proba,
                        'y_pred': y_pred
                    }
                }
                
                if verbose:
                    print(f"     Accuracy: {evaluation['accuracy']:.3f}, "
                          f"AUROC: {evaluation['auroc']:.3f}, "
                          f"Time: {training_time:.2f}s")
                    
            except Exception as e:
                if verbose:
                    print(f"     Error: {str(e)}")
                feature_map_results[fm_name] = None
        
        # Step 4: Analyze results
        if verbose:
            print("4. Analyzing results...")
        
        analysis = self._analyze_feature_map_performance(
            feature_map_results, splits['y_test']
        )
        
        # Compile results
        comparison_results = {
            'dataset': dataset_name,
            'model_type': model_type,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'data_splits': {
                'n_train': len(splits['X_train']),
                'n_val': len(splits['X_val']),
                'n_test': len(splits['X_test'])
            },
            'feature_map_results': feature_map_results,
            'analysis': analysis,
            'test_labels': splits['y_test']
        }
        
        # Store results
        key = f"{dataset_name}_{model_type}_feature_maps"
        self.results[key] = comparison_results
        
        # Print summary
        if verbose:
            self._print_comparison_summary(comparison_results)
        
        return comparison_results
    
    def _create_feature_map(self, fm_name: str):
        """Create feature map instance."""
        if fm_name == 'angle':
            return AngleEncoding(n_features=self.n_qubits)
        elif fm_name == 'amplitude':
            return AmplitudeEncoding(n_features=self.n_qubits)
        elif fm_name == 'zz':
            return ZZFeatureMap(
                n_qubits=self.n_qubits,
                reps=2,
                entanglement='linear'
            )
        else:
            raise ValueError(f"Unknown feature map: {fm_name}")
    
    def _train_vqc_model(self, feature_map, splits: Dict[str, np.ndarray], 
                        verbose: bool = False) -> Tuple[VQC, Dict[str, Any]]:
        """Train VQC model."""
        ansatz = HardwareEfficientAnsatz(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            entangling_pattern='linear'
        )
        
        povm = POVM.projective(n_outcomes=2, dim=2**self.n_qubits)
        
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            povm=povm,
            shots=1024
        )
        
        history = vqc.fit(
            splits['X_train'], splits['y_train'],
            X_val=splits['X_val'], y_val=splits['y_val'],
            epochs=25,
            learning_rate=0.1,
            batch_size=32,
            early_stopping_patience=5,
            verbose=verbose
        )
        
        return vqc, history
    
    def _train_kernel_model(self, feature_map, splits: Dict[str, np.ndarray],
                          verbose: bool = False) -> Tuple[QuantumKernel, Dict[str, Any]]:
        """Train Quantum Kernel model."""
        kernel = QuantumKernel(
            feature_map=feature_map,
            kernel_type='fidelity',
            shots=1024,
            random_state=self.random_state
        )
        
        # Fit kernel (no explicit history for kernel methods)
        kernel.fit(splits['X_train'], splits['y_train'])
        
        # Create dummy history
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        return kernel, history
    
    def _analyze_feature_map_performance(self, results: Dict[str, Any], 
                                       y_test: np.ndarray) -> Dict[str, Any]:
        """Analyze feature map performance."""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        # Performance comparison
        performance_metrics = ['accuracy', 'auroc', 'f1_score', 'precision', 'recall']
        performance_comparison = {}
        
        for metric in performance_metrics:
            performance_comparison[metric] = {
                fm: result['evaluation'][metric] 
                for fm, result in valid_results.items()
            }
        
        # Find best performing feature map for each metric
        best_performers = {}
        for metric in performance_metrics:
            best_fm = max(performance_comparison[metric].keys(),
                         key=lambda x: performance_comparison[metric][x])
            best_performers[metric] = {
                'feature_map': best_fm,
                'score': performance_comparison[metric][best_fm]
            }
        
        # Training time comparison
        training_times = {
            fm: result['training_time'] 
            for fm, result in valid_results.items()
        }
        
        fastest_fm = min(training_times.keys(), key=lambda x: training_times[x])
        
        # Overall ranking (weighted average of normalized metrics)
        rankings = self._calculate_overall_rankings(valid_results)
        
        return {
            'performance_comparison': performance_comparison,
            'best_performers': best_performers,
            'training_times': training_times,
            'fastest_feature_map': fastest_fm,
            'overall_rankings': rankings,
            'n_valid_results': len(valid_results)
        }
    
    def _calculate_overall_rankings(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall rankings based on multiple metrics."""
        metrics = ['accuracy', 'auroc', 'f1_score']
        weights = [0.4, 0.4, 0.2]  # Weights for each metric
        
        # Normalize metrics to [0, 1]
        normalized_scores = {}
        
        for metric in metrics:
            scores = [result['evaluation'][metric] for result in results.values()]
            min_score, max_score = min(scores), max(scores)
            
            if max_score > min_score:
                for fm, result in results.items():
                    if fm not in normalized_scores:
                        normalized_scores[fm] = {}
                    normalized_scores[fm][metric] = (
                        (result['evaluation'][metric] - min_score) / 
                        (max_score - min_score)
                    )
            else:
                for fm in results.keys():
                    if fm not in normalized_scores:
                        normalized_scores[fm] = {}
                    normalized_scores[fm][metric] = 1.0
        
        # Calculate weighted average
        overall_scores = {}
        for fm in results.keys():
            overall_scores[fm] = sum(
                normalized_scores[fm][metric] * weight
                for metric, weight in zip(metrics, weights)
            )
        
        # Sort by score (descending)
        sorted_fms = sorted(overall_scores.keys(), 
                           key=lambda x: overall_scores[x], reverse=True)
        
        rankings = {fm: i + 1 for i, fm in enumerate(sorted_fms)}
        rankings['scores'] = overall_scores
        
        return rankings
    
    def _print_comparison_summary(self, results: Dict[str, Any]):
        """Print feature map comparison summary."""
        print("\n" + "="*70)
        print("FEATURE MAP COMPARISON SUMMARY")
        print("="*70)
        
        fm_results = results['feature_map_results']
        analysis = results['analysis']
        
        print(f"Dataset: {results['dataset']}")
        print(f"Model Type: {results['model_type'].upper()}")
        print(f"Data splits: {results['data_splits']['n_train']}/"
              f"{results['data_splits']['n_val']}/"
              f"{results['data_splits']['n_test']} (train/val/test)")
        
        # Performance table
        print("\nPerformance Comparison:")
        valid_results = {k: v for k, v in fm_results.items() if v is not None}
        
        if valid_results:
            print(f"{'Feature Map':<12} {'Accuracy':<9} {'AUROC':<8} {'F1':<8} {'Time(s)':<8}")
            print("-" * 50)
            
            for fm, result in valid_results.items():
                eval_res = result['evaluation']
                time_taken = result['training_time']
                print(f"{fm:<12} {eval_res['accuracy']:<9.3f} {eval_res['auroc']:<8.3f} "
                      f"{eval_res['f1_score']:<8.3f} {time_taken:<8.2f}")
        
        # Best performers
        if 'best_performers' in analysis:
            print("\nBest Performers:")
            for metric, best in analysis['best_performers'].items():
                print(f"  {metric.title()}: {best['feature_map']} ({best['score']:.3f})")
        
        # Overall ranking
        if 'overall_rankings' in analysis:
            rankings = analysis['overall_rankings']
            print("\nOverall Rankings:")
            sorted_fms = sorted([fm for fm in rankings.keys() if fm != 'scores'],
                               key=lambda x: rankings[x])
            
            for i, fm in enumerate(sorted_fms):
                score = rankings['scores'][fm]
                print(f"  {i+1}. {fm} (score: {score:.3f})")
        
        print(f"\nFastest training: {analysis.get('fastest_feature_map', 'N/A')}")
    
    def run_comprehensive_comparison(self, 
                                   datasets: List[str] = ['moons', 'circles', 'xor'],
                                   feature_maps: List[str] = ['angle', 'amplitude', 'zz'],
                                   model_types: List[str] = ['vqc', 'kernel'],
                                   verbose: bool = True) -> Dict[str, Any]:
        """Run comprehensive feature map comparison.
        
        Args:
            datasets: List of datasets to test
            feature_maps: List of feature maps to test
            model_types: List of model types to test
            verbose: Whether to print progress
            
        Returns:
            Dictionary with all comparison results
        """
        if verbose:
            print("\n" + "="*70)
            print("COMPREHENSIVE FEATURE MAP COMPARISON")
            print("="*70)
        
        comprehensive_results = {}
        
        for dataset in datasets:
            comprehensive_results[dataset] = {}
            
            for model_type in model_types:
                if verbose:
                    print(f"\nTesting {model_type.upper()} on {dataset}...")
                
                try:
                    result = self.compare_feature_maps(
                        dataset_name=dataset,
                        feature_maps=feature_maps,
                        model_type=model_type,
                        verbose=False
                    )
                    
                    comprehensive_results[dataset][model_type] = result
                    
                    if verbose:
                        analysis = result['analysis']
                        if 'overall_rankings' in analysis:
                            best_fm = min(analysis['overall_rankings'].keys(),
                                         key=lambda x: analysis['overall_rankings'][x] 
                                         if x != 'scores' else float('inf'))
                            best_score = analysis['overall_rankings']['scores'][best_fm]
                            print(f"  Best feature map: {best_fm} (score: {best_score:.3f})")
                        
                except Exception as e:
                    if verbose:
                        print(f"  Error: {str(e)}")
                    comprehensive_results[dataset][model_type] = None
        
        # Cross-analysis
        if verbose:
            print("\nCross-Analysis Summary:")
            self._print_cross_analysis(comprehensive_results)
        
        return comprehensive_results
    
    def _print_cross_analysis(self, results: Dict[str, Any]):
        """Print cross-analysis of results."""
        # Count wins for each feature map
        feature_map_wins = {}
        total_experiments = 0
        
        for dataset, dataset_results in results.items():
            for model_type, model_results in dataset_results.items():
                if model_results is not None and 'analysis' in model_results:
                    analysis = model_results['analysis']
                    if 'overall_rankings' in analysis:
                        rankings = analysis['overall_rankings']
                        best_fm = min([fm for fm in rankings.keys() if fm != 'scores'],
                                     key=lambda x: rankings[x])
                        
                        if best_fm not in feature_map_wins:
                            feature_map_wins[best_fm] = 0
                        feature_map_wins[best_fm] += 1
                        total_experiments += 1
        
        if total_experiments > 0:
            print(f"\nFeature Map Win Rates (out of {total_experiments} experiments):")
            for fm, wins in sorted(feature_map_wins.items(), 
                                  key=lambda x: x[1], reverse=True):
                win_rate = wins / total_experiments * 100
                print(f"  {fm}: {wins} wins ({win_rate:.1f}%)")
    
    def visualize_comparison_results(self, experiment_key: str, 
                                   save_plots: bool = False, 
                                   plot_dir: str = "plots"):
        """Visualize feature map comparison results.
        
        Args:
            experiment_key: Key of experiment to visualize
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
        """
        if experiment_key not in self.results:
            raise ValueError(f"No results found for experiment: {experiment_key}")
        
        results = self.results[experiment_key]
        fm_results = results['feature_map_results']
        
        # Prepare data for visualization
        valid_results = {k: v for k, v in fm_results.items() if v is not None}
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        # Plot model comparison
        try:
            models_data = []
            for fm, result in valid_results.items():
                eval_res = result['evaluation']
                models_data.append({
                    'name': fm,
                    'accuracy': eval_res['accuracy'],
                    'auroc': eval_res['auroc'],
                    'f1_score': eval_res['f1_score'],
                    'training_time': result['training_time']
                })
            
            plot_model_comparison(
                models_data,
                title=f"Feature Map Comparison ({results['dataset']}, {results['model_type'].upper()})",
                save_path=f"{plot_dir}/feature_map_comparison_{experiment_key}.png" if save_plots else None
            )
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Visualization error: {str(e)}")
    
    def get_feature_map_recommendations(self, 
                                      dataset_characteristics: Dict[str, Any]) -> Dict[str, str]:
        """Get feature map recommendations based on dataset characteristics.
        
        Args:
            dataset_characteristics: Dictionary with dataset info
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {}
        
        n_features = dataset_characteristics.get('n_features', 4)
        n_samples = dataset_characteristics.get('n_samples', 1000)
        complexity = dataset_characteristics.get('complexity', 'medium')
        noise_level = dataset_characteristics.get('noise_level', 'low')
        
        # General recommendations
        if n_features <= 4:
            if complexity == 'low':
                recommendations['primary'] = 'angle'
                recommendations['reason'] = 'Simple angle encoding works well for low-complexity, low-dimensional data'
            elif complexity == 'high':
                recommendations['primary'] = 'zz'
                recommendations['reason'] = 'ZZ feature map provides better expressivity for complex patterns'
            else:
                recommendations['primary'] = 'amplitude'
                recommendations['reason'] = 'Amplitude encoding offers good balance of expressivity and efficiency'
        else:
            recommendations['primary'] = 'amplitude'
            recommendations['reason'] = 'Amplitude encoding is more efficient for higher-dimensional data'
        
        # Secondary recommendation
        if recommendations['primary'] == 'angle':
            recommendations['secondary'] = 'zz'
        elif recommendations['primary'] == 'zz':
            recommendations['secondary'] = 'amplitude'
        else:
            recommendations['secondary'] = 'angle'
        
        # Noise considerations
        if noise_level == 'high':
            recommendations['note'] = 'Consider using more shots or noise-robust feature maps'
        
        return recommendations


def run_feature_map_demo():
    """Run a basic feature map comparison demo."""
    comparison = FeatureMapComparison(n_qubits=4, n_layers=2)
    
    # Run comparison
    result = comparison.compare_feature_maps(
        dataset_name='moons',
        feature_maps=['angle', 'amplitude', 'zz'],
        model_type='vqc',
        verbose=True
    )
    
    # Visualize results
    try:
        comparison.visualize_comparison_results('moons_vqc_feature_maps')
    except ImportError:
        print("\nVisualization skipped (matplotlib not available)")
    
    return result


def run_comprehensive_feature_map_comparison():
    """Run comprehensive feature map comparison."""
    comparison = FeatureMapComparison(n_qubits=4, n_layers=2)
    
    # Run comprehensive comparison
    results = comparison.run_comprehensive_comparison(
        datasets=['moons', 'circles'],
        feature_maps=['angle', 'amplitude'],
        model_types=['vqc'],
        verbose=True
    )
    
    return results


if __name__ == "__main__":
    # Run basic demo
    print("Running feature map comparison demo...")
    run_feature_map_demo()
    
    print("\n" + "="*70)
    
    # Run comprehensive comparison
    print("Running comprehensive comparison...")
    run_comprehensive_feature_map_comparison()