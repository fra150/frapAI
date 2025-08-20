"""Evaluation metrics for FPAI framework.

Implements calibration metrics, fairness measures, and standard ML metrics
for quantum machine learning models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss
)


def calibration_curve(y_prob: np.ndarray, y_true: np.ndarray, 
                     n_bins: int = 10, strategy: str = 'uniform') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve for binary classification.
    
    Args:
        y_prob: Predicted probabilities for positive class
        y_true: True binary labels
        n_bins: Number of bins
        strategy: Binning strategy ('uniform' or 'quantile')
        
    Returns:
        Tuple of (bin_centers, fraction_positives, bin_counts)
    """
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_boundaries = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
        
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    fraction_positives = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_center = (bin_lower + bin_upper) / 2
            frac_pos = np.mean(y_true[in_bin])
            count = np.sum(in_bin)
            
            bin_centers.append(bin_center)
            fraction_positives.append(frac_pos)
            bin_counts.append(count)
    
    return np.array(bin_centers), np.array(fraction_positives), np.array(bin_counts)


def expected_calibration_error(y_prob: np.ndarray, y_true: np.ndarray, 
                             n_bins: int = 10, strategy: str = 'uniform') -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy.
    
    Args:
        y_prob: Predicted probabilities [n_samples, n_classes]
        y_true: True labels [n_samples]
        n_bins: Number of confidence bins
        strategy: Binning strategy ('uniform' or 'quantile')
        
    Returns:
        ECE score (lower is better)
    """
    # Get predicted class and confidence
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    
    # Create bins
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_boundaries = np.quantile(confidences, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
        
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Compute accuracy and confidence in this bin
            accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Add weighted difference to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def maximum_calibration_error(y_prob: np.ndarray, y_true: np.ndarray,
                            n_bins: int = 10, strategy: str = 'uniform') -> float:
    """Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between confidence and accuracy across all bins.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        n_bins: Number of confidence bins
        strategy: Binning strategy
        
    Returns:
        MCE score (lower is better)
    """
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    
    # Create bins
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_boundaries = np.quantile(confidences, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
        
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_error = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_error = max(max_error, error)
            
    return max_error


def reliability_diagram(y_prob: np.ndarray, y_true: np.ndarray,
                       n_bins: int = 10, strategy: str = 'uniform') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability diagram data.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        n_bins: Number of bins
        strategy: Binning strategy
        
    Returns:
        (bin_centers, accuracies, confidences): Data for plotting reliability diagram
    """
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    
    # Create bins
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_boundaries = np.quantile(confidences, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
        
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy = (y_pred[in_bin] == y_true[in_bin]).mean()
            confidence = confidences[in_bin].mean()
        else:
            accuracy = 0.0
            confidence = bin_centers[i]
            
        bin_accuracies.append(accuracy)
        bin_confidences.append(confidence)
        
    return bin_centers, np.array(bin_accuracies), np.array(bin_confidences)


def brier_score(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Brier score.
    
    Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes.
    
    Args:
        y_prob: Predicted probabilities [n_samples, n_classes]
        y_true: True labels [n_samples]
        
    Returns:
        Brier score (lower is better)
    """
    n_classes = y_prob.shape[1]
    
    # Convert labels to one-hot encoding
    y_one_hot = np.zeros((len(y_true), n_classes))
    y_one_hot[np.arange(len(y_true)), y_true] = 1
    
    # Compute Brier score
    brier = np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1))
    
    return brier


def negative_log_likelihood(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Compute negative log-likelihood.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        
    Returns:
        NLL score (lower is better)
    """
    # Extract probabilities for true classes
    true_probs = y_prob[np.arange(len(y_true)), y_true]
    
    # Clip to avoid log(0)
    true_probs = np.clip(true_probs, 1e-12, 1.0)
    
    return -np.mean(np.log(true_probs))


def entropy_of_predictions(y_prob: np.ndarray) -> float:
    """Compute average entropy of predictions.
    
    High entropy indicates high uncertainty.
    
    Args:
        y_prob: Predicted probabilities
        
    Returns:
        Average entropy
    """
    # Clip probabilities
    y_prob_clipped = np.clip(y_prob, 1e-12, 1.0)
    
    # Compute entropy for each sample
    entropies = -np.sum(y_prob_clipped * np.log(y_prob_clipped), axis=1)
    
    return np.mean(entropies)


def mutual_information(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Compute mutual information between predictions and true labels.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        
    Returns:
        Mutual information score
    """
    n_classes = y_prob.shape[1]
    
    # Compute class probabilities
    class_probs = np.bincount(y_true, minlength=n_classes) / len(y_true)
    
    # Compute conditional entropy H(Y|X)
    conditional_entropy = 0.0
    for class_idx in range(n_classes):
        class_mask = (y_true == class_idx)
        if class_mask.sum() > 0:
            class_pred_probs = y_prob[class_mask]
            class_entropy = entropy_of_predictions(class_pred_probs)
            conditional_entropy += class_probs[class_idx] * class_entropy
            
    # Compute marginal entropy H(Y)
    marginal_entropy = -np.sum(class_probs * np.log(np.clip(class_probs, 1e-12, 1.0)))
    
    # Mutual information = H(Y) - H(Y|X)
    return marginal_entropy - conditional_entropy


def fairness_metrics(y_prob: np.ndarray, y_true: np.ndarray, 
                    sensitive_attr: np.ndarray) -> Dict[str, float]:
    """Compute fairness metrics across sensitive attributes.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        sensitive_attr: Sensitive attribute values (e.g., gender, race)
        
    Returns:
        Dictionary of fairness metrics
    """
    y_pred = np.argmax(y_prob, axis=1)
    unique_groups = np.unique(sensitive_attr)
    
    # Compute metrics for each group
    group_metrics = {}
    for group in unique_groups:
        group_mask = (sensitive_attr == group)
        group_metrics[group] = {
            'accuracy': accuracy_score(y_true[group_mask], y_pred[group_mask]),
            'precision': precision_score(y_true[group_mask], y_pred[group_mask], average='weighted', zero_division=0),
            'recall': recall_score(y_true[group_mask], y_pred[group_mask], average='weighted', zero_division=0),
            'f1': f1_score(y_true[group_mask], y_pred[group_mask], average='weighted', zero_division=0)
        }
        
        # Add AUC for binary classification
        if y_prob.shape[1] == 2:
            try:
                group_metrics[group]['auc'] = roc_auc_score(y_true[group_mask], y_prob[group_mask, 1])
            except ValueError:
                group_metrics[group]['auc'] = 0.0
                
    # Compute fairness metrics
    fairness = {}
    
    # Demographic parity: P(Y_hat = 1 | A = 0) = P(Y_hat = 1 | A = 1)
    if len(unique_groups) == 2:
        group0, group1 = unique_groups
        pos_rate_0 = np.mean(y_pred[sensitive_attr == group0])
        pos_rate_1 = np.mean(y_pred[sensitive_attr == group1])
        fairness['demographic_parity_diff'] = abs(pos_rate_0 - pos_rate_1)
        
        # Equalized odds: TPR and FPR should be equal across groups
        tpr_0 = group_metrics[group0]['recall']
        tpr_1 = group_metrics[group1]['recall']
        fairness['equalized_odds_tpr_diff'] = abs(tpr_0 - tpr_1)
        
        # Equal opportunity: TPR should be equal across groups
        fairness['equal_opportunity_diff'] = abs(tpr_0 - tpr_1)
        
        # Accuracy parity
        acc_0 = group_metrics[group0]['accuracy']
        acc_1 = group_metrics[group1]['accuracy']
        fairness['accuracy_parity_diff'] = abs(acc_0 - acc_1)
        
    # Overall fairness score (lower is more fair)
    fairness_scores = [v for k, v in fairness.items() if 'diff' in k]
    fairness['overall_fairness'] = np.mean(fairness_scores) if fairness_scores else 0.0
    
    return {
        'group_metrics': group_metrics,
        'fairness_metrics': fairness
    }


def calibration_curve(y_prob: np.ndarray, y_true: np.ndarray,
                     n_bins: int = 10, strategy: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve.
    
    Args:
        y_prob: Predicted probabilities (for positive class in binary case)
        y_true: True binary labels
        n_bins: Number of bins
        strategy: Binning strategy
        
    Returns:
        (fraction_of_positives, mean_predicted_value): Calibration curve data
    """
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        # Multi-class: use probability of predicted class
        y_pred = np.argmax(y_prob, axis=1)
        y_prob = y_prob[np.arange(len(y_prob)), y_pred]
        y_true = (y_pred == y_true).astype(int)
    elif y_prob.ndim > 1:
        y_prob = y_prob[:, 1]  # Positive class probability
        
    # Create bins
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_boundaries = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
        
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    fraction_of_positives = []
    mean_predicted_values = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            fraction_pos = y_true[in_bin].mean()
            mean_pred = y_prob[in_bin].mean()
        else:
            fraction_pos = 0.0
            mean_pred = bin_centers[i]
            
        fraction_of_positives.append(fraction_pos)
        mean_predicted_values.append(mean_pred)
        
    return np.array(fraction_of_positives), np.array(mean_predicted_values)


def comprehensive_evaluation(y_prob: np.ndarray, y_true: np.ndarray,
                           sensitive_attr: Optional[np.ndarray] = None) -> Dict[str, Union[float, Dict]]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        sensitive_attr: Sensitive attributes for fairness evaluation
        
    Returns:
        Dictionary of all evaluation metrics
    """
    y_pred = np.argmax(y_prob, axis=1)
    
    metrics = {
        # Standard classification metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        
        # Probabilistic metrics
        'nll': negative_log_likelihood(y_prob, y_true),
        'brier_score': brier_score(y_prob, y_true),
        
        # Calibration metrics
        'ece': expected_calibration_error(y_prob, y_true),
        'mce': maximum_calibration_error(y_prob, y_true),
        
        # Uncertainty metrics
        'entropy': entropy_of_predictions(y_prob),
        'mutual_information': mutual_information(y_prob, y_true)
    }
    
    # Add AUC for binary/multi-class
    try:
        if y_prob.shape[1] == 2:
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            metrics['ap'] = average_precision_score(y_true, y_prob[:, 1])
        else:
            metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        metrics['auc'] = 0.0
        metrics['ap'] = 0.0
        
    # Add fairness metrics if sensitive attributes provided
    if sensitive_attr is not None:
        metrics['fairness'] = fairness_metrics(y_prob, y_true, sensitive_attr)
        
    return metrics


def quantum_specific_metrics(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute quantum-specific evaluation metrics.
    
    Args:
        model: Trained FPAI model
        X: Input features
        y: True labels
        
    Returns:
        Dictionary of quantum-specific metrics
    """
    metrics = {}
    
    # Circuit depth and complexity
    if hasattr(model, 'ansatz'):
        metrics['circuit_depth'] = model.ansatz.get_circuit_depth()
        metrics['n_parameters'] = model.ansatz.n_parameters()
        metrics['n_qubits'] = model.ansatz.n_qubits
        
    # Feature map properties
    if hasattr(model, 'feature_map'):
        metrics['feature_map_type'] = type(model.feature_map).__name__
        
    # Quantum kernel properties (if applicable)
    if hasattr(model, 'kernel_matrix_analysis'):
        try:
            kernel_analysis = model.kernel_matrix_analysis(X)
            metrics.update({
                f'kernel_{k}': v for k, v in kernel_analysis.items()
            })
        except:
            pass
            
    # Training convergence (if available)
    if hasattr(model, 'training_history') and model.training_history:
        history = model.training_history
        metrics['final_train_loss'] = history[-1].get('train_loss', 0.0)
        metrics['final_val_loss'] = history[-1].get('val_loss', 0.0)
        metrics['n_epochs'] = len(history)
        
        # Convergence rate
        if len(history) > 10:
            early_loss = np.mean([h['train_loss'] for h in history[:10]])
            late_loss = np.mean([h['train_loss'] for h in history[-10:]])
            metrics['loss_improvement'] = early_loss - late_loss
            
    return metrics