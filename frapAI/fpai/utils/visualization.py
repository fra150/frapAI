"""Visualization utilities for FPAI framework.

Provides plotting functions for calibration curves, quantum states,
training history, and model analysis.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization. Install with: pip install matplotlib")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Optional[List[str]] = None,
                         normalize: Optional[str] = None,
                         title: str = 'Confusion Matrix',
                         cmap: str = 'Blues',
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (8, 6)):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        title: Plot title
        cmap: Colormap
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ax


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model,
                          feature_indices: Tuple[int, int] = (0, 1),
                          resolution: int = 100,
                          title: str = 'Decision Boundary',
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)):
    """Plot decision boundary for 2D data.
    
    Args:
        X: Feature matrix
        y: Labels
        model: Trained model with predict method
        feature_indices: Which features to plot
        resolution: Grid resolution
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    # Extract the two features
    X_plot = X[:, feature_indices]
    
    # Create a mesh
    h = (X_plot.max(axis=0) - X_plot.min(axis=0)) / resolution
    xx, yy = np.meshgrid(
        np.arange(X_plot[:, 0].min() - h[0], X_plot[:, 0].max() + h[0], h[0]),
        np.arange(X_plot[:, 1].min() - h[1], X_plot[:, 1].max() + h[1], h[1])
    )
    
    # Create full feature matrix for prediction
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    if X.shape[1] > 2:
        # For higher dimensional data, use mean values for other features
        other_features = np.tile(X.mean(axis=0), (mesh_points.shape[0], 1))
        other_features[:, feature_indices] = mesh_points
        mesh_points = other_features
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(mesh_points)[:, 1]
    else:
        Z = model.predict(mesh_points)
    
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot decision boundary
    contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    
    # Plot data points
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    
    ax.set_xlabel(f'Feature {feature_indices[0]}')
    ax.set_ylabel(f'Feature {feature_indices[1]}')
    ax.set_title(title)
    
    plt.colorbar(contour, ax=ax, label='Prediction')
    plt.colorbar(scatter, ax=ax, label='True Class')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ax


def plot_calibration_curve(y_prob: np.ndarray, y_true: np.ndarray,
                          n_bins: int = 10, strategy: str = 'uniform',
                          title: str = 'Calibration Curve',
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 6)):
    """Plot calibration curve.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        n_bins: Number of bins
        strategy: Binning strategy
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    from .metrics import calibration_curve
    
    # Compute calibration curve
    fraction_pos, mean_pred = calibration_curve(y_prob, y_true, n_bins, strategy)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot calibration curve
    ax.plot(mean_pred, fraction_pos, 'o-', label='Model', linewidth=2, markersize=6)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_reliability_diagram(y_prob: np.ndarray, y_true: np.ndarray,
                           n_bins: int = 10, strategy: str = 'uniform',
                           title: str = 'Reliability Diagram',
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6)):
    """Plot reliability diagram with histogram.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        n_bins: Number of bins
        strategy: Binning strategy
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    from .metrics import reliability_diagram, expected_calibration_error
    
    # Compute reliability data
    bin_centers, accuracies, confidences = reliability_diagram(y_prob, y_true, n_bins, strategy)
    ece = expected_calibration_error(y_prob, y_true, n_bins, strategy)
    
    # Get confidence values for histogram
    if y_prob.ndim > 1:
        conf_values = np.max(y_prob, axis=1)
    else:
        conf_values = y_prob
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Top plot: Reliability diagram
    ax1.bar(bin_centers, accuracies, width=1.0/n_bins, alpha=0.7, 
           edgecolor='black', label='Accuracy')
    ax1.plot(bin_centers, confidences, 'ro-', label='Confidence', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
    
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{title} (ECE = {ece:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Bottom plot: Confidence histogram
    ax2.hist(conf_values, bins=n_bins, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_quantum_state(state, title: str = 'Quantum State',
                      representation: str = 'bloch',
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (8, 6)):
    """Plot quantum state visualization.
    
    Args:
        state: QuantumState object
        title: Plot title
        representation: Visualization type ('bloch', 'density', 'amplitudes')
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    if representation == 'bloch' and state.n_qubits == 1:
        _plot_bloch_sphere(state, title, save_path, figsize)
    elif representation == 'density':
        _plot_density_matrix(state, title, save_path, figsize)
    elif representation == 'amplitudes':
        _plot_state_amplitudes(state, title, save_path, figsize)
    else:
        raise ValueError(f"Unsupported representation '{representation}' for {state.n_qubits}-qubit state")


def _plot_bloch_sphere(state, title: str, save_path: Optional[str], figsize: Tuple[int, int]):
    """Plot single qubit state on Bloch sphere."""
    if state.n_qubits != 1:
        raise ValueError("Bloch sphere visualization only supports single qubits")
    
    # Extract Bloch vector
    rho = state.density_matrix
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    # Compute Bloch vector components
    x = np.real(np.trace(rho @ sigma_x))
    y = np.real(np.trace(rho @ sigma_y))
    z = np.real(np.trace(rho @ sigma_z))
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Bloch sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.1, color='lightblue')
    
    # Draw coordinate axes
    ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.3)
    
    # Draw state vector
    ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
    
    # Labels
    ax.text(1.2, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.2, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.2, 'Z', fontsize=12)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def _plot_density_matrix(state, title: str, save_path: Optional[str], figsize: Tuple[int, int]):
    """Plot density matrix heatmap."""
    rho = state.density_matrix
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
    
    # Real part
    im1 = ax1.imshow(np.real(rho), cmap='RdBu', vmin=-1, vmax=1)
    ax1.set_title('Real Part')
    ax1.set_xlabel('Basis State')
    ax1.set_ylabel('Basis State')
    plt.colorbar(im1, ax=ax1)
    
    # Imaginary part
    im2 = ax2.imshow(np.imag(rho), cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_title('Imaginary Part')
    ax2.set_xlabel('Basis State')
    ax2.set_ylabel('Basis State')
    plt.colorbar(im2, ax=ax2)
    
    fig.suptitle(f'{title} - Density Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def _plot_state_amplitudes(state, title: str, save_path: Optional[str], figsize: Tuple[int, int]):
    """Plot state amplitudes."""
    if state.is_pure:
        amplitudes = state.state_vector
        n_states = len(amplitudes)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Amplitude magnitudes
        x = range(n_states)
        ax1.bar(x, np.abs(amplitudes)**2, alpha=0.7)
        ax1.set_ylabel('Probability')
        ax1.set_title('Probability Amplitudes')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'|{i:0{state.n_qubits}b}⟩' for i in x])
        
        # Phases
        phases = np.angle(amplitudes)
        ax2.bar(x, phases, alpha=0.7, color='orange')
        ax2.set_ylabel('Phase (rad)')
        ax2.set_xlabel('Basis State')
        ax2.set_title('Phases')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'|{i:0{state.n_qubits}b}⟩' for i in x])
        
        fig.suptitle(title)
        plt.tight_layout()
    else:
        # For mixed states, plot diagonal elements
        rho = state.density_matrix
        probs = np.real(np.diag(rho))
        
        fig, ax = plt.subplots(figsize=figsize)
        x = range(len(probs))
        ax.bar(x, probs, alpha=0.7)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Basis State')
        ax.set_title(f'{title} - Population Probabilities')
        ax.set_xticks(x)
        ax.set_xticklabels([f'|{i:0{state.n_qubits}b}⟩' for i in x])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history: List[Dict[str, Any]], 
                         title: str = 'Training History',
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 4)):
    """Plot training history.
    
    Args:
        history: List of training history dictionaries
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    if not history:
        print("No training history available")
        return
    
    epochs = [h['epoch'] for h in history]
    train_losses = [h.get('train_loss', 0) for h in history]
    val_losses = [h.get('val_loss', None) for h in history]
    
    # Filter out None values
    val_losses = [v for v in val_losses if v is not None]
    has_validation = len(val_losses) > 0
    
    if has_validation:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    # Training loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if has_validation:
        val_epochs = [h['epoch'] for h in history if h.get('val_loss') is not None]
        ax1.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if has_validation:
        # Validation metrics (if available)
        val_accs = [h.get('val_acc', None) for h in history]
        val_accs = [v for v in val_accs if v is not None]
        
        if val_accs:
            val_acc_epochs = [h['epoch'] for h in history if h.get('val_acc') is not None]
            ax2.plot(val_acc_epochs, val_accs, 'g-', label='Validation Accuracy', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Plot loss comparison
            ax2.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
            ax2.plot(val_epochs, val_losses, 'r-', label='Validation', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(importance: np.ndarray, feature_names: Optional[List[str]] = None,
                          title: str = 'Feature Importance',
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6)):
    """Plot feature importance.
    
    Args:
        importance: Feature importance scores
        feature_names: Names of features
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_importance, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix',
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (8, 6)):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_quantum_circuit(ansatz, title: str = 'Quantum Circuit',
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 6)):
    """Plot quantum circuit diagram.
    
    Args:
        ansatz: Ansatz object
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_qubits = ansatz.n_qubits
    n_layers = ansatz.n_layers
    
    # Draw qubit lines
    for i in range(n_qubits):
        ax.plot([0, n_layers + 1], [i, i], 'k-', linewidth=2)
        ax.text(-0.2, i, f'q{i}', ha='right', va='center', fontsize=12)
    
    # Draw gates (simplified representation)
    gate_width = 0.3
    gate_height = 0.2
    
    for layer in range(n_layers):
        x_pos = layer + 0.5
        
        # Single-qubit gates
        for qubit in range(n_qubits):
            rect = patches.Rectangle(
                (x_pos - gate_width/2, qubit - gate_height/2),
                gate_width, gate_height,
                linewidth=1, edgecolor='blue', facecolor='lightblue'
            )
            ax.add_patch(rect)
            ax.text(x_pos, qubit, 'R', ha='center', va='center', fontsize=10)
        
        # Entangling gates
        if hasattr(ansatz, 'entangling_pattern'):
            if ansatz.entangling_pattern == 'linear':
                for i in range(n_qubits - 1):
                    # Draw CNOT
                    ax.plot(x_pos + 0.2, i, 'ko', markersize=8)
                    ax.plot([x_pos + 0.2, x_pos + 0.2], [i, i + 1], 'k-', linewidth=2)
                    ax.plot(x_pos + 0.2, i + 1, 'ko', markersize=12, fillstyle='none')
    
    ax.set_xlim([-0.5, n_layers + 1.5])
    ax.set_ylim([-0.5, n_qubits - 0.5])
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]],
                         metrics: List[str] = ['accuracy', 'ece', 'nll'],
                         title: str = 'Model Comparison',
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8)):
    """Plot comparison of multiple models.
    
    Args:
        results: Dictionary of model results
        metrics: List of metrics to compare
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    _check_matplotlib()
    
    n_metrics = len(metrics)
    n_models = len(results)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    model_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in model_names]
        
        bars = axes[i].bar(model_names, values, color=colors, alpha=0.7)
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(max(model_names, key=len)) > 8:
            axes[i].tick_params(axis='x', rotation=45)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()