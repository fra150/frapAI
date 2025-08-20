"""Quantum feature maps for classical-to-quantum embedding."""

import numpy as np
from typing import Union, Callable, Optional, List
from abc import ABC, abstractmethod
from .quantum_state import QuantumState


class FeatureMap(ABC):
    """Abstract base class for quantum feature maps Φ: R^m → D(H).
    
    Maps classical data to quantum states on Hilbert space.
    """
    
    def __init__(self, n_features: int, n_qubits: int):
        """Initialize feature map.
        
        Args:
            n_features: Number of classical features
            n_qubits: Number of qubits in quantum state
        """
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
    @abstractmethod
    def embed(self, x: np.ndarray) -> QuantumState:
        """Embed classical data into quantum state.
        
        Args:
            x: Classical feature vector
            
        Returns:
            Quantum state |ψ(x)⟩ or ρ(x)
        """
        pass
        
    def embed_batch(self, X: np.ndarray) -> List[QuantumState]:
        """Embed batch of classical data.
        
        Args:
            X: Batch of feature vectors (n_samples, n_features)
            
        Returns:
            List of quantum states
        """
        return [self.embed(x) for x in X]
        
    def kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute quantum kernel matrix K(x,x') = |⟨ψ(x)|ψ(x')⟩|².
        
        Args:
            X: First set of samples
            Y: Second set of samples (default: X)
            
        Returns:
            Kernel matrix
        """
        if Y is None:
            Y = X
            
        states_X = self.embed_batch(X)
        states_Y = self.embed_batch(Y)
        
        K = np.zeros((len(X), len(Y)))
        for i, state_x in enumerate(states_X):
            for j, state_y in enumerate(states_Y):
                K[i, j] = state_x.fidelity(state_y)
                
        return K


class AngleEncoding(FeatureMap):
    """Angle encoding feature map.
    
    Encodes features as rotation angles: |ψ(x)⟩ = ⊗_i R_y(x_i)|0⟩
    """
    
    def __init__(self, n_features: int, scaling: float = 1.0):
        """Initialize angle encoding.
        
        Args:
            n_features: Number of features (= number of qubits)
            scaling: Scaling factor for angles
        """
        super().__init__(n_features, n_features)
        self.scaling = scaling
        
    def embed(self, x: np.ndarray) -> QuantumState:
        """Embed using angle encoding.
        
        Args:
            x: Feature vector
            
        Returns:
            Quantum state with angle-encoded features
        """
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")
            
        # Start with |0⟩^⊗n
        state_vec = np.zeros(self.dim, dtype=complex)
        state_vec[0] = 1.0
        state = QuantumState(state_vec, is_pure=True)
        
        # Apply rotation gates
        for i, feature in enumerate(x):
            angle = self.scaling * feature
            ry_gate = self._ry_gate(angle)
            
            # Create full unitary for qubit i
            unitary = self._single_qubit_unitary(ry_gate, i)
            state = state.evolve_unitary(unitary)
            
        return state
        
    def _ry_gate(self, angle: float) -> np.ndarray:
        """Y-rotation gate.
        
        Args:
            angle: Rotation angle
            
        Returns:
            2x2 rotation matrix
        """
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        return np.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=complex)
        
    def _single_qubit_unitary(self, gate: np.ndarray, qubit_idx: int) -> np.ndarray:
        """Construct full unitary for single-qubit gate.
        
        Args:
            gate: 2x2 single-qubit gate
            qubit_idx: Index of target qubit
            
        Returns:
            Full unitary matrix
        """
        if qubit_idx >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_idx} >= {self.n_qubits}")
            
        # Tensor product construction
        unitary = np.array([[1.0]], dtype=complex)
        
        for i in range(self.n_qubits):
            if i == qubit_idx:
                unitary = np.kron(unitary, gate)
            else:
                unitary = np.kron(unitary, np.eye(2))
                
        return unitary


class AmplitudeEncoding(FeatureMap):
    """Amplitude encoding feature map.
    
    Encodes features directly as amplitudes: |ψ(x)⟩ = Σ_i x_i|i⟩ (normalized)
    """
    
    def __init__(self, n_features: int, padding: str = 'zero'):
        """Initialize amplitude encoding.
        
        Args:
            n_features: Number of features
            padding: Padding strategy ('zero', 'repeat')
        """
        n_qubits = int(np.ceil(np.log2(n_features)))
        super().__init__(n_features, n_qubits)
        self.padding = padding
        
    def embed(self, x: np.ndarray) -> QuantumState:
        """Embed using amplitude encoding.
        
        Args:
            x: Feature vector
            
        Returns:
            Quantum state with amplitude-encoded features
        """
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")
            
        # Pad to power of 2
        amplitudes = np.zeros(self.dim, dtype=complex)
        amplitudes[:len(x)] = x
        
        if self.padding == 'repeat' and len(x) < self.dim:
            # Repeat pattern
            for i in range(len(x), self.dim):
                amplitudes[i] = x[i % len(x)]
                
        # Normalize
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes /= norm
        else:
            amplitudes[0] = 1.0  # Default to |0⟩
            
        return QuantumState(amplitudes, is_pure=True)


class PhaseEncoding(FeatureMap):
    """Phase encoding feature map.
    
    Encodes features as phases: |ψ(x)⟩ = (1/√2^n) Σ_i e^(i φ(x,i))|i⟩
    """
    
    def __init__(self, n_features: int, n_qubits: int, 
                 phase_func: Optional[Callable] = None):
        """Initialize phase encoding.
        
        Args:
            n_features: Number of features
            n_qubits: Number of qubits
            phase_func: Function to compute phases φ(x,i)
        """
        super().__init__(n_features, n_qubits)
        
        if phase_func is None:
            # Default: linear phase function
            self.phase_func = lambda x, i: np.sum(x) * i / self.dim
        else:
            self.phase_func = phase_func
            
    def embed(self, x: np.ndarray) -> QuantumState:
        """Embed using phase encoding.
        
        Args:
            x: Feature vector
            
        Returns:
            Quantum state with phase-encoded features
        """
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")
            
        # Equal superposition with phases
        amplitudes = np.zeros(self.dim, dtype=complex)
        
        for i in range(self.dim):
            phase = self.phase_func(x, i)
            amplitudes[i] = np.exp(1j * phase) / np.sqrt(self.dim)
            
        return QuantumState(amplitudes, is_pure=True)


class ParametricFeatureMap(FeatureMap):
    """Parametric feature map with trainable parameters.
    
    General form: |ψ(x,θ)⟩ = U(x,θ)|0⟩^⊗n
    """
    
    def __init__(self, n_features: int, n_qubits: int, 
                 circuit_func: Callable, n_params: int):
        """Initialize parametric feature map.
        
        Args:
            n_features: Number of features
            n_qubits: Number of qubits
            circuit_func: Function that builds circuit U(x,θ)
            n_params: Number of trainable parameters
        """
        super().__init__(n_features, n_qubits)
        self.circuit_func = circuit_func
        self.n_params = n_params
        self.params = np.random.normal(0, 0.1, n_params)
        
    def embed(self, x: np.ndarray) -> QuantumState:
        """Embed using parametric circuit.
        
        Args:
            x: Feature vector
            
        Returns:
            Quantum state from parametric circuit
        """
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")
            
        # Start with |0⟩^⊗n
        state_vec = np.zeros(self.dim, dtype=complex)
        state_vec[0] = 1.0
        state = QuantumState(state_vec, is_pure=True)
        
        # Apply parametric circuit
        unitary = self.circuit_func(x, self.params)
        return state.evolve_unitary(unitary)
        
    def update_params(self, new_params: np.ndarray):
        """Update trainable parameters.
        
        Args:
            new_params: New parameter values
        """
        if len(new_params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(new_params)}")
        self.params = new_params.copy()


class ZZFeatureMap(FeatureMap):
    """ZZ feature map (Havlíček et al.).
    
    Implements: U_Φ(x) = exp(i Σ_j φ_j(x) Z_j) exp(i Σ_{j<k} φ_{jk}(x) Z_j Z_k)
    """
    
    def __init__(self, n_features: int, reps: int = 2, 
                 entanglement: str = 'linear'):
        """Initialize ZZ feature map.
        
        Args:
            n_features: Number of features (= number of qubits)
            reps: Number of repetitions
            entanglement: Entanglement pattern ('linear', 'full')
        """
        super().__init__(n_features, n_features)
        self.reps = reps
        self.entanglement = entanglement
        
    def embed(self, x: np.ndarray) -> QuantumState:
        """Embed using ZZ feature map.
        
        Args:
            x: Feature vector
            
        Returns:
            Quantum state from ZZ feature map
        """
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")
            
        # Start with Hadamard layer
        state_vec = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        state = QuantumState(state_vec, is_pure=True)
        
        # Apply ZZ layers
        for rep in range(self.reps):
            # Single-qubit rotations
            for i in range(self.n_qubits):
                angle = x[i]
                rz_gate = self._rz_gate(angle)
                unitary = self._single_qubit_unitary(rz_gate, i)
                state = state.evolve_unitary(unitary)
                
            # Entangling gates
            if self.entanglement == 'linear':
                for i in range(self.n_qubits - 1):
                    angle = x[i] * x[i + 1]
                    rzz_gate = self._rzz_gate(angle, i, i + 1)
                    state = state.evolve_unitary(rzz_gate)
            elif self.entanglement == 'full':
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        angle = x[i] * x[j]
                        rzz_gate = self._rzz_gate(angle, i, j)
                        state = state.evolve_unitary(rzz_gate)
                        
        return state
        
    def _rz_gate(self, angle: float) -> np.ndarray:
        """Z-rotation gate.
        
        Args:
            angle: Rotation angle
            
        Returns:
            2x2 rotation matrix
        """
        return np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=complex)
        
    def _rzz_gate(self, angle: float, qubit1: int, qubit2: int) -> np.ndarray:
        """ZZ-rotation gate between two qubits.
        
        Args:
            angle: Rotation angle
            qubit1: First qubit index
            qubit2: Second qubit index
            
        Returns:
            Full unitary matrix
        """
        # Simplified implementation using matrix exponentiation
        zz_op = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i in range(self.dim):
            # Get bit representation
            bits = [(i >> k) & 1 for k in range(self.n_qubits)]
            
            # ZZ eigenvalue
            z1 = 1 if bits[qubit1] == 0 else -1
            z2 = 1 if bits[qubit2] == 0 else -1
            eigenval = z1 * z2
            
            zz_op[i, i] = eigenval
            
        # Matrix exponential
        return np.array([[np.exp(-1j * angle * zz_op[i, i] / 2) if i == j else 0 
                         for j in range(self.dim)] for i in range(self.dim)], dtype=complex)
        
    def _single_qubit_unitary(self, gate: np.ndarray, qubit_idx: int) -> np.ndarray:
        """Construct full unitary for single-qubit gate.
        
        Args:
            gate: 2x2 single-qubit gate
            qubit_idx: Index of target qubit
            
        Returns:
            Full unitary matrix
        """
        unitary = np.array([[1.0]], dtype=complex)
        
        for i in range(self.n_qubits):
            if i == qubit_idx:
                unitary = np.kron(unitary, gate)
            else:
                unitary = np.kron(unitary, np.eye(2))
                
        return unitary