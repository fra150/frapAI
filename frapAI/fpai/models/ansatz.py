"""Variational quantum circuit ansätze for FPAI."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from ..core.quantum_state import QuantumState


class Ansatz(ABC):
    """Abstract base class for variational quantum circuits.
    
    An ansatz defines the parametric quantum circuit U_θ that acts on
    the embedded quantum state.
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 1):
        """Initialize ansatz.
        Args:
            n_qubits: Number of qubits
            n_layers: Number of circuit layers
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = None
        
    @abstractmethod
    def n_parameters(self) -> int:
        """Return number of parameters."""
        pass
        
    @abstractmethod
    def evolve(self, state: QuantumState) -> QuantumState:
        """Apply parametric evolution to quantum state.
        
        Args:
            state: Input quantum state
            
        Returns:
            Evolved quantum state
        """
        pass
        
    def initialize_params(self, method: str = 'random', seed: Optional[int] = None):
        """Initialize circuit parameters.
        
        Args:
            method: Initialization method ('random', 'zeros', 'xavier')
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_params = self.n_parameters()
        
        if method == 'random':
            self.params = np.random.uniform(0, 2*np.pi, n_params)
        elif method == 'zeros':
            self.params = np.zeros(n_params)
        elif method == 'xavier':
            # Xavier/Glorot initialization adapted for quantum circuits
            limit = np.sqrt(6.0 / (self.n_qubits + n_params))
            self.params = np.random.uniform(-limit, limit, n_params)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
            
    def set_params(self, params: np.ndarray):
        """Set circuit parameters.
        
        Args:
            params: Parameter array
        """
        if len(params) != self.n_parameters():
            raise ValueError(f"Expected {self.n_parameters()} parameters, got {len(params)}")
        self.params = params.copy()
        
    def get_params(self) -> np.ndarray:
        """Get current parameters.
        
        Returns:
            Parameter array
        """
        if self.params is None:
            raise ValueError("Parameters not initialized")
        return self.params.copy()
            
    def get_circuit_depth(self) -> int:
        """Return circuit depth."""
        return self.n_layers
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_qubits={self.n_qubits}, n_layers={self.n_layers}, n_params={self.n_parameters()})"

class HardwareEfficientAnsatz(Ansatz):
    """Hardware-efficient ansatz.
    
    Alternates single-qubit rotations with entangling gates.
    Structure: [RY-RZ]⊗n → CNOT_ring → [RY-RZ]⊗n → ...
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 1, 
                 rotation_gates: List[str] = ['RY', 'RZ'],
                 entangling_gate: str = 'CNOT',
                 entangling_pattern: str = 'linear'):
        """Initialize hardware-efficient ansatz.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of layers
            rotation_gates: Single-qubit rotation gates
            entangling_gate: Two-qubit entangling gate
            entangling_pattern: Entangling pattern ('linear', 'circular', 'all')
        """
        super().__init__(n_qubits, n_layers)
        self.rotation_gates = rotation_gates
        self.entangling_gate = entangling_gate
        self.entangling_pattern = entangling_pattern
        
    def n_parameters(self) -> int:
        """Number of parameters = n_layers * n_qubits * n_rotation_gates."""
        return self.n_layers * self.n_qubits * len(self.rotation_gates)
        
    def evolve(self, state: QuantumState) -> QuantumState:
        """Apply hardware-efficient circuit.
        
        Args:
            state: Input quantum state
            
        Returns:
            Evolved quantum state
        """
        if self.params is None:
            raise ValueError("Parameters not initialized")
            
        current_state = state
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                for gate in self.rotation_gates:
                    angle = self.params[param_idx]
                    current_state = self._apply_rotation(current_state, qubit, gate, angle)
                    param_idx += 1
                    
            # Entangling gates (except last layer)
            if layer < self.n_layers - 1 or self.n_layers == 1:
                current_state = self._apply_entangling_layer(current_state)
                
        return current_state
        
    def _apply_rotation(self, state: QuantumState, qubit: int, gate: str, angle: float) -> QuantumState:
        """Apply single-qubit rotation.
        
        Args:
            state: Quantum state
            qubit: Target qubit
            gate: Gate type ('RX', 'RY', 'RZ')
            angle: Rotation angle
            
        Returns:
            Rotated state
        """
        # Create rotation matrix
        if gate == 'RX':
            U = np.array([[np.cos(angle/2), -1j*np.sin(angle/2)],
                         [-1j*np.sin(angle/2), np.cos(angle/2)]])
        elif gate == 'RY':
            U = np.array([[np.cos(angle/2), -np.sin(angle/2)],
                         [np.sin(angle/2), np.cos(angle/2)]])
        elif gate == 'RZ':
            U = np.array([[np.exp(-1j*angle/2), 0],
                         [0, np.exp(1j*angle/2)]])
        else:
            raise ValueError(f"Unknown rotation gate: {gate}")
            
        # Apply to full system
        full_U = self._expand_single_qubit_gate(U, qubit)
        return state.evolve_unitary(full_U)
        
    def _apply_entangling_layer(self, state: QuantumState) -> QuantumState:
        """Apply layer of entangling gates.
        
        Args:
            state: Quantum state
            
        Returns:
            Entangled state
        """
        current_state = state
        
        if self.entangling_pattern == 'linear':
            # Linear chain: 0-1, 1-2, 2-3, ...
            for i in range(self.n_qubits - 1):
                current_state = self._apply_two_qubit_gate(current_state, i, i+1)
        elif self.entangling_pattern == 'circular':
            # Circular: 0-1, 1-2, ..., (n-1)-0
            for i in range(self.n_qubits):
                current_state = self._apply_two_qubit_gate(current_state, i, (i+1) % self.n_qubits)
        elif self.entangling_pattern == 'all':
            # All-to-all connectivity
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    current_state = self._apply_two_qubit_gate(current_state, i, j)
        else:
            raise ValueError(f"Unknown entangling pattern: {self.entangling_pattern}")
            
        return current_state
        
    def _apply_two_qubit_gate(self, state: QuantumState, control: int, target: int) -> QuantumState:
        """Apply two-qubit gate.
        
        Args:
            state: Quantum state
            control: Control qubit
            target: Target qubit
            
        Returns:
            Evolved state
        """
        if self.entangling_gate == 'CNOT':
            # CNOT gate
            gate = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]])
        elif self.entangling_gate == 'CZ':
            # Controlled-Z gate
            gate = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, -1]])
        else:
            raise ValueError(f"Unknown entangling gate: {self.entangling_gate}")
            
        # Expand to full system
        full_gate = self._expand_two_qubit_gate(gate, control, target)
        return state.evolve_unitary(full_gate)
        
    def _expand_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Expand single-qubit gate to full system.
        
        Args:
            gate: 2x2 single-qubit gate
            qubit: Target qubit index
            
        Returns:
            Full system unitary
        """
        # Tensor product construction
        full_gate = np.array([[1]])
        
        for i in range(self.n_qubits):
            if i == qubit:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
                
        return full_gate
        
    def _expand_two_qubit_gate(self, gate: np.ndarray, control: int, target: int) -> np.ndarray:
        """Expand two-qubit gate to full system.
        
        Args:
            gate: 4x4 two-qubit gate
            control: Control qubit
            target: Target qubit
            
        Returns:
            Full system unitary
        """
        # For simplicity, assume control < target
        if control > target:
            control, target = target, control
            # Swap gate matrix accordingly
            swap = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])
            gate = swap @ gate @ swap
            
        # Build full unitary
        dim = 2 ** self.n_qubits
        full_gate = np.eye(dim, dtype=complex)
        
        # Apply gate to computational basis states
        for i in range(dim):
            # Extract bits for control and target qubits
            bits = [(i >> (self.n_qubits - 1 - j)) & 1 for j in range(self.n_qubits)]
            control_bit = bits[control]
            target_bit = bits[target]
            
            # Map through two-qubit gate
            two_qubit_state = control_bit * 2 + target_bit
            
            for j in range(4):
                if abs(gate[j, two_qubit_state]) > 1e-12:
                    # Construct output state
                    new_control = (j >> 1) & 1
                    new_target = j & 1
                    
                    new_bits = bits.copy()
                    new_bits[control] = new_control
                    new_bits[target] = new_target
                    
                    # Convert back to index
                    new_i = sum(bit * (2 ** (self.n_qubits - 1 - k)) for k, bit in enumerate(new_bits))
                    full_gate[new_i, i] = gate[j, two_qubit_state]
                    
        return full_gate


class QAOAAnsatz(Ansatz):
    """Quantum Approximate Optimization Algorithm (QAOA) ansatz.
    
    Alternates problem Hamiltonian and mixer Hamiltonian evolution.
    U(β,γ) = ∏ᵢ e^{-iβᵢHₘ} e^{-iγᵢHₚ}
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 1, 
                 problem_hamiltonian: Optional[np.ndarray] = None):
        """Initialize QAOA ansatz.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of QAOA layers (p)
            problem_hamiltonian: Problem Hamiltonian Hₚ
        """
        super().__init__(n_qubits, n_layers)
        
        if problem_hamiltonian is None:
            # Default: MaxCut Hamiltonian on linear graph
            self.problem_hamiltonian = self._default_maxcut_hamiltonian()
        else:
            self.problem_hamiltonian = problem_hamiltonian
            
        # Mixer Hamiltonian: Hₘ = ∑ᵢ Xᵢ
        self.mixer_hamiltonian = self._mixer_hamiltonian()
        
    def n_parameters(self) -> int:
        """Number of parameters = 2 * n_layers (β and γ for each layer)."""
        return 2 * self.n_layers
        
    def evolve(self, state: QuantumState) -> QuantumState:
        """Apply QAOA circuit.
        
        Args:
            state: Input quantum state
            
        Returns:
            Evolved quantum state
        """
        if self.params is None:
            raise ValueError("Parameters not initialized")
            
        current_state = state
        
        for layer in range(self.n_layers):
            gamma = self.params[2 * layer]
            beta = self.params[2 * layer + 1]
            
            # Apply problem Hamiltonian evolution
            U_p = self._matrix_exp(-1j * gamma * self.problem_hamiltonian)
            current_state = current_state.evolve_unitary(U_p)
            
            # Apply mixer Hamiltonian evolution
            U_m = self._matrix_exp(-1j * beta * self.mixer_hamiltonian)
            current_state = current_state.evolve_unitary(U_m)
            
        return current_state
        
    def _default_maxcut_hamiltonian(self) -> np.ndarray:
        """Create default MaxCut Hamiltonian for linear graph.
        
        Returns:
            Problem Hamiltonian matrix
        """
        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim))
        
        # Add ZZ terms for adjacent qubits
        for i in range(self.n_qubits - 1):
            # ZᵢZᵢ₊₁ term
            zz_term = np.array([[1]])
            
            for j in range(self.n_qubits):
                if j == i or j == i + 1:
                    zz_term = np.kron(zz_term, np.array([[1, 0], [0, -1]]))  # Z
                else:
                    zz_term = np.kron(zz_term, np.eye(2))  # I
                    
            H += 0.5 * (np.eye(dim) - zz_term)  # (I - ZZ)/2
            
        return H
        
    def _mixer_hamiltonian(self) -> np.ndarray:
        """Create mixer Hamiltonian ∑ᵢ Xᵢ.
        
        Returns:
            Mixer Hamiltonian matrix
        """
        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim))
        
        # Add X terms for each qubit
        for i in range(self.n_qubits):
            x_term = np.array([[1]])
            
            for j in range(self.n_qubits):
                if j == i:
                    x_term = np.kron(x_term, np.array([[0, 1], [1, 0]]))  # X
                else:
                    x_term = np.kron(x_term, np.eye(2))  # I
                    
            H += x_term
            
        return H
        
    def _matrix_exp(self, A: np.ndarray) -> np.ndarray:
        """Compute matrix exponential.
        
        Args:
            A: Input matrix
            
        Returns:
            exp(A)
        """
        # Use eigendecomposition for Hermitian matrices
        if np.allclose(A, A.conj().T):
            eigenvals, eigenvecs = np.linalg.eigh(A)
            return eigenvecs @ np.diag(np.exp(eigenvals)) @ eigenvecs.conj().T
        else:
            # General case (slower)
            return np.linalg.matrix_power(np.eye(A.shape[0]) + A / 1000, 1000)
            
    def get_expectation_value(self, state: QuantumState, observable: np.ndarray) -> float:
        """Compute expectation value ⟨ψ|O|ψ⟩.
        
        Args:
            state: Quantum state
            observable: Observable operator
            
        Returns:
            Expectation value
        """
        return state.expectation_value(observable)