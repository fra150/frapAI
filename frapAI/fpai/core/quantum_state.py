"""Quantum state representations for FPAI framework."""

import numpy as np
from typing import Union, Optional
from scipy.linalg import sqrtm


class QuantumState:
    """Quantum state representation on Hilbert space H.
    
    Supports both pure states |ψ⟩ and mixed states ρ (density matrices).
    Dimension d = 2^n for n qubits.
    """
    
    def __init__(self, state: Union[np.ndarray, None] = None, 
                 is_pure: bool = True, n_qubits: Optional[int] = None):
        """Initialize quantum state.
        
        Args:
            state: State vector |ψ⟩ (pure) or density matrix ρ (mixed)
            is_pure: Whether state is pure (True) or mixed (False)
            n_qubits: Number of qubits (inferred if not provided)
        """
        if state is None and n_qubits is None:
            raise ValueError("Must provide either state or n_qubits")
            
        if state is None:
            # Initialize |0⟩^⊗n state
            self.n_qubits = n_qubits
            self.dim = 2 ** n_qubits
            self._state_vector = np.zeros(self.dim, dtype=complex)
            self._state_vector[0] = 1.0
            self.is_pure = True
        else:
            state = np.asarray(state, dtype=complex)
            self.is_pure = is_pure
            
            if is_pure:
                # Pure state |ψ⟩
                if state.ndim != 1:
                    raise ValueError("Pure state must be 1D array")
                self._state_vector = state / np.linalg.norm(state)
                self.dim = len(state)
            else:
                # Mixed state ρ
                if state.ndim != 2 or state.shape[0] != state.shape[1]:
                    raise ValueError("Mixed state must be square matrix")
                self._density_matrix = state
                self.dim = state.shape[0]
                
            self.n_qubits = int(np.log2(self.dim))
            if 2 ** self.n_qubits != self.dim:
                raise ValueError(f"Dimension {self.dim} is not a power of 2")
                
    @property
    def state_vector(self) -> np.ndarray:
        """Get state vector |ψ⟩ (pure states only)."""
        if not self.is_pure:
            raise ValueError("Mixed states don't have state vectors")
        return self._state_vector
        
    @property
    def density_matrix(self) -> np.ndarray:
        """Get density matrix ρ."""
        if self.is_pure:
            return np.outer(self._state_vector, np.conj(self._state_vector))
        return self._density_matrix
        
    def normalize(self) -> 'QuantumState':
        """Normalize the quantum state.
        
        Returns:
            Normalized state
        """
        if self.is_pure:
            norm = np.linalg.norm(self._state_vector)
            if norm > 0:
                self._state_vector /= norm
        else:
            trace = np.trace(self._density_matrix)
            if trace > 0:
                self._density_matrix /= trace
        return self
        
    def is_valid(self) -> bool:
        """Check if state is valid quantum state.
        
        Returns:
            True if valid quantum state
        """
        if self.is_pure:
            return np.isclose(np.linalg.norm(self._state_vector), 1.0)
        else:
            rho = self._density_matrix
            # Check trace = 1
            if not np.isclose(np.trace(rho), 1.0):
                return False
            # Check positive semidefinite
            eigenvals = np.linalg.eigvals(rho)
            return np.all(eigenvals >= -1e-10)  # Allow small numerical errors
            
    def purity(self) -> float:
        """Compute purity Tr(ρ²).
        
        Returns:
            Purity (1 for pure states, < 1 for mixed)
        """
        rho = self.density_matrix
        return np.real(np.trace(rho @ rho))
        
    def von_neumann_entropy(self) -> float:
        """Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ).
        
        Returns:
            Entropy (0 for pure states)
        """
        rho = self.density_matrix
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove zero eigenvalues
        return -np.sum(eigenvals * np.log2(eigenvals))
        
    def evolve_unitary(self, unitary: np.ndarray) -> 'QuantumState':
        """Apply unitary evolution U|ψ⟩ or UρU†.
        
        Args:
            unitary: Unitary matrix U
            
        Returns:
            Evolved state
        """
        U = np.asarray(unitary, dtype=complex)
        if U.shape != (self.dim, self.dim):
            raise ValueError(f"Unitary shape {U.shape} doesn't match state dimension {self.dim}")
            
        if self.is_pure:
            new_state_vector = U @ self._state_vector
            return QuantumState(new_state_vector, is_pure=True)
        else:
            new_density_matrix = U @ self._density_matrix @ np.conj(U.T)
            return QuantumState(new_density_matrix, is_pure=False)
            
    def measure_expectation(self, observable: np.ndarray) -> float:
        """Compute expectation value ⟨O⟩ = Tr(ρO).
        
        Args:
            observable: Hermitian observable O
            
        Returns:
            Expectation value
        """
        O = np.asarray(observable, dtype=complex)
        if O.shape != (self.dim, self.dim):
            raise ValueError(f"Observable shape {O.shape} doesn't match state dimension {self.dim}")
            
        rho = self.density_matrix
        return np.real(np.trace(rho @ O))
        
    def partial_trace(self, subsystem_dims: list, traced_systems: list) -> 'QuantumState':
        """Compute partial trace over specified subsystems.
        
        Args:
            subsystem_dims: Dimensions of each subsystem
            traced_systems: Indices of systems to trace out
            
        Returns:
            Reduced state
        """
        # Simplified implementation for qubit systems
        if not all(d == 2 for d in subsystem_dims):
            raise NotImplementedError("Only qubit systems supported")
            
        rho = self.density_matrix
        n_systems = len(subsystem_dims)
        
        # Reshape to tensor form
        shape = subsystem_dims + subsystem_dims
        rho_tensor = rho.reshape(shape)
        
        # Trace out specified systems
        for sys_idx in sorted(traced_systems, reverse=True):
            axes = (sys_idx, sys_idx + n_systems)
            rho_tensor = np.trace(rho_tensor, axis1=axes[0], axis2=axes[1])
            n_systems -= 1
            
        # Reshape back to matrix
        remaining_dim = 2 ** (len(subsystem_dims) - len(traced_systems))
        reduced_rho = rho_tensor.reshape(remaining_dim, remaining_dim)
        
        return QuantumState(reduced_rho, is_pure=False)
        
    def fidelity(self, other: 'QuantumState') -> float:
        """Compute fidelity F(ρ, σ) between two states.
        
        Args:
            other: Other quantum state
            
        Returns:
            Fidelity (0 to 1)
        """
        if self.dim != other.dim:
            raise ValueError("States must have same dimension")
            
        rho = self.density_matrix
        sigma = other.density_matrix
        
        if self.is_pure and other.is_pure:
            # F = |⟨ψ|φ⟩|²
            overlap = np.vdot(self._state_vector, other._state_vector)
            return np.abs(overlap) ** 2
        else:
            # F = Tr(√(√ρ σ √ρ))²
            sqrt_rho = sqrtm(rho)
            M = sqrt_rho @ sigma @ sqrt_rho
            sqrt_M = sqrtm(M)
            return np.real(np.trace(sqrt_M)) ** 2
            
    def __repr__(self) -> str:
        state_type = "pure" if self.is_pure else "mixed"
        return f"QuantumState({state_type}, {self.n_qubits} qubits, dim={self.dim})"