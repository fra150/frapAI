"""POVM (Positive Operator-Valued Measure) implementations for FPAI framework."""

import numpy as np
from typing import List, Union, Optional
from .quantum_state import QuantumState


class POVM:
    """POVM measurement {E_y} with E_y ≥ 0 and Σ_y E_y = I.
    
    Implements generalized quantum measurements for classification.
    Probabilities given by Born rule: P(y|x) = Tr(ρ(x) E_y).
    """
    
    def __init__(self, effects: List[np.ndarray], labels: Optional[List] = None):
        """Initialize POVM with measurement effects.
        
        Args:
            effects: List of POVM effects {E_y}
            labels: Class labels (default: 0, 1, 2, ...)
        """
        self.effects = [np.asarray(E, dtype=complex) for E in effects]
        self.n_outcomes = len(effects)
        
        if labels is None:
            self.labels = list(range(self.n_outcomes))
        else:
            self.labels = labels
            if len(labels) != self.n_outcomes:
                raise ValueError("Number of labels must match number of effects")
                
        # Validate POVM
        if not self._is_valid_povm():
            raise ValueError("Invalid POVM: effects must be positive and sum to identity")
            
    @property
    def dim(self) -> int:
        """Hilbert space dimension."""
        return self.effects[0].shape[0]
        
    def _is_valid_povm(self, tol: float = 1e-10) -> bool:
        """Check if POVM is valid.
        
        Args:
            tol: Numerical tolerance
            
        Returns:
            True if valid POVM
        """
        if not self.effects:
            return False
            
        dim = self.effects[0].shape[0]
        
        # Check all effects have same dimension and are square
        for E in self.effects:
            if E.shape != (dim, dim):
                return False
                
        # Check positive semidefinite
        for E in self.effects:
            eigenvals = np.linalg.eigvals(E)
            if np.any(eigenvals < -tol):
                return False
                
        # Check completeness: Σ E_y = I
        total = sum(self.effects)
        identity = np.eye(dim)
        return np.allclose(total, identity, atol=tol)
        
    def measure_probs(self, state: Union[QuantumState, np.ndarray]) -> np.ndarray:
        """Compute measurement probabilities P(y) = Tr(ρ E_y).
        
        Args:
            state: Quantum state (QuantumState or density matrix)
            
        Returns:
            Probability vector [P(y=0), P(y=1), ...]
        """
        if isinstance(state, QuantumState):
            rho = state.density_matrix
        else:
            rho = np.asarray(state, dtype=complex)
            
        if rho.shape != (self.dim, self.dim):
            raise ValueError(f"State dimension {rho.shape} doesn't match POVM dimension {self.dim}")
            
        probs = np.zeros(self.n_outcomes)
        for i, E in enumerate(self.effects):
            probs[i] = np.real(np.trace(rho @ E))
            
        # Ensure probabilities are valid
        probs = np.clip(probs, 0, 1)
        probs /= np.sum(probs)  # Renormalize
        
        return probs
        
    def sample_outcome(self, state: Union[QuantumState, np.ndarray], 
                      n_shots: int = 1) -> Union[int, np.ndarray]:
        """Sample measurement outcomes.
        
        Args:
            state: Quantum state
            n_shots: Number of measurement shots
            
        Returns:
            Sampled outcome(s)
        """
        probs = self.measure_probs(state)
        outcomes = np.random.choice(self.n_outcomes, size=n_shots, p=probs)
        
        if n_shots == 1:
            return outcomes[0]
        return outcomes
        
    def post_measurement_state(self, state: Union[QuantumState, np.ndarray], 
                              outcome: int) -> QuantumState:
        """Compute post-measurement state after observing outcome y.
        
        Lüders rule: ρ_y = √E_y ρ √E_y / Tr(ρ E_y)
        
        Args:
            state: Pre-measurement state
            outcome: Observed outcome
            
        Returns:
            Post-measurement state
        """
        if isinstance(state, QuantumState):
            rho = state.density_matrix
        else:
            rho = np.asarray(state, dtype=complex)
            
        E_y = self.effects[outcome]
        prob_y = np.real(np.trace(rho @ E_y))
        
        if prob_y < 1e-12:
            raise ValueError(f"Outcome {outcome} has zero probability")
            
        # Compute √E_y
        sqrt_E_y = self._matrix_sqrt(E_y)
        
        # Post-measurement state
        rho_post = sqrt_E_y @ rho @ sqrt_E_y / prob_y
        
        return QuantumState(rho_post, is_pure=False)
        
    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root.
        
        Args:
            matrix: Positive semidefinite matrix
            
        Returns:
            Matrix square root
        """
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
        sqrt_eigenvals = np.sqrt(eigenvals)
        return eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T.conj()
        
    @classmethod
    def projective(cls, n_outcomes: int, dim: Optional[int] = None) -> 'POVM':
        """Create projective POVM with computational basis projectors.
        
        Args:
            n_outcomes: Number of outcomes
            dim: Hilbert space dimension (default: n_outcomes)
            
        Returns:
            Projective POVM
        """
        if dim is None:
            dim = n_outcomes
            
        if n_outcomes > dim:
            raise ValueError("Number of outcomes cannot exceed dimension")
            
        effects = []
        for i in range(n_outcomes):
            proj = np.zeros((dim, dim), dtype=complex)
            proj[i, i] = 1.0
            effects.append(proj)
            
        # Add remaining projector if needed
        if n_outcomes < dim:
            remaining_proj = np.eye(dim) - sum(effects)
            effects.append(remaining_proj)
            
        return cls(effects)
        
    @classmethod
    def symmetric_informationally_complete(cls, dim: int) -> 'POVM':
        """Create SIC-POVM (when it exists).
        
        Args:
            dim: Hilbert space dimension
            
        Returns:
            SIC-POVM
        """
        # Simplified implementation for qubits (dim=2)
        if dim == 2:
            # Tetrahedron SIC-POVM
            effects = []
            # Pauli matrices
            sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
            sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
            identity = np.eye(2, dtype=complex)
            
            # SIC vectors (tetrahedron vertices on Bloch sphere)
            directions = [
                identity,
                sigma_z,
                (-sigma_x + sigma_z) / np.sqrt(2),
                (-sigma_x - sigma_z) / np.sqrt(2)
            ]
            
            for direction in directions:
                # Project to |+⟩ state along direction
                eigenvals, eigenvecs = np.linalg.eigh(direction)
                max_idx = np.argmax(eigenvals)
                state_vec = eigenvecs[:, max_idx]
                proj = np.outer(state_vec, np.conj(state_vec))
                effects.append(proj / 2)  # Normalize for POVM
                
            return cls(effects)
        else:
            raise NotImplementedError(f"SIC-POVM not implemented for dimension {dim}")
            
    def __repr__(self) -> str:
        return f"POVM({self.n_outcomes} outcomes, dim={self.dim})"