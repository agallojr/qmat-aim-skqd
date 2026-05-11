"""
Particle-conserving variational ansatz for multi-orbital Anderson impurity models.

Circuit structure (two Givens layers per spin block):
    Layer 1: imp-bath rotations — M params (one per orbital)
    Layer 2: adjacent imp-imp rotations — M-1 params (inter-orbital entanglement)

With symmetric_spin=True the up and down blocks share parameters (spin-SU(2)
invariant, appropriate for crystal-field-split systems without a magnetic field).
With symmetric_spin=False each block gets independent parameters.

Orbital ordering matches multi_orbital_aim_hamiltonian / skqd_helpers.py:
    spatial: [imp_0 .. imp_{M-1}, bath_{0,0} .. bath_{M-1,B-1}]
    qubit (JW, ffsim convention):
        spin-down block: qubits 0 .. num_orbs-1
        spin-up block:   qubits num_orbs .. 2*num_orbs-1

Ref for Givens rotation: PhysRevResearch.7.023186
"""
from __future__ import annotations

import numpy as np
from numpy import pi
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _givens_2q(theta, qc: QuantumCircuit, q0: int, q1: int) -> None:
    """Append a particle-conserving Givens rotation on qubits q0, q1.

    Mixes |10> <-> |01>; preserves |00> and |11>.
    """
    qc.ry(pi / 2, q0)
    qc.cx(q0, q1)
    qc.ry(theta / 4, q0)
    qc.ry(theta / 4, q1)
    qc.cx(q0, q1)
    qc.ry(-pi / 2, q0)


class MultiOrbitalAIMAnsatz:
    """Particle-conserving Givens ansatz for M-orbital AIM (B=1 bath per orbital).

    Args:
        num_imp_orbs: M, number of impurity (correlated) orbitals
        num_bath_per_imp: B, bath sites per impurity — only B=1 is supported
        symmetric_spin: if True, spin-up/down blocks share parameters

    Raises:
        NotImplementedError: if num_bath_per_imp != 1
    """

    def __init__(
        self,
        num_imp_orbs: int,
        num_bath_per_imp: int = 1,
        symmetric_spin: bool = True,
    ) -> None:
        if num_bath_per_imp != 1:
            raise NotImplementedError("Only B=1 is supported")
        self.M = num_imp_orbs
        self.B = num_bath_per_imp
        self.symmetric_spin = symmetric_spin
        self.num_orbs = num_imp_orbs * (1 + num_bath_per_imp)
        self.num_qubits = 2 * self.num_orbs

    @property
    def num_params(self) -> int:
        block = self.M + (self.M - 1)  # layer1 + layer2 per spin block
        return block if self.symmetric_spin else 2 * block

    def circuit(self, thetas: np.ndarray | None = None) -> QuantumCircuit:
        """Return the ansatz circuit, optionally with bound parameters.

        Args:
            thetas: 1-D array of length num_params. If None, returns a
                parametric circuit with a ParameterVector named "θ".

        Returns:
            Qiskit QuantumCircuit with num_qubits qubits and no measurements.
        """
        M = self.M
        num_orbs = self.num_orbs
        n_block = 2 * M - 1  # params per spin block
        params = ParameterVector("θ", self.num_params)

        qc = QuantumCircuit(self.num_qubits)

        # Initial state: all impurity orbitals occupied, all bath orbitals empty.
        # Gives exactly M electrons in each spin block (half-filling).
        for m in range(M):
            qc.x(m)            # imp_m spin-down
            qc.x(num_orbs + m) # imp_m spin-up

        def _apply_block(spin: int) -> None:
            """Apply both Givens layers to one spin block."""
            off = 0 if (self.symmetric_spin or spin == 0) else n_block
            base = spin * num_orbs  # qubit offset for this spin block

            # Layer 1: imp_m <-> bath_m (orbital m <-> orbital M+m)
            for m in range(M):
                _givens_2q(params[off + m], qc, base + m, base + M + m)

            # Layer 2: adjacent imp-imp (imp_m <-> imp_{m+1})
            for m in range(M - 1):
                _givens_2q(params[off + M + m], qc, base + m, base + m + 1)

        _apply_block(0)  # spin-down
        _apply_block(1)  # spin-up

        if thetas is not None:
            mapping = {params[i]: float(thetas[i]) for i in range(self.num_params)}
            qc = qc.assign_parameters(mapping)

        return qc

    def hf_params(self) -> np.ndarray:
        """Return zero parameters, corresponding to the HF initial state."""
        return np.zeros(self.num_params)

    def random_params(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return small random parameters suitable as a VQE starting point."""
        if rng is None:
            rng = np.random.default_rng()
        return 0.1 * rng.standard_normal(self.num_params)
