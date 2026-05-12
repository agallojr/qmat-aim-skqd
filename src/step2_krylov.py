"""
Step 2: Construct Krylov circuits for AIM.
"""

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister

import skqd_helpers


def construct_krylov_siam(
    num_orbs: int,
    impurity_index: int,
    hamiltonian: tuple[np.ndarray, np.ndarray],
    dt: float,
    krylov_dim: int,
    num_imp_orbs: int = 1,
    trotter_order: int = 2,
    trotter_substeps: int = 1,
    n_alpha: int | None = None,
    n_beta: int | None = None,
) -> list[QuantumCircuit]:
    """Generate Krylov circuits for AIM.

    Args:
        num_orbs: Number of spatial orbitals
        impurity_index: Index of impurity orbital (center orbital)
        hamiltonian: (h1e, h2e) — one- and two-body Hamiltonian terms (raw)
        dt: Time step
        krylov_dim: Number of Krylov basis states
        num_imp_orbs: Number of impurity orbitals (1 for SIAM, M for multi-orbital)
        trotter_order: Suzuki product-formula order (1, 2, or 4); default 2
        trotter_substeps: Sub-divisions per Krylov increment; default 1
        n_alpha: Number of spin-up electrons (default num_orbs // 2)
        n_beta: Number of spin-down electrons (default num_orbs // 2)

    Returns:
        AIM Krylov circuits.
    """
    h1e, h2e = hamiltonian
    num_qubits = 2 * num_orbs
    if n_alpha is None:
        n_alpha = num_orbs // 2
    if n_beta is None:
        n_beta = num_orbs // 2

    circuits = []
    for k in range(krylov_dim):
        qreg = QuantumRegister(num_qubits, 'q')
        qc = QuantumCircuit(qreg)

        for instruction in skqd_helpers.prepare_initial_state(
            qreg, num_orbs, n_alpha, n_beta
        ):
            qc.append(instruction)

        for _ in range(k):
            for instruction in skqd_helpers.trotter_step(
                qreg, dt, h1e, h2e, impurity_index, num_orbs,
                num_imp_orbs=num_imp_orbs,
                order=trotter_order, substeps=trotter_substeps,
            ):
                qc.append(instruction)

        circuits.append(qc)

    return circuits


def _resolve_dt(
    h1e: np.ndarray, h2e: np.ndarray, dt_multiplier: float, mode: str
) -> tuple[float, float, float, float]:
    """Pick dt from the requested scale mode.

    Returns (dt, omega_h1e, omega_h2e, omega_total).

    Mode 'h1e' (default): dt = pi * dt_mult / ||h1e||_2. This is the original
        Krylov-SQD heuristic. The empirically-good regime: Krylov states are
        spread enough that the SQD subspace covers the ground state.

    Mode 'full': dt = pi * dt_mult / (||h1e||_2 + max|h2e|). The per-Pauli-
        term scale; bounds the rotation angle of any single Pauli per Trotter
        step. Gives a Trotter-accurate evolution (proper dt^{order+1} error
        scaling) but produces tightly clustered Krylov states. To use this
        mode for SQD, pair with --trotter-substeps and/or a larger --dt-mult
        so the *total* evolution time across k Krylov circuits stays in the
        spread-enough regime.

    SQD subspace coverage and Trotter accuracy are not the same metric.
    """
    omega_h1e, omega_h2e, omega_total = skqd_helpers.estimate_hamiltonian_scale(
        h1e, h2e
    )
    if mode == 'h1e':
        scale = omega_h1e
    elif mode == 'full':
        scale = omega_total
    else:
        raise ValueError(f"Unknown dt_scale_mode: {mode!r} (use 'h1e' or 'full')")
    dt = dt_multiplier * np.pi / scale
    return dt, omega_h1e, omega_h2e, omega_total


