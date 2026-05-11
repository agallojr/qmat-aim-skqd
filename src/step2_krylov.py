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


def run_step2(
    h1e: np.ndarray,
    h2e: np.ndarray,
    krylov_dim: int = 5,
    dt_multiplier: float = 1.0,
    add_measurements: bool = True,
    num_imp_orbs: int = 1,
    trotter_order: int = 2,
    trotter_substeps: int = 1,
    dt_scale_mode: str = 'h1e',
    n_alpha: int | None = None,
    n_beta: int | None = None,
) -> list[QuantumCircuit]:
    """Run step 2: construct AIM Krylov circuits.

    Args:
        h1e: One-body Hamiltonian (num_orbs × num_orbs)
        h2e: Two-body Hamiltonian
        krylov_dim: Number of Krylov basis states
        dt_multiplier: Multiplier for time step
        add_measurements: Whether to add measurement gates
        num_imp_orbs: Number of impurity orbitals (1 for SIAM, M for multi-orbital)
        trotter_order: Suzuki product-formula order (1, 2, or 4); default 2
        trotter_substeps: Sub-divisions per Krylov increment; default 1
        dt_scale_mode: 'full' (omega_h1e + omega_h2e) or 'h1e' (legacy)

    Returns:
        List of Krylov circuits.
    """
    num_orbs = h1e.shape[0]
    dt, _, _, _ = _resolve_dt(h1e, h2e, dt_multiplier, dt_scale_mode)
    impurity_index = (num_orbs - 1) // 2

    circuits = construct_krylov_siam(
        num_orbs, impurity_index, (h1e, h2e), dt, krylov_dim,
        num_imp_orbs=num_imp_orbs,
        trotter_order=trotter_order, trotter_substeps=trotter_substeps,
        n_alpha=n_alpha, n_beta=n_beta,
    )

    if add_measurements:
        for qc in circuits:
            qc.measure_all()
    
    num_qubits = 2 * num_orbs
    print(f"Constructed {len(circuits)} AIM Krylov circuits ({num_qubits} qubits).")
    
    return circuits


def main():
    """Run step 2 standalone from a case directory."""
    import json
    import sys
    from pathlib import Path
    from qiskit import qpy
    
    if len(sys.argv) < 2:
        print("Usage: python step2_krylov.py <case_dir>")
        sys.exit(1)
    
    case_dir = Path(sys.argv[1])
    
    # Load case info
    case_info_path = case_dir / 'case_info.json'
    if not case_info_path.exists():
        print(f"Error: {case_info_path} not found")
        sys.exit(1)
    with open(case_info_path, 'r', encoding='utf-8') as f:
        case_info = json.load(f)
    
    # Load step 1 outputs
    h1e_path = case_dir / 'h1e_momentum.npy'
    h2e_path = case_dir / 'h2e_momentum.npy'
    if not h1e_path.exists() or not h2e_path.exists():
        print(f"Error: h1e_momentum.npy or h2e_momentum.npy not found in {case_dir}")
        sys.exit(1)
    h1e = np.load(h1e_path)
    h2e = np.load(h2e_path)
    
    # Build circuits
    num_orbs = case_info['num_orbs']
    num_imp_orbs = case_info.get('num_imp_orbs', 1)
    trotter_order = case_info.get('trotter_order', 2)
    trotter_substeps = case_info.get('trotter_substeps', 1)
    dt_scale_mode = case_info.get('dt_scale_mode', 'full')
    dt, omega_h1e, omega_h2e, omega_total = _resolve_dt(
        h1e, h2e, case_info['dt_mult'], dt_scale_mode
    )
    impurity_index = (num_orbs - 1) // 2

    n_alpha = case_info.get('nelec_alpha')
    n_beta = case_info.get('nelec_beta')
    circuits = construct_krylov_siam(
        num_orbs, impurity_index, (h1e, h2e), dt, case_info['krylov_dim'],
        num_imp_orbs=num_imp_orbs,
        trotter_order=trotter_order, trotter_substeps=trotter_substeps,
        n_alpha=n_alpha, n_beta=n_beta,
    )
    for qc in circuits:
        qc.measure_all()

    print(f"Constructed {len(circuits)} circuits ({2 * num_orbs} qubits)")

    # Save outputs
    with open(case_dir / 'circuits.qpy', 'wb') as f:
        qpy.dump(circuits, f)
    circuit_metadata = {
        'dt': dt,
        'impurity_index': impurity_index,
        'krylov_dim': case_info['krylov_dim'],
        'num_qubits': 2 * num_orbs,
        'trotter_order': trotter_order,
        'trotter_substeps': trotter_substeps,
        'dt_scale_mode': dt_scale_mode,
        'omega_h1e': omega_h1e,
        'omega_h2e': omega_h2e,
        'omega_total': omega_total,
    }
    with open(case_dir / 'circuit_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(circuit_metadata, f, indent=2)
    print("Saved: circuits.qpy, circuit_metadata.json")


if __name__ == "__main__":
    main()
