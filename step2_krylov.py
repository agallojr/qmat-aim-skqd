"""
Step 2: Construct Krylov circuits for SIAM.
"""

import numpy as np
import scipy

from qiskit import QuantumCircuit, QuantumRegister

import skqd_helpers
from step1_siam import siam_hamiltonian_momentum


def construct_krylov_siam(
    num_orbs: int,
    impurity_index: int,
    hamiltonian: tuple[np.ndarray, np.ndarray],
    dt: float,
    krylov_dim: int
) -> list[QuantumCircuit]:
    """Generate Krylov circuits for SIAM.

    Args:
        num_orbs: Number of spatial orbitals
        impurity_index: Index of impurity orbital
        hamiltonian: one- and two-body Hamiltonian terms
        dt: Time step
        krylov_dim: Number of Krylov basis states
    
    Returns:
        SIAM Krylov circuits.
    """
    circuits = []
    
    # SIAM requires 2*num_orbs qubits (spin up and spin down)
    num_qubits = 2 * num_orbs
    
    # Occupancy number (half-filled system)
    occ_num = num_orbs // 2
    
    # Convert h1e to unitary time evolution operator for OrbitalRotationJW
    h1e_unitary = scipy.linalg.expm(-1j * dt * hamiltonian[0])
    hamiltonian_unitary = (h1e_unitary, hamiltonian[1])
    
    for k in range(krylov_dim):
        # Create quantum circuit with quantum register
        qreg = QuantumRegister(num_qubits, 'q')
        qc = QuantumCircuit(qreg)
        
        # Prepare initial state using helper
        for instruction in skqd_helpers.prepare_initial_state(qreg, num_orbs, occ_num):
            qc.append(instruction)
        
        # Apply k Trotter steps for k-th Krylov state using helper
        for _ in range(k):
            for instruction in \
                skqd_helpers.trotter_step(qreg, dt, hamiltonian_unitary,
                    impurity_index, num_orbs):
                qc.append(instruction)
        
        circuits.append(qc)
    
    return circuits


def run_step2(
    num_orbs: int = 10,
    hopping: float = 1.0,
    onsite: float = 5.0,
    hybridization: float = 1.0,
    krylov_dim: int = 5,
    filling_factor: float = -0.5,
    dt_multiplier: float = 1.0,
    add_measurements: bool = True,
) -> list[QuantumCircuit]:
    """Run step 2: construct SIAM Krylov circuits.
    
    Args:
        num_orbs: Number of spatial orbitals (qubits = 2 * num_orbs)
        hopping: Hopping parameter
        onsite: Onsite energy (U)
        hybridization: Hybridization strength
        krylov_dim: Number of Krylov basis states
        filling_factor: Multiplier for chemical potential
        dt_multiplier: Multiplier for time step (default 1.0, try 6-10 for more evolution)
        add_measurements: Whether to add measurement gates
        
    Returns:
        List of Krylov circuits.
    """
    chemical_potential = filling_factor * onsite
    
    hamiltonian = siam_hamiltonian_momentum(
        num_orbs, hopping, onsite, hybridization, chemical_potential
    )
    
    dt = dt_multiplier * np.pi / np.linalg.norm(hamiltonian[0], ord=2)
    impurity_index = (num_orbs - 1) // 2
    
    circuits = construct_krylov_siam(
        num_orbs, impurity_index, hamiltonian, dt, krylov_dim
    )
    
    if add_measurements:
        for qc in circuits:
            qc.measure_all()
    
    num_qubits = 2 * num_orbs
    print(f"Constructed {len(circuits)} SIAM Krylov circuits ({num_qubits} qubits).")
    
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
    dt = case_info['dt_mult'] * np.pi / np.linalg.norm(h1e, ord=2)
    impurity_index = (num_orbs - 1) // 2
    
    circuits = construct_krylov_siam(
        num_orbs, impurity_index, (h1e, h2e), dt, case_info['krylov_dim']
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
    }
    with open(case_dir / 'circuit_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(circuit_metadata, f, indent=2)
    print("Saved: circuits.qpy, circuit_metadata.json")


if __name__ == "__main__":
    main()
