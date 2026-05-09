"""
Step 5: Post-process execution results using SQD classical diagonalization.

Takes bitstring counts from step 4 and computes ground state energy
using the SKQD (Subspace-Search Quantum Diagonalization) method.
"""

#pylint: disable=import-outside-toplevel

import numpy as np
from qiskit.primitives import BitArray

import skqd_helpers

def counts_to_bitarray(counts: dict, num_qubits: int, reverse: bool = False) -> BitArray:
    """Convert counts dictionary to BitArray.
    
    Args:
        counts: Dictionary of bitstring -> count
        num_qubits: Number of qubits in the circuit
        reverse: If True, reverse bitstring order (Qiskit uses little-endian)
        
    Returns:
        BitArray suitable for classically_diagonalize()
    """
    # Expand counts into individual samples
    samples_list = []
    for bitstring, count in counts.items():
        # Convert bitstring to integer array
        # Qiskit returns bitstrings in little-endian (qubit 0 is rightmost)
        if reverse:
            bits = np.array([int(b) for b in reversed(bitstring)], dtype=np.uint8)
        else:
            bits = np.array([int(b) for b in bitstring], dtype=np.uint8)
        # Repeat for each count
        for _ in range(count):
            samples_list.append(bits)
    
    # Stack into 2D array
    samples = np.array(samples_list, dtype=np.uint8)
    
    # Pack bits into bytes for BitArray
    packed = np.packbits(samples, axis=1, bitorder='big')
    
    return BitArray(packed, num_bits=num_qubits)


def postprocess(
    counts: dict,
    num_orbs: int,
    hopping: float = 1.0,
    onsite: float = 5.0,
    hybridization: float = 1.0,
    filling_factor: float = -0.5,
    energy_tol: float = 1e-4,
    occupancies_tol: float = 1e-3,
    max_iterations: int = 10,
    num_batches: int = 5,
    samples_per_batch: int = 200,
    max_cycle: int = 200,
    symmetrize_spin: bool = True,
    carryover_threshold: float = 1e-5,
) -> list[float]:
    """Post-process bitstring counts to compute ground state energy.
    
    Args:
        counts: Dictionary of bitstring -> count from execution
        num_orbs: Number of spatial orbitals
        hopping: Hopping parameter
        onsite: Onsite energy (U)
        hybridization: Hybridization strength
        filling_factor: Multiplier for chemical potential
        energy_tol: Energy convergence tolerance
        occupancies_tol: Occupancy convergence tolerance
        max_iterations: Maximum SQD iterations
        num_batches: Number of batches for eigenstate solver
        samples_per_batch: Samples per batch
        max_cycle: Maximum CASCI cycles
        symmetrize_spin: Whether to symmetrize spin
        carryover_threshold: Threshold for carryover
        
    Returns:
        List of energies per iteration (final energy is result[-1])
    """
    num_qubits = 2 * num_orbs
    chemical_potential = filling_factor * onsite
    
    # Convert counts to BitArray
    print(f"Converting {len(counts)} unique bitstrings to BitArray...")
    bit_array = counts_to_bitarray(counts, num_qubits)
    print(f"BitArray: shape={bit_array.array.shape}, num_bits={bit_array.num_bits}")
    
    # Build SIAM Hamiltonian in site basis
    print("Building SIAM Hamiltonian (site basis)...")
    hcore, eri = skqd_helpers.siam_hamiltonian(
        num_orbs, hopping, onsite, hybridization, chemical_potential
    )
    
    # Run classical diagonalization
    print(f"Starting SQD diagonalization (max_iter={max_iterations}, "
          f"batches={num_batches}, samples={samples_per_batch})...")
    result = skqd_helpers.classically_diagonalize(
        bit_array=bit_array,
        hcore=hcore,
        eri=eri,
        num_orbitals=num_orbs,
        nelec=num_orbs,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        num_batches=num_batches,
        samples_per_batch=samples_per_batch,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        max_cycle=max_cycle,
        local=True
    )
    
    return result


def exact_siam_energy(
    hcore: np.ndarray,
    eri: np.ndarray,
    num_orbs: int,
) -> float:
    """Compute exact ground state energy for SIAM using FCI.
    
    Args:
        hcore: One-body Hamiltonian
        eri: Two-body electron repulsion integrals
        num_orbs: Number of spatial orbitals
        
    Returns:
        Exact ground state energy
    """
    from pyscf import fci
    nelec = num_orbs  # Half-filled
    exact_energy, _ = fci.direct_spin1.kernel(hcore, eri, num_orbs, (nelec // 2, nelec // 2))
    return exact_energy


def run_step5(
    counts: dict,
    num_orbs: int = 10,
    **kwargs
) -> list[float]:
    """Run step 5: SQD post-processing.
    
    Args:
        counts: Bitstring counts from step 4
        num_orbs: Number of orbitals
        **kwargs: Additional arguments passed to postprocess()
        
    Returns:
        List of energies per iteration
    """
    # Extract Hamiltonian params
    hopping = kwargs.get('hopping', 1.0)
    onsite = kwargs.get('onsite', 5.0)
    hybridization = kwargs.get('hybridization', 1.0)
    filling_factor = kwargs.get('filling_factor', -0.5)
    chemical_potential = filling_factor * onsite
    
    result = postprocess(counts, num_orbs, **kwargs)
    
    # Compute exact energy for comparison (in site basis)
    print("\nComputing exact ground state energy (FCI)...")
    hcore, eri = skqd_helpers.siam_hamiltonian(
        num_orbs, hopping, onsite, hybridization, chemical_potential
    )
    exact_energy = exact_siam_energy(hcore, eri, num_orbs)
    
    sqd_energy = result[-1]
    error = abs(sqd_energy - exact_energy)
    error_pct = abs(error / exact_energy) * 100
    
    print("\n" + "=" * 60)
    print("SQD RESULTS:")
    print("=" * 60)
    print(f"Energy history: {result}")
    print(f"Final SQD energy: {sqd_energy:.6f}")
    print(f"Exact FCI energy: {exact_energy:.6f}")
    print(f"Error:            {error:.6f} ({error_pct:.4f}%)")
    print(f"Iterations:       {len(result)}")
    print("=" * 60)
    
    return result


def main():
    """Run step 5 standalone from a case directory."""
    import json
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python step5_postprocess.py <case_dir>")
        sys.exit(1)
    
    case_dir = Path(sys.argv[1])
    
    # Load case info
    case_info_path = case_dir / 'case_info.json'
    if not case_info_path.exists():
        print(f"Error: {case_info_path} not found")
        sys.exit(1)
    with open(case_info_path, 'r', encoding='utf-8') as f:
        case_info = json.load(f)
    
    # Load step 4 outputs
    counts_path = case_dir / 'counts.json'
    if not counts_path.exists():
        print(f"Error: counts.json not found in {case_dir}")
        sys.exit(1)
    with open(counts_path, 'r', encoding='utf-8') as f:
        counts = json.load(f)
    
    # Run postprocessing
    num_orbs = case_info['num_orbs']
    result = run_step5(
        counts,
        num_orbs=num_orbs,
        hopping=case_info.get('hopping', 1.0),
        onsite=case_info.get('onsite', 5.0),
        hybridization=case_info.get('hybridization', 1.0),
        filling_factor=case_info.get('filling_factor', -0.5),
        max_iterations=case_info.get('max_iter', 10),
        num_batches=case_info.get('num_batches', 5),
        samples_per_batch=case_info.get('samples_per_batch', 200),
    )
    
    # Save outputs
    energy_history = {
        'energies': result,
        'num_iterations': len(result),
    }
    with open(case_dir / 'energy_history.json', 'w', encoding='utf-8') as f:
        json.dump(energy_history, f, indent=2)
    print("Saved: energy_history.json")


if __name__ == "__main__":
    main()
