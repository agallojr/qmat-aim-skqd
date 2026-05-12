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
    samples_list = []
    for bitstring, count in counts.items():
        if reverse:
            bits = np.array([int(b) for b in reversed(bitstring)], dtype=bool)
        else:
            bits = np.array([int(b) for b in bitstring], dtype=bool)
        for _ in range(count):
            samples_list.append(bits)

    samples = np.array(samples_list, dtype=bool)
    return BitArray.from_bool_array(samples)


def postprocess(
    counts: dict,
    num_orbs: int,
    h1e: np.ndarray,
    h2e: np.ndarray,
    energy_tol: float = 1e-4,
    occupancies_tol: float = 1e-3,
    max_iterations: int = 10,
    num_batches: int = 5,
    samples_per_batch: int = 200,
    max_cycle: int = 200,
    symmetrize_spin: bool = True,
    carryover_threshold: float = 1e-5,
    n_alpha: int | None = None,
    n_beta: int | None = None,
    **_kwargs,
) -> list[float]:
    """Post-process bitstring counts to compute ground state energy.
    
    Args:
        counts: Dictionary of bitstring -> count from execution
        num_orbs: Number of spatial orbitals
        h1e: One-body Hamiltonian
        h2e: Two-body Hamiltonian
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

    # Resolve filling: default to half-filling (preserves legacy behavior).
    if n_alpha is None:
        n_alpha = num_orbs // 2
    if n_beta is None:
        n_beta = num_orbs // 2

    # qiskit-addon-sqd rejects symmetrize_spin when n_alpha != n_beta
    # (the symmetrization swaps alpha/beta bitmasks which is invalid off Sz=0).
    if n_alpha != n_beta and symmetrize_spin:
        print(
            f"Disabling symmetrize_spin: nelec=({n_alpha}, {n_beta}) is "
            f"off Sz=0."
        )
        symmetrize_spin = False

    # Convert counts to BitArray
    print(f"Converting {len(counts)} unique bitstrings to BitArray...")
    bit_array = counts_to_bitarray(counts, num_qubits)
    print(f"BitArray: shape={bit_array.array.shape}, num_bits={bit_array.num_bits}")

    print(f"Using Hamiltonian ({num_orbs} orbs, nelec=({n_alpha}, {n_beta}))...")

    # Run classical diagonalization
    print(f"Starting SQD diagonalization (max_iter={max_iterations}, "
          f"batches={num_batches}, samples={samples_per_batch})...")
    result = skqd_helpers.classically_diagonalize(
        bit_array=bit_array,
        hcore=h1e,
        eri=h2e,
        num_orbitals=num_orbs,
        num_elec_a=n_alpha,
        num_elec_b=n_beta,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        num_batches=num_batches,
        samples_per_batch=samples_per_batch,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        max_cycle=max_cycle,
        local=True,
    )
    assert result is not None  # local mode always returns a list[float]
    return result


def exact_siam_energy(
    hcore: np.ndarray,
    eri: np.ndarray,
    num_orbs: int,
    n_alpha: int | None = None,
    n_beta: int | None = None,
) -> float:
    """Compute exact ground state energy for SIAM using FCI.

    Args:
        hcore: One-body Hamiltonian
        eri: Two-body electron repulsion integrals
        num_orbs: Number of spatial orbitals
        n_alpha: Number of spin-up electrons (default num_orbs // 2)
        n_beta: Number of spin-down electrons (default num_orbs // 2)

    Returns:
        Exact ground state energy
    """
    from pyscf import fci
    if n_alpha is None:
        n_alpha = num_orbs // 2
    if n_beta is None:
        n_beta = num_orbs // 2
    if n_alpha + n_beta == 0:
        return 0.0
    exact_energy, _ = fci.direct_spin1.kernel(
        hcore, eri, num_orbs, (int(n_alpha), int(n_beta))
    )
    return exact_energy


def run_step5(
    counts: dict,
    num_orbs: int,
    h1e: np.ndarray,
    h2e: np.ndarray,
    **kwargs,
) -> list[float]:
    """Run step 5: SQD post-processing.

    Args:
        counts: Bitstring counts from step 4
        num_orbs: Number of orbitals
        h1e: One-body Hamiltonian
        h2e: Two-body Hamiltonian
        **kwargs: Additional arguments passed to postprocess() — including
            n_alpha, n_beta for Sz-resolved filling.

    Returns:
        List of energies per iteration
    """
    result = postprocess(counts, num_orbs, h1e=h1e, h2e=h2e, **kwargs)

    # Compute exact energy for comparison
    print("\nComputing exact ground state energy (FCI)...")
    exact_energy = exact_siam_energy(
        h1e, h2e, num_orbs,
        n_alpha=kwargs.get('n_alpha'),
        n_beta=kwargs.get('n_beta'),
    )
    
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
