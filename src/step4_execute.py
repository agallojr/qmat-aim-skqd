"""
Step: Execute circuits on a backend and return results.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit_addon_sqd.counts import counts_to_arrays


def execute_circuits(
    circuits: list[QuantumCircuit],
    backend=None,
    shots: int = 1024
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Execute circuits on a backend and return bitstrings.
    
    Args:
        circuits: List of transpiled quantum circuits.
        backend: Target backend. Defaults to AerSimulator.
        shots: Number of shots per circuit.
        
    Returns:
        Tuple of (bitstrings, probabilities, combined_counts).
    """
    if backend is None:
        backend = AerSimulator(method='automatic')
    
    # Run circuits directly on backend
    job = backend.run(circuits, shots=shots)
    result = job.result()
    
    print(f"Executed {len(circuits)} circuits with {shots} shots each.")
    
    # Combine counts from all circuit results
    combined_counts = {}
    for i in range(len(circuits)):
        counts = result.get_counts(i)
        for bitstring, count in counts.items():
            combined_counts[bitstring] = combined_counts.get(bitstring, 0) + count
    
    # Convert to arrays
    bitstrings, probabilities = counts_to_arrays(combined_counts)
    
    print(f"Collected {len(combined_counts)} unique bitstrings.")
    print(f"Bitstrings shape: {bitstrings.shape}")
    
    return bitstrings, probabilities, combined_counts


def run_step_execute(
    circuits: list[QuantumCircuit],
    shots: int = 1024
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run execution step with default Aer density_matrix backend.
    
    Args:
        circuits: Transpiled circuits to execute.
        shots: Number of shots per circuit.
        
    Returns:
        Tuple of (bitstrings, probabilities, combined_counts).
    """
    return execute_circuits(circuits, shots=shots)


def main():
    """Run step 4 standalone from a case directory."""
    import json
    import sys
    from pathlib import Path
    from qiskit import qpy
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    
    if len(sys.argv) < 2:
        print("Usage: python step4_execute.py <case_dir>")
        sys.exit(1)
    
    case_dir = Path(sys.argv[1])
    
    # Load case info
    case_info_path = case_dir / 'case_info.json'
    if not case_info_path.exists():
        print(f"Error: {case_info_path} not found")
        sys.exit(1)
    with open(case_info_path, 'r', encoding='utf-8') as f:
        case_info = json.load(f)
    
    # Load step 3 outputs
    transpiled_path = case_dir / 'transpiled_circuits.qpy'
    if not transpiled_path.exists():
        print(f"Error: transpiled_circuits.qpy not found in {case_dir}")
        sys.exit(1)
    with open(transpiled_path, 'rb') as f:
        transpiled = qpy.load(f)
    
    # Create backend with noise if specified
    noise = case_info.get('noise', 0.0)
    if noise > 0:
        noise_model = NoiseModel()
        error_2q = depolarizing_error(noise, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'cp'])
        backend = AerSimulator(noise_model=noise_model)
    else:
        backend = AerSimulator(method='automatic')
    
    # Execute
    shots = case_info.get('shots', 1024)
    bitstrings, probabilities, counts = execute_circuits(transpiled, backend=backend, shots=shots)
    
    # Save outputs
    with open(case_dir / 'counts.json', 'w', encoding='utf-8') as f:
        json.dump(counts, f)
    np.save(case_dir / 'bitstrings.npy', bitstrings)
    np.save(case_dir / 'probabilities.npy', probabilities)
    print("Saved: counts.json, bitstrings.npy, probabilities.npy")


if __name__ == "__main__":
    main()
