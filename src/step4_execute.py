"""
Step: Execute circuits on a backend and return results.

When shots=0 the circuits are evaluated via exact statevector simulation
and bitstrings are drawn from the ideal probability distribution
(no shot noise).  This mirrors the shots=0 convention used by q8020-cfd.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qiskit_addon_sqd.counts import counts_to_arrays

_SV_DEFAULT_SAMPLES = 10_000


def _statevector_counts(
    circuits: list[QuantumCircuit],
    num_samples: int = _SV_DEFAULT_SAMPLES,
) -> dict:
    """Sample bitstrings from exact statevector probabilities.

    Each circuit is stripped of measurements, its statevector computed,
    and *num_samples* bitstrings are drawn from the ideal distribution.
    Counts from all circuits are merged.
    """
    rng = np.random.default_rng(42)
    combined: dict[str, int] = {}
    per_circuit = max(1, num_samples // len(circuits))

    for qc in circuits:
        # Remove measurements so Statevector can evaluate
        bare = qc.remove_final_measurements(inplace=False)
        assert bare is not None  # inplace=False always returns a new circuit
        sv = Statevector(bare)
        probs = sv.probabilities()
        num_qubits = qc.num_qubits

        indices = rng.choice(len(probs), size=per_circuit, p=probs)
        for idx in indices:
            bs = format(idx, f"0{num_qubits}b")
            combined[bs] = combined.get(bs, 0) + 1

    return combined


def execute_circuits(
    circuits: list[QuantumCircuit],
    backend=None,
    shots: int = 1024,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Execute circuits on a backend and return bitstrings.

    Args:
        circuits: List of transpiled quantum circuits.
        backend: Target backend. Defaults to AerSimulator.
        shots: Number of shots per circuit.  When 0, exact statevector
               probabilities are used (no shot noise).

    Returns:
        Tuple of (bitstrings, probabilities, combined_counts).
    """
    # --- statevector path (shots == 0) ---
    if shots == 0:
        print(
            f"Statevector mode: sampling {_SV_DEFAULT_SAMPLES} "
            f"bitstrings from exact distribution."
        )
        combined_counts = _statevector_counts(circuits)
        bitstrings, probabilities = counts_to_arrays(combined_counts)
        print(f"Collected {len(combined_counts)} unique bitstrings.")
        print(f"Bitstrings shape: {bitstrings.shape}")
        return bitstrings, probabilities, combined_counts

    # --- shot-based path ---
    if backend is None:
        backend = AerSimulator(method='automatic')

    job = backend.run(circuits, shots=shots)
    result = job.result()

    print(f"Executed {len(circuits)} circuits with {shots} shots each.")

    combined_counts: dict[str, int] = {}
    for i in range(len(circuits)):
        counts = result.get_counts(i)
        for bitstring, count in counts.items():
            combined_counts[bitstring] = (
                combined_counts.get(bitstring, 0) + count
            )

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
    from q8020_cfd_qutil import get_backend

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

    # Create backend via qutil
    backend = get_backend(
        name=case_info.get('backend'),
        backend_type=case_info.get('backend_type', 'sim'),
        t1=case_info.get('t1'),
        t2=case_info.get('t2'),
        coupling_map=case_info.get('coupling_map', 'default'),
    )

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
