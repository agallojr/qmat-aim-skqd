"""
Step: Execute circuits on a backend and return results.

Shot-based execution delegates to ``q8020_cfd_qutil.execute_circuit_counts``,
which dispatches to ``backend.run`` for AerSimulator and to SamplerV2 for
real IBM hardware, and normalises multi-creg bitstrings.

When shots=0 the circuits are evaluated via exact statevector simulation
and bitstrings are drawn from the ideal probability distribution
(no shot noise — and, intentionally, no device noise either; see warning
in ``execute_circuits``).
"""

import time
import warnings

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qiskit_addon_sqd.counts import counts_to_arrays
from q8020_cfd_qutil import execute_circuit_counts

_SV_DEFAULT_SAMPLES = 10_000


def _statevector_counts(
    circuits: list[QuantumCircuit],
    num_samples: int = _SV_DEFAULT_SAMPLES,
    seed: int = 42,
) -> dict:
    """Sample bitstrings from exact statevector probabilities.

    Each circuit is stripped of measurements, its statevector computed,
    and *num_samples* bitstrings are drawn from the ideal distribution.
    Counts from all circuits are merged.
    """
    rng = np.random.default_rng(seed)
    combined: dict[str, int] = {}
    per_circuit = max(1, num_samples // len(circuits))

    for qc in circuits:
        bare = qc.remove_final_measurements(inplace=False)
        assert bare is not None
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
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Execute circuits on a backend and return bitstrings.

    Args:
        circuits: List of transpiled quantum circuits.
        backend: Target backend. Defaults to AerSimulator.
        shots: Number of shots per circuit.  When 0, exact statevector
               probabilities are used (no shot noise).
        seed: Optional simulator seed for reproducibility.

    Returns:
        Tuple of (bitstrings, probabilities, combined_counts, exec_info).

        ``exec_info`` summarises the run: backend repr/class, shots config,
        per-circuit timing & job ids (when available), aggregate wall times.
        For the statevector path it just records mode + sample count.
    """
    if shots == 0:
        if backend is not None and not _is_ideal_aer(backend):
            warnings.warn(
                "shots=0 uses exact statevector simulation and IGNORES the "
                f"configured backend ({backend!r}), including any noise model "
                "and coupling map. Use shots>0 to exercise the backend.",
                stacklevel=2,
            )
        print(
            f"Statevector mode: sampling {_SV_DEFAULT_SAMPLES} "
            f"bitstrings from exact distribution."
        )
        sv_seed = seed if seed is not None else 42
        t0 = time.time()
        combined_counts = _statevector_counts(circuits, seed=sv_seed)
        wall = time.time() - t0
        bitstrings, probabilities = counts_to_arrays(combined_counts)
        print(f"Collected {len(combined_counts)} unique bitstrings.")
        print(f"Bitstrings shape: {bitstrings.shape}")
        exec_info = {
            'mode': 'statevector',
            'samples_total': _SV_DEFAULT_SAMPLES,
            'num_circuits': len(circuits),
            'seed': sv_seed,
            'wall_time': wall,
            'backend_repr': repr(backend) if backend is not None else None,
        }
        return bitstrings, probabilities, combined_counts, exec_info

    if backend is None:
        backend = AerSimulator(method='automatic')

    combined_counts: dict[str, int] = {}
    per_circuit_info: list[dict] = []
    t0 = time.time()
    for qc in circuits:
        counts, info = execute_circuit_counts(qc, backend, shots=shots, seed=seed)
        per_circuit_info.append(info)
        for bitstring, count in counts.items():
            combined_counts[bitstring] = (
                combined_counts.get(bitstring, 0) + count
            )
    total_wall = time.time() - t0

    print(f"Executed {len(circuits)} circuits with {shots} shots each.")

    bitstrings, probabilities = counts_to_arrays(combined_counts)

    print(f"Collected {len(combined_counts)} unique bitstrings.")
    print(f"Bitstrings shape: {bitstrings.shape}")

    exec_info = {
        'mode': 'shot',
        'shots_per_circuit': shots,
        'num_circuits': len(circuits),
        'seed': seed,
        'wall_time': total_wall,
        'backend_repr': repr(backend),
        'backend_class': type(backend).__name__,
        'shots_executed_total': sum(
            i.get('shots_executed', shots) for i in per_circuit_info
        ),
        'backend_time_total': sum(
            i.get('backend_time', 0.0) or 0.0 for i in per_circuit_info
        ),
        'job_ids': [i.get('job_id') for i in per_circuit_info if i.get('job_id')],
        'per_circuit': per_circuit_info,
    }
    return bitstrings, probabilities, combined_counts, exec_info


def _is_ideal_aer(backend) -> bool:
    """True iff backend is an AerSimulator with no noise model attached."""
    if not isinstance(backend, AerSimulator):
        return False
    return getattr(backend.options, "noise_model", None) is None
