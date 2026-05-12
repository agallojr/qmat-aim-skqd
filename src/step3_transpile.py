"""
Step: Transpile circuits for a target backend.
"""

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator


def transpile_circuits(
    circuits: list[QuantumCircuit],
    backend=None,
    optimization_level: int = 1
) -> list[QuantumCircuit]:
    """Transpile circuits for a target backend.
    
    Args:
        circuits: List of quantum circuits to transpile.
        backend: Target backend. Defaults to AerSimulator with density_matrix method.
        optimization_level: Transpiler optimization level (0-3).
        
    Returns:
        List of transpiled circuits.
    """
    if backend is None:
        backend = AerSimulator(method='automatic')
    
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend
    )
    transpiled = pass_manager.run(circuits)
    
    print(f"Transpiled {len(transpiled)} circuits for {backend}.")
    return transpiled


