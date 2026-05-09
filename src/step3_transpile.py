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


def run_step_transpile(circuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
    """Run transpilation step with default Aer density_matrix backend.
    
    Args:
        circuits: Circuits to transpile.
        
    Returns:
        Transpiled circuits.
    """
    return transpile_circuits(circuits)


def main():
    """Run step 3 standalone from a case directory."""
    import json
    import sys
    from pathlib import Path
    from qiskit import qpy
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    
    if len(sys.argv) < 2:
        print("Usage: python step3_transpile.py <case_dir>")
        sys.exit(1)
    
    case_dir = Path(sys.argv[1])
    
    # Load case info
    case_info_path = case_dir / 'case_info.json'
    if not case_info_path.exists():
        print(f"Error: {case_info_path} not found")
        sys.exit(1)
    with open(case_info_path, 'r', encoding='utf-8') as f:
        case_info = json.load(f)
    
    # Load step 2 outputs
    circuits_path = case_dir / 'circuits.qpy'
    if not circuits_path.exists():
        print(f"Error: circuits.qpy not found in {case_dir}")
        sys.exit(1)
    with open(circuits_path, 'rb') as f:
        circuits = qpy.load(f)
    
    # Create backend with noise if specified
    noise = case_info.get('noise', 0.0)
    if noise > 0:
        noise_model = NoiseModel()
        error_2q = depolarizing_error(noise, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'cp'])
        backend = AerSimulator(noise_model=noise_model)
    else:
        backend = AerSimulator(method='automatic')
    
    # Transpile
    opt_level = case_info.get('opt_level', 1)
    transpiled = transpile_circuits(circuits, backend=backend, optimization_level=opt_level)
    
    # Compute stats
    depths = [c.depth() for c in transpiled]
    gate_counts = [dict(c.count_ops()) for c in transpiled]
    
    # Save outputs
    with open(case_dir / 'transpiled_circuits.qpy', 'wb') as f:
        qpy.dump(transpiled, f)
    transpile_stats = {
        'optimization_level': opt_level,
        'backend': str(backend),
        'depths': depths,
        'gate_counts': gate_counts,
    }
    with open(case_dir / 'transpile_stats.json', 'w', encoding='utf-8') as f:
        json.dump(transpile_stats, f, indent=2)
    print("Saved: transpiled_circuits.qpy, transpile_stats.json")


if __name__ == "__main__":
    main()
