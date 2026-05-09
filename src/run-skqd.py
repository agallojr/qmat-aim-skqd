#!/usr/bin/env python3
"""
SKQD Solver — Anderson Impurity Model workflow.

Unified interface: all problems are specified via --num-imp-orbs and
--num-bath-per-imp.  Single-orbital star geometry is the special case
--num-imp-orbs 1.

Usage:
    python src/run-skqd.py [OPTIONS]

Options:
    --num-imp-orbs N       Number of impurity orbitals M (default: 1)
    --num-bath-per-imp N   Bath sites per impurity orbital B (default: 1)
    --onsite N             Intra-orbital Coulomb U (default: 4.0)
    --U-prime N            Inter-orbital Coulomb (default: U - 2*J_H)
    --J-H N               Hund's exchange coupling (default: 0.0)
    --mu N                 Chemical potential (default: 2.0)
    --hybridization N      Impurity-bath coupling V (default: 0.8)
    --crystal-field a,b,c  Crystal-field energies per imp orbital (comma-sep)
    --krylov-dim N         Krylov dimension (default: 5)
    --dt-mult N            Time step multiplier (default: 1.0)
    --shots N              Number of shots for execution (default: 1024)
    --opt-level N          Transpiler optimization level 0-3 (default: 1)
    --max-iter N           SQD max iterations (default: 10)
    --num-batches N        SQD number of batches (default: 5)
    --samples-per-batch N  SQD samples per batch (default: 200)
    --output-dir DIR       Output directory for case metadata (optional)
    --system-label STR     Chemical system label for metadata (e.g. 'Ce 4f')
    --no-persist           Disable saving intermediate results
    --backend NAME         Fake backend name for topology/noise
    --backend-type T       Backend mode: sim, fake, hardware (default: sim)
    --coupling-map M       Coupling map: default or all-to-all (default: default)
    --t1 N                 T1 relaxation time in µs
    --t2 N                 T2 dephasing time in µs
"""

#pylint: disable=import-outside-toplevel, unused-variable

import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    """Parse command line arguments."""
    args = {
        'num_imp_orbs': 1,
        'num_bath_per_imp': 1,
        'onsite': 4.0,
        'U_prime': None,
        'J_H': None,
        'mu': 2.0,
        'hybridization': 0.8,
        'crystal_field': None,
        'krylov_dim': 5,
        'dt_mult': 1.0,
        'shots': 1024,
        'opt_level': 1,
        'max_iter': 10,
        'num_batches': 5,
        'samples_per_batch': 200,
        'system_label': None,
        'output_dir': None,
        'persist': True,
        'backend': None,
        'backend_type': 'sim',
        'coupling_map': 'default',
        't1': None,
        't2': None,
    }

    for i, arg in enumerate(sys.argv):
        if arg == '--num-imp-orbs' and i + 1 < len(sys.argv):
            args['num_imp_orbs'] = int(sys.argv[i + 1])
        elif arg == '--num-bath-per-imp' and i + 1 < len(sys.argv):
            args['num_bath_per_imp'] = int(sys.argv[i + 1])
        elif arg == '--onsite' and i + 1 < len(sys.argv):
            args['onsite'] = float(sys.argv[i + 1])
        elif arg == '--U-prime' and i + 1 < len(sys.argv):
            args['U_prime'] = float(sys.argv[i + 1])
        elif arg == '--J-H' and i + 1 < len(sys.argv):
            args['J_H'] = float(sys.argv[i + 1])
        elif arg == '--mu' and i + 1 < len(sys.argv):
            args['mu'] = float(sys.argv[i + 1])
        elif arg == '--hybridization' and i + 1 < len(sys.argv):
            args['hybridization'] = float(sys.argv[i + 1])
        elif arg == '--crystal-field' and i + 1 < len(sys.argv):
            args['crystal_field'] = [float(x) for x in sys.argv[i + 1].split(',')]
        elif arg == '--krylov-dim' and i + 1 < len(sys.argv):
            args['krylov_dim'] = int(sys.argv[i + 1])
        elif arg == '--dt-mult' and i + 1 < len(sys.argv):
            args['dt_mult'] = float(sys.argv[i + 1])
        elif arg == '--shots' and i + 1 < len(sys.argv):
            args['shots'] = int(sys.argv[i + 1])
        elif arg == '--opt-level' and i + 1 < len(sys.argv):
            args['opt_level'] = int(sys.argv[i + 1])
        elif arg == '--max-iter' and i + 1 < len(sys.argv):
            args['max_iter'] = int(sys.argv[i + 1])
        elif arg == '--num-batches' and i + 1 < len(sys.argv):
            args['num_batches'] = int(sys.argv[i + 1])
        elif arg == '--samples-per-batch' and i + 1 < len(sys.argv):
            args['samples_per_batch'] = int(sys.argv[i + 1])
        elif arg == '--system-label' and i + 1 < len(sys.argv):
            args['system_label'] = sys.argv[i + 1]
        elif arg == '--output-dir' and i + 1 < len(sys.argv):
            args['output_dir'] = sys.argv[i + 1]
        elif arg == '--no-persist':
            args['persist'] = False
        elif arg == '--backend' and i + 1 < len(sys.argv):
            args['backend'] = sys.argv[i + 1]
        elif arg == '--backend-type' and i + 1 < len(sys.argv):
            args['backend_type'] = sys.argv[i + 1]
        elif arg == '--coupling-map' and i + 1 < len(sys.argv):
            args['coupling_map'] = sys.argv[i + 1]
        elif arg == '--t1' and i + 1 < len(sys.argv):
            args['t1'] = float(sys.argv[i + 1])
        elif arg == '--t2' and i + 1 < len(sys.argv):
            args['t2'] = float(sys.argv[i + 1])

    return args


def main():
    start_time = time.time()
    args = parse_args()

    from q8020_cfd_qutil import get_backend
    from q8020_cfd_metautil.meta_fragment import (
        generate_experiment_id,
        make_experiment_meta,
        make_case_meta,
        make_code_meta,
        make_backend_meta,
        make_library_meta,
        write_experiment,
        write_case,
        write_code,
        write_backend,
        write_exec_stats,
        write_results,
    )

    M = args['num_imp_orbs']
    B = args['num_bath_per_imp']
    num_orbs = M * (1 + B)
    num_qubits = 2 * num_orbs
    experiment_id = generate_experiment_id()

    print("=" * 60)
    print("SKQD Solver")
    print(f"  experiment_id={experiment_id}")
    print(f"  AIM: {M} imp × {B} bath = {num_orbs} orbs, {num_qubits} qubits")
    print(f"  U={args['onsite']}, U'={args['U_prime']}, J_H={args['J_H']}, "
          f"mu={args['mu']}, V={args['hybridization']}")
    if args['crystal_field']:
        print(f"  crystal_field={args['crystal_field']}")
    print(f"  krylov_dim={args['krylov_dim']}, dt_mult={args['dt_mult']}, "
          f"shots={args['shots']}, opt_level={args['opt_level']}")
    print(f"  max_iter={args['max_iter']}, num_batches={args['num_batches']}, "
          f"samples_per_batch={args['samples_per_batch']}")
    print(f"  backend={args['backend']}, backend_type={args['backend_type']}, "
          f"coupling_map={args['coupling_map']}, t1={args['t1']}, t2={args['t2']}")
    print("=" * 60)

    # Determine if we should persist intermediate results
    output_dir = Path(args['output_dir']) if args['output_dir'] else None
    persist = args['persist'] and output_dir is not None

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Backend setup via qutil ---
    backend = get_backend(
        name=args['backend'],
        backend_type=args['backend_type'],
        t1=args['t1'],
        t2=args['t2'],
        coupling_map=args['coupling_map'],
    )
    print(f"Backend: {backend}")

    # --- Write metadata fragments (experiment, case, code, backend) ---
    if persist:
        exp_meta = make_experiment_meta(
            name="skqd-aim",
            experiment_id=experiment_id,
        )
        write_experiment(output_dir, exp_meta, experiment_id=experiment_id)

        from math import comb
        nelec = num_orbs  # half-filled
        hilbert_dim = comb(num_orbs, nelec // 2) ** 2
        # Resolve effective U'/J_H (step1 defaults: J_H=0, U'=U)
        eff_JH = args['J_H'] if args['J_H'] is not None else 0.0
        eff_Up = args['U_prime'] if args['U_prime'] is not None else args['onsite'] - 2.0 * eff_JH

        case_meta = make_case_meta(
            name="aim",
            system_label=args['system_label'],
            # --- Orbital geometry ---
            num_imp_orbs=M,
            num_bath_per_imp=B,
            num_orbs=num_orbs,
            num_qubits=num_qubits,
            filling="half",
            nelec=nelec,
            nelec_alpha=nelec // 2,
            nelec_beta=nelec // 2,
            hilbert_dim=hilbert_dim,
            # --- Kanamori / AIM interactions ---
            U=args['onsite'],
            U_prime=eff_Up,
            J_H=eff_JH,
            mu=args['mu'],
            hybridization=args['hybridization'],
            crystal_field=args['crystal_field'],
            # --- Krylov / circuit ---
            krylov_dim=args['krylov_dim'],
            dt_mult=args['dt_mult'],
            shots=args['shots'],
            opt_level=args['opt_level'],
            # --- SQD solver ---
            max_iter=args['max_iter'],
            num_batches=args['num_batches'],
            samples_per_batch=args['samples_per_batch'],
        )
        write_case(output_dir, case_meta, experiment_id=experiment_id)

        code_meta = make_code_meta(
            algorithm="skqd",
            entry_point="src/run-skqd.py",
            run_args={k: v for k, v in args.items() if k != 'persist'},
            libraries=make_library_meta(),
        )
        write_code(output_dir, code_meta, experiment_id=experiment_id)

        backend_meta = make_backend_meta(backend)
        write_backend(output_dir, backend_meta, experiment_id=experiment_id)

    # Step 1: AIM Hamiltonian
    print(f"\n--- Step 1: AIM Hamiltonian ({M} imp × {B} bath) ---")
    from step1_siam import run_step1
    pre_start = time.time()
    h1e, h2e = run_step1(
        num_imp_orbs=M,
        num_bath_per_imp=B,
        onsite=args['onsite'],
        hybridization=args['hybridization'],
        mu=args['mu'],
        U_prime=args['U_prime'],
        J_H=args['J_H'],
        crystal_field=args['crystal_field'],
    )
    pre_time = time.time() - pre_start
    print(f"Hamiltonian setup in {pre_time:.2f}s")

    if persist:
        np.save(output_dir / 'h1e.npy', h1e)
        np.save(output_dir / 'h2e.npy', h2e)
        print("  Saved: h1e.npy, h2e.npy")

    # Step 2: Build Krylov circuits
    print(f"\n--- Step 2: Build Krylov Circuits ({num_qubits} qubits) ---")
    from step2_krylov import construct_krylov_siam
    build_start = time.time()
    dt = args['dt_mult'] * np.pi / np.linalg.norm(h1e, ord=2)
    impurity_index = (num_orbs - 1) // 2
    circuits = construct_krylov_siam(
        num_orbs, impurity_index, (h1e, h2e), dt, args['krylov_dim']
    )
    for qc in circuits:
        qc.measure_all()
    build_time = time.time() - build_start
    print(f"Built {len(circuits)} circuits in {build_time:.2f}s")

    if persist:
        from qiskit import qpy
        with open(output_dir / 'circuits.qpy', 'wb') as f:
            qpy.dump(circuits, f)
        circuit_metadata = {
            'dt': dt,
            'impurity_index': impurity_index,
            'krylov_dim': args['krylov_dim'],
            'num_qubits': num_qubits,
        }
        with open(output_dir / 'circuit_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(circuit_metadata, f, indent=2)
        print("  Saved: circuits.qpy, circuit_metadata.json")

    # Step 3: Transpile
    print(f"\n--- Step 3: Transpile for {backend} ---")
    from step3_transpile import transpile_circuits
    transpile_start = time.time()
    transpiled = transpile_circuits(circuits, backend=backend,
                                      optimization_level=args['opt_level'])
    transpile_time = time.time() - transpile_start

    # Circuit depth stats
    depths = [c.depth() for c in transpiled]
    gate_counts = [dict(c.count_ops()) for c in transpiled]
    avg_depth = sum(depths) / len(depths)
    print(f"Transpiled {len(transpiled)} circuits in {transpile_time:.2f}s")
    print(f"Circuit depths: min={min(depths)}, max={max(depths)}, avg={avg_depth:.1f}")

    if persist:
        from qiskit import qpy
        with open(output_dir / 'transpiled_circuits.qpy', 'wb') as f:
            qpy.dump(transpiled, f)
        transpile_stats = {
            'optimization_level': args['opt_level'],
            'backend': str(backend),
            'depths': depths,
            'gate_counts': gate_counts,
        }
        with open(output_dir / 'transpile_stats.json', 'w', encoding='utf-8') as f:
            json.dump(transpile_stats, f, indent=2)
        print("  Saved: transpiled_circuits.qpy, transpile_stats.json")

    # Step 4: Execute
    print(f"\n--- Step 4: Execute ({args['shots']} shots) ---")
    from step4_execute import execute_circuits
    exec_start = time.time()
    bitstrings, probabilities, counts = execute_circuits(transpiled, backend=backend,
        shots=args['shots'])
    exec_time = time.time() - exec_start
    print(f"Executed in {exec_time:.2f}s, unique bitstrings: {len(counts)}")

    if persist:
        with open(output_dir / 'counts.json', 'w', encoding='utf-8') as f:
            json.dump(counts, f)
        np.save(output_dir / 'bitstrings.npy', bitstrings)
        np.save(output_dir / 'probabilities.npy', probabilities)
        print("  Saved: counts.json, bitstrings.npy, probabilities.npy")

    # Step 5: Post-process with SQD
    print("\n--- Step 5: SQD Post-processing ---")
    from step5_postprocess import run_step5, exact_siam_energy

    postprocess_start = time.time()
    result = run_step5(
        counts, num_orbs=num_orbs,
        h1e=h1e, h2e=h2e,
        max_iterations=args['max_iter'],
        num_batches=args['num_batches'],
        samples_per_batch=args['samples_per_batch'],
    )
    postprocess_time = time.time() - postprocess_start

    if persist:
        energy_history = {
            'energies': result,
            'num_iterations': len(result),
        }
        with open(output_dir / 'energy_history.json', 'w', encoding='utf-8') as f:
            json.dump(energy_history, f, indent=2)
        print("  Saved: energy_history.json")

    # Compute exact energy for JSON output
    fci_start = time.time()
    exact_energy = exact_siam_energy(h1e, h2e, num_orbs)
    fci_time = time.time() - fci_start
    print(f"FCI exact energy computed in {fci_time:.2f}s")
    sqd_energy = result[-1]
    error_pct = abs((sqd_energy - exact_energy) / exact_energy) * 100

    total_time = time.time() - start_time

    # Output JSON result line for parsing by sweeper
    result_json = {
        'sqd_energy': sqd_energy,
        'exact_energy': exact_energy,
        'error_pct': error_pct,
        'total_time': total_time,
        'pre_time': pre_time,
        'build_time': build_time,
        'transpile_time': transpile_time,
        'exec_time': exec_time,
        'postprocess_time': postprocess_time,
        'fci_time': fci_time,
        'num_circuits': len(circuits),
        'avg_depth': avg_depth,
    }
    print(f"SKQD_RESULT_JSON:{json.dumps(result_json)}")

    # --- Write exec_stats and results metadata fragments ---
    if persist:
        total_samples = args['shots'] * len(circuits) if args['shots'] > 0 else 0
        exec_stats_data = {
            # --- Timing ---
            'total_time': total_time,
            'pre_time': pre_time,
            'build_time': build_time,
            'transpile_time': transpile_time,
            'exec_time': exec_time,
            'postprocess_time': postprocess_time,
            'fci_time': fci_time,
            # --- Circuit stats ---
            'num_circuits': len(circuits),
            'num_qubits': num_qubits,
            'avg_depth': avg_depth,
            'min_depth': min(depths),
            'max_depth': max(depths),
            'depths': depths,
            'gate_counts': gate_counts,
            # --- Sampling stats ---
            'shots_per_circuit': args['shots'],
            'total_samples': total_samples,
            'unique_bitstrings': len(counts),
        }
        write_exec_stats(output_dir, exec_stats_data, experiment_id=experiment_id)

        error_abs = abs(sqd_energy - exact_energy)
        converged = len(result) < args['max_iter']
        results_data = {
            # --- Energies ---
            'sqd_energy': sqd_energy,
            'exact_energy': exact_energy,
            'error_eV': error_abs,
            'error_meV': error_abs * 1000.0,
            'error_pct': error_pct,
            # --- Convergence ---
            'converged': converged,
            'energy_history': result,
            'num_iterations': len(result),
            'max_iter': args['max_iter'],
            'energy_tol': 1e-4,
        }
        write_results(output_dir, results_data, experiment_id=experiment_id)

    print("\n" + "=" * 60)
    print(f"SKQD Solver Complete (total: {total_time:.1f}s, postprocess: {postprocess_time:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
