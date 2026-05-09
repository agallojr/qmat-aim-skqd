#!/usr/bin/env python3
"""
SKQD Study Runner - Drive multiple SKQD runs from a TOML file.

Usage:
    python run-study-skqd.py <study.toml>

TOML format:
    [global]
    # Default parameters for all cases
    num_orbs = 8
    shots = 1024

    [case1]
    # Override specific parameters
    noise = 0.01

    [case2]
    # List values expand into multiple subcases
    noise = [0.0, 0.01, 0.05]
    krylov_dim = [3, 5, 7]
"""

#pylint: disable=import-outside-toplevel, consider-iterating-dictionary

import json
import subprocess
import sys
import uuid
from pathlib import Path

from q8020_cfd_metautil.sweep import load_sweep_config


def run_case(case_id: str, params: dict, case_output_dir: Path) -> dict:
    """Run a single SKQD case by calling run-skqd.py.
    
    Args:
        case_id: Identifier for this case
        params: Dictionary of parameters (num_orbs, krylov_dim, dt_mult, shots, noise)
        case_output_dir: Directory to write case output files
        
    Returns:
        Dictionary with case metadata
    """
    # Extract parameters with defaults
    num_orbs = params.get('num_orbs', 10)
    krylov_dim = params.get('krylov_dim', 5)
    dt_mult = params.get('dt_mult', 1.0)
    shots = params.get('shots', 1024)
    noise = params.get('noise', 0.0)
    hopping = params.get('hopping', 1.0)
    onsite = params.get('onsite', 5.0)
    hybridization = params.get('hybridization', 1.0)
    filling_factor = params.get('filling_factor', -0.5)
    opt_level = params.get('opt_level', 1)
    max_iter = params.get('max_iter', 10)
    num_batches = params.get('num_batches', 5)
    samples_per_batch = params.get('samples_per_batch', 200)
    
    print(f"\n{'='*60}")
    print(f"Case: {case_id}")
    print("=" * 60)
    
    # Build command to call run-skqd.py
    script_dir = Path(__file__).parent
    run_skqd = script_dir / "run-skqd.py"
    
    cmd = [
        sys.executable, str(run_skqd),
        '--num-orbs', str(num_orbs),
        '--krylov-dim', str(krylov_dim),
        '--dt-mult', str(dt_mult),
        '--shots', str(shots),
        '--noise', str(noise),
        '--hopping', str(hopping),
        '--onsite', str(onsite),
        '--hybridization', str(hybridization),
        '--filling-factor', str(filling_factor),
        '--opt-level', str(opt_level),
        '--max-iter', str(max_iter),
        '--num-batches', str(num_batches),
        '--samples-per-batch', str(samples_per_batch),
        '--output-dir', str(case_output_dir),
    ]
    
    # Run the subprocess and capture output
    result = subprocess.run(cmd, cwd=script_dir, check=False,
                            capture_output=True, text=True)
    
    # Print stdout so user sees progress
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # Parse JSON result from output
    sqd_energy = None
    exact_energy = None
    error_pct = None
    total_time = None
    pre_time = None
    build_time = None
    transpile_time = None
    exec_time = None
    postprocess_time = None
    fci_time = None
    num_circuits = None
    avg_depth = None
    for line in result.stdout.splitlines():
        if line.startswith('SKQD_RESULT_JSON:'):
            json_str = line[len('SKQD_RESULT_JSON:'):]
            data = json.loads(json_str)
            sqd_energy = data.get('sqd_energy')
            exact_energy = data.get('exact_energy')
            error_pct = data.get('error_pct')
            total_time = data.get('total_time')
            pre_time = data.get('pre_time')
            build_time = data.get('build_time')
            transpile_time = data.get('transpile_time')
            exec_time = data.get('exec_time')
            postprocess_time = data.get('postprocess_time')
            fci_time = data.get('fci_time')
            num_circuits = data.get('num_circuits')
            avg_depth = data.get('avg_depth')
            break
    
    return {
        'case_id': case_id,
        'num_orbs': num_orbs,
        'krylov_dim': krylov_dim,
        'dt_mult': dt_mult,
        'shots': shots,
        'noise': noise,
        'hopping': hopping,
        'onsite': onsite,
        'hybridization': hybridization,
        'filling_factor': filling_factor,
        'opt_level': opt_level,
        'max_iter': max_iter,
        'num_batches': num_batches,
        'samples_per_batch': samples_per_batch,
        'returncode': result.returncode,
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
        'num_circuits': num_circuits,
        'avg_depth': avg_depth,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run-study-skqd.py <study.toml>")
        sys.exit(1)
    
    toml_path = sys.argv[1]
    
    # Load cases from TOML (using q8020-cfd-metautil sweep config loader)
    config = load_sweep_config(toml_path)
    global_config = config["global"]
    # Flatten into the {case_id: params} dict expected downstream
    cases = {"global": global_config}
    for group_data in config["groups"].values():
        for case_id, case_params in group_data["expanded_cases"].items():
            cases[case_id] = case_params
    
    # Get output directory from global config
    output_base = global_config.get('output_dir', 'output')
    # Expand ~ to user home directory
    output_base = Path(output_base).expanduser()
    
    # Create study directory with short UUID (8 chars)
    study_id = uuid.uuid4().hex[:8]
    study_dir = Path(output_base) / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy input TOML to study directory
    import shutil
    shutil.copy(toml_path, study_dir / Path(toml_path).name)
    
    print("=" * 60)
    print("SKQD Study Runner")
    print(f"  TOML: {toml_path}")
    print(f"  Study ID: {study_id}")
    print(f"  Output: {study_dir}")
    print("=" * 60)
    
    # Filter out 'global' - it's already merged into each case
    case_ids = [k for k in cases.keys() if k != 'global']
    
    print(f"\nFound {len(case_ids)} case(s) to run:")
    for case_id in case_ids:
        params = cases[case_id]
        # Filter out metadata for display
        display_params = {k: v for k, v in params.items() if not k.startswith('_')}
        print(f"  {case_id}: {display_params}")
    
    # Run each case
    results = []
    for case_id in case_ids:
        params = cases[case_id]
        # Create case output directory
        case_output_dir = study_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)
        result = run_case(case_id, params, case_output_dir)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    print(f"{'Case':<20} {'Orbs':>5} {'Krylov':>6} {'dt':>5} {'Noise':>6} {'Shots':>6} {'Opt':>3} "
          f"{'Iter':>4} {'Bat':>3} {'Samp':>4} "
          f"{'hop':>5} {'U':>5} {'hyb':>5} {'fill':>6} "
          f"{'Circ':>4} {'Qubits':>6} {'Depth':>5} "
          f"{'Energy':>12} {'Exact':>12} {'Err%':>8} "
          f"{'Post':>6}")
    print("-" * 175)
    for r in results:
        energy_str = f"{r['sqd_energy']:.4f}" if r['sqd_energy'] is not None else 'N/A'
        exact_str = f"{r['exact_energy']:.4f}" if r['exact_energy'] is not None else 'N/A'
        error_str = f"{r['error_pct']:.4f}" if r['error_pct'] is not None else 'N/A'
        post_str = f"{r['postprocess_time']:.1f}s" if r['postprocess_time'] is not None else 'N/A'
        circ_str = f"{r['num_circuits']}" if r['num_circuits'] is not None else 'N/A'
        depth_str = f"{r['avg_depth']:.0f}" if r['avg_depth'] is not None else 'N/A'
        qubits = 2 * r['num_orbs']
        print(f"{r['case_id']:<20} {r['num_orbs']:>5} {r['krylov_dim']:>6} "
              f"{r['dt_mult']:>5.1f} {r['noise']:>6.3f} {r['shots']:>6} {r['opt_level']:>3} "
              f"{r['max_iter']:>4} {r['num_batches']:>3} {r['samples_per_batch']:>4} "
              f"{r['hopping']:>5.1f} {r['onsite']:>5.1f} {r['hybridization']:>5.1f} {r['filling_factor']:>6.2f} "
              f"{circ_str:>4} {qubits:>6} {depth_str:>5} "
              f"{energy_str:>12} {exact_str:>12} {error_str:>8} "
              f"{post_str:>6}")
    
    print("\n" + "=" * 60)
    print("SKQD Study Complete")
    print("=" * 60)
    
    # Persist study summary to study directory
    study_summary = {
        'study_id': study_id,
        'num_cases': len(results),
        'cases': results,
    }
    summary_path = study_dir / 'study_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(study_summary, f, indent=2)
    print(f"\nStudy summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
