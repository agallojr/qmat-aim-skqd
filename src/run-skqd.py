#!/usr/bin/env python3
"""
SKQD Solver — Anderson Impurity Model workflow.

Quantum-classical hybrid solver for ground-state energies of the multi-orbital
Anderson Impurity Model with rotationally-invariant Kanamori interactions.
The quantum side prepares Krylov-evolved states and samples bitstrings; the
classical side diagonalizes H projected onto the sampled subspace via SCI.

USAGE
    python src/run-skqd.py [OPTIONS]
    python src/run-skqd.py --help

The geometry is fully specified by --num-imp-orbs (M correlated orbitals) and
--num-bath-per-imp (B bath sites per impurity orbital, star geometry, no
bath-bath hopping). Total spatial orbitals = M * (1 + B); qubits = 2 * orbitals.
Single-orbital Anderson is the special case --num-imp-orbs 1.

All energies are in eV. Chemists: see the README "Physics scope" section for
the explicit second-quantized Hamiltonian and known limitations (Kanamori
only, no spin-orbit coupling, diagonal crystal field, ground state only).

==============================================================================
HAMILTONIAN PARAMETERS
==============================================================================

  --num-imp-orbs M                Number of correlated impurity orbitals.
                                  [default: 1]

  --num-bath-per-imp B            Bath sites per impurity orbital. Star
                                  geometry: every bath site couples directly
                                  and only to its impurity orbital.
                                  [default: 1]

  --onsite U                      Intra-orbital Coulomb repulsion U (eV).
                                  Acts on the impurity orbitals.
                                  [default: 4.0]

  --U-prime UP                    Inter-orbital Coulomb U' (eV) between
                                  impurity orbitals. The rotationally-
                                  invariant Kanamori relation is
                                  U' = U - 2*J_H, which is used by default.
                                  [default: U - 2*J_H]

  --J-H J                         Hund's exchange coupling J_H (eV). Drives
                                  same-spin density attraction (-J_H term)
                                  plus the off-diagonal spin-flip and
                                  pair-hopping terms in the impurity block.
                                  [default: 0.0]

  --mu MU                         Chemical potential (eV). Subtracted from
                                  every diagonal one-body entry (impurity
                                  and bath).
                                  [default: 2.0]

  --hybridization V               Scalar impurity-bath coupling V (eV),
                                  broadcast across all M*B bath sites.
                                  Ignored if --bath-couplings is given.
                                  [default: 0.8]

  --bath-energies "e1,e2,..."     Per-bath-site on-site energies (eV),
                                  comma-separated, length M*B, row-major
                                  over (impurity m, bath site b). For B=1
                                  this is just M values. Default is all-zero
                                  (legacy behavior); pair with non-zero
                                  values when B>1 to get distinct bath
                                  channels rather than degenerate ones.
                                  [default: zeros]

  --bath-couplings "v1,v2,..."    Per-bath-site coupling V (eV),
                                  comma-separated, length M*B, row-major
                                  over (m, b). When given, overrides the
                                  --hybridization broadcast.
                                  [default: broadcast --hybridization]

  --crystal-field "e1,e2,..."     Diagonal crystal-field eigenvalues (eV)
                                  for each impurity orbital, comma-separated,
                                  length M. The orbital basis is whatever
                                  the user chose; CF must be diagonal in
                                  that basis (off-diagonal CF rotations are
                                  not currently supported).
                                  [default: none, all CF eigenvalues = 0]

==============================================================================
PARTICLE SECTOR
==============================================================================

  --n-electrons N                 Total electron count. Splits evenly between
                                  spins (alpha takes the extra when odd).
                                  [default: num_orbs (half-filling)]

  --n-electrons-alpha N           Spin-up (alpha) electron count. Wins over
                                  --n-electrons if both given.
                                  [default: from --n-electrons]

  --n-electrons-beta N            Spin-down (beta) electron count. Wins over
                                  --n-electrons if both given.
                                  [default: from --n-electrons]

  Examples:
    Ce 4f^1:   --n-electrons-alpha 1 --n-electrons-beta 0
    Pr 4f^2:   --n-electrons-alpha 2 --n-electrons-beta 0   (high-spin S=1)
    Half-fill: omit all three (legacy default)

==============================================================================
KRYLOV / TROTTER PARAMETERS
==============================================================================

  --krylov-dim K                  Number of Krylov basis states |psi_k>,
                                  where |psi_k> = T(dt)^k |psi_0>.
                                  [default: 5]

  --dt-mult X                     Time-step multiplier:
                                    dt = pi * X / scale,
                                  where 'scale' depends on --dt-scale-mode.
                                  [default: 1.0]

  --dt-scale-mode {h1e,full}      How to set the dt scale:
                                    h1e   scale = ||h1e||_2  (legacy; the
                                          empirically-good large-dt regime
                                          for SCI subspace coverage)
                                    full  scale = ||h1e||_2 + max|h2e|
                                          (Trotter-accurate; produces
                                          tightly-clustered Krylov states.
                                          Pair with --trotter-substeps and
                                          a larger --dt-mult for SQD use.)
                                  [default: h1e]

  --trotter-order {1,2,4}         Suzuki product-formula order:
                                    1  Lie-Trotter (first-order)
                                    2  Strang (second-order, default)
                                    4  Suzuki S4 (fourth-order, ~5x deeper)
                                  [default: 2]

  --trotter-substeps N            Subdivide each Krylov increment into N
                                  equal sub-steps of size dt/N. Per-sub-step
                                  local Trotter error scales as
                                  (dt/N)^(order+1). [default: 1]

  --check-trotter                 Pre-flight diagnostic: compute
                                  ||T(dt) - exp(-i H dt)||_2 on the
                                  half-filled sector. Restricted to
                                  num_orbs <= 6. Reports the rigorous
                                  fidelity bound 1 - epsilon^2.
                                  [default: off]

==============================================================================
SQD POST-PROCESSING PARAMETERS
==============================================================================

  --max-iter N                    Maximum SQD self-consistent iterations.
                                  [default: 10]

  --num-batches N                 Number of batches for subspace
                                  construction. [default: 5]

  --samples-per-batch N           Samples drawn per batch from the bitstring
                                  distribution. [default: 200]

==============================================================================
EXECUTION / BACKEND PARAMETERS
==============================================================================

  --shots N                       Measurement shots per circuit. Use 0 for
                                  exact statevector sampling on the sim
                                  backend. [default: 1024]

  --opt-level {0,1,2,3}           Qiskit transpiler optimization level.
                                  [default: 1]

  --backend NAME                  Fake backend name (for topology + noise),
                                  e.g. 'manila', 'jakarta', 'brisbane', or
                                  'ibm_brisbane' for hardware mode.
                                  [default: none, full connectivity]

  --backend-type {sim,fake,hardware}
                                  Execution mode:
                                    sim       AerSimulator (default)
                                    fake      FakeBackendV2 with calibrated
                                              noise
                                    hardware  Real IBM Quantum backend
                                  [default: sim]

  --coupling-map {default,all-to-all}
                                  default     backend's native coupling
                                  all-to-all  full connectivity
                                  [default: default]

  --t1 N                          Override T1 relaxation time (microseconds).
                                  Applied on top of the backend noise model.
                                  [default: from backend]

  --t2 N                          Override T2 dephasing time (microseconds).
                                  [default: from backend]

==============================================================================
OUTPUT / METADATA
==============================================================================

  --output-dir DIR                Directory for per-case artifacts and q8020
                                  metadata fragments. [default: none]

  --system-label STR              Free-form chemical system label written
                                  into output metadata, e.g. "Ce 4f^1",
                                  "Co 3d / Cu". [default: none]

  --no-persist                    Skip writing intermediate result files.
                                  [default: persist on]

  -h, --help                      Print this help message and exit.

==============================================================================
EXAMPLES
==============================================================================

  # Single-orbital Anderson at half-filling: 8 qubits, fast smoke test
  python src/run-skqd.py --num-imp-orbs 1 --num-bath-per-imp 3 --shots 4096

  # Ce 4f^1 — proper atomic filling (1 electron, spin-up)
  python src/run-skqd.py \\
      --num-imp-orbs 5 --num-bath-per-imp 1 \\
      --n-electrons-alpha 1 --n-electrons-beta 0 \\
      --onsite 6.0 --U-prime 4.6 --J-H 0.7 --mu 3.0 --hybridization 0.4 \\
      --crystal-field "-0.15,-0.05,0.0,0.05,0.15" \\
      --shots 65536

  # Pr 4f^2 high-spin (Hund's-rule S=1 ground state)
  python src/run-skqd.py \\
      --num-imp-orbs 5 --num-bath-per-imp 1 \\
      --n-electrons-alpha 2 --n-electrons-beta 0 \\
      --onsite 6.5 --U-prime 5.0 --J-H 0.75 --mu 3.0 --hybridization 0.3 \\
      --crystal-field "-0.15,-0.05,0.0,0.05,0.15" \\
      --shots 65536

  # B=2 bath with distinct (eps, V) per site
  python src/run-skqd.py \\
      --num-imp-orbs 2 --num-bath-per-imp 2 \\
      --bath-energies "-1.0,1.0,-1.0,1.0" \\
      --bath-couplings "0.5,0.2,0.5,0.2" \\
      --shots 16384

  # Trotter pre-flight on a 4-orbital case before scaling up
  python src/run-skqd.py --num-imp-orbs 1 --num-bath-per-imp 3 \\
      --trotter-order 4 --trotter-substeps 2 --check-trotter --shots 0

==============================================================================
INTEGRATION WITH q8020-sweep
==============================================================================

Most users invoke this script via `q8020-sweep input/*.toml` (from the
q8020-cfd-metautil package), which expands list-valued TOML parameters into
a cross-product of cases and runs them in batch. When invoked that way:

  * --output-dir is INJECTED automatically per case by the sweeper (driven
    by the TOML's `_inject_outdir = "--output-dir"` line). You do not need
    to pass it; do not pass it. Each case lands in
    `<_output_dir>/<YYYY-MM-DD>/_<wf-id>/<case-name>/`.

  * The TOML's underscore-prefixed keys (`_output_dir`, `_script`,
    `_inject_outdir`, `_final_postproc`) are sweeper directives — they do
    NOT reach this script. Only `--`-prefixed TOML keys become CLI args.

  * The q8020_case_*.json, q8020_code_*.json, q8020_backend_*.json,
    q8020_exec_stats_*.json, and q8020_results_*.json metadata fragments
    you find in each case directory are written by the sweeper's
    instrumentation hooks, NOT by this script. This script writes the
    physics artifacts (h1e.npy, h2e.npy, circuits.qpy, counts.json,
    bitstrings.npy, energy_history.json) and emits one
    `SKQD_RESULT_JSON:{...}` line on stdout that the sweeper parses.

  * `_final_postproc` commands (typically `python src/plot_convergence.py`)
    run once after the entire sweep completes, with the run directory
    passed in. See plot_convergence.py for that side.

For standalone runs (no sweeper), pass --output-dir manually if you want
artifacts persisted; otherwise only the SKQD_RESULT_JSON line goes to stdout.

For sweep configurations and the high-level workflow, see the README.
"""

#pylint: disable=import-outside-toplevel, unused-variable

import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    """Parse command line arguments. -h / --help prints the module docstring."""
    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        print(__doc__)
        sys.exit(0)

    args = {
        'num_imp_orbs': 1,
        'num_bath_per_imp': 1,
        'onsite': 4.0,
        'U_prime': None,
        'J_H': None,
        'mu': 2.0,
        'hybridization': 0.8,
        'bath_energies': None,
        'bath_couplings': None,
        'n_electrons': None,
        'n_electrons_alpha': None,
        'n_electrons_beta': None,
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
        'trotter_order': 2,
        'trotter_substeps': 1,
        # 'h1e' (legacy) sets dt = pi * dt_mult / ||h1e||_2 — SQD wants the
        # large-dt regime so Krylov states spread across the Hilbert space.
        # 'full' sets dt = pi * dt_mult / (||h1e||_2 + max|h2e|) — more
        # Trotter-accurate but produces tightly clustered Krylov states; pair
        # with --trotter-substeps and a higher --dt-mult for SQD use.
        'dt_scale_mode': 'h1e',
        'check_trotter': False,
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
        elif arg == '--bath-energies' and i + 1 < len(sys.argv):
            args['bath_energies'] = [float(x) for x in sys.argv[i + 1].split(',')]
        elif arg == '--bath-couplings' and i + 1 < len(sys.argv):
            args['bath_couplings'] = [float(x) for x in sys.argv[i + 1].split(',')]
        elif arg == '--n-electrons' and i + 1 < len(sys.argv):
            args['n_electrons'] = int(sys.argv[i + 1])
        elif arg == '--n-electrons-alpha' and i + 1 < len(sys.argv):
            args['n_electrons_alpha'] = int(sys.argv[i + 1])
        elif arg == '--n-electrons-beta' and i + 1 < len(sys.argv):
            args['n_electrons_beta'] = int(sys.argv[i + 1])
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
        elif arg == '--trotter-order' and i + 1 < len(sys.argv):
            args['trotter_order'] = int(sys.argv[i + 1])
        elif arg == '--trotter-substeps' and i + 1 < len(sys.argv):
            args['trotter_substeps'] = int(sys.argv[i + 1])
        elif arg == '--dt-scale-mode' and i + 1 < len(sys.argv):
            args['dt_scale_mode'] = sys.argv[i + 1]
        elif arg == '--check-trotter':
            args['check_trotter'] = True

    return args


def _resolve_nelec(args: dict, num_orbs: int) -> tuple[int, int]:
    """Resolve (n_alpha, n_beta) from CLI args.

    Resolution rule:
      * If both --n-electrons-alpha and --n-electrons-beta are given, they win
        (and --n-electrons is ignored).
      * Else if --n-electrons is given, split evenly (alpha gets the extra
        when odd).
      * Else fall back to half-filling: n_alpha = n_beta = num_orbs // 2.

    Validates 0 <= n_alpha, n_beta <= num_orbs.
    """
    a, b, total = (
        args.get('n_electrons_alpha'),
        args.get('n_electrons_beta'),
        args.get('n_electrons'),
    )
    if a is not None and b is not None:
        n_alpha, n_beta = int(a), int(b)
    elif total is not None:
        n_total = int(total)
        n_alpha = (n_total + 1) // 2
        n_beta = n_total - n_alpha
    else:
        n_alpha = num_orbs // 2
        n_beta = num_orbs // 2

    if not (0 <= n_alpha <= num_orbs and 0 <= n_beta <= num_orbs):
        raise ValueError(
            f"Invalid filling: (n_alpha={n_alpha}, n_beta={n_beta}) outside "
            f"[0, {num_orbs}]"
        )
    return n_alpha, n_beta


def _resolve_bath_arrays(
    args: dict, M: int, B: int
) -> tuple[list[float] | None, list[float] | None, bool]:
    """Validate and return (bath_energies, bath_couplings, used_defaults).

    Returns the lists as-given (None preserves the default codepath inside
    step1).  ``used_defaults`` is True when both arrays are absent — used
    to gate the B>1 degeneracy warning.
    """
    eps = args.get('bath_energies')
    V = args.get('bath_couplings')
    if eps is not None and len(eps) != M * B:
        raise ValueError(
            f"--bath-energies length {len(eps)} must equal M*B={M*B}"
        )
    if V is not None and len(V) != M * B:
        raise ValueError(
            f"--bath-couplings length {len(V)} must equal M*B={M*B}"
        )
    used_defaults = eps is None and V is None
    return eps, V, used_defaults


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

    n_alpha, n_beta = _resolve_nelec(args, num_orbs)
    bath_energies, bath_couplings, defaults_used = _resolve_bath_arrays(
        args, M, B
    )
    if B > 1 and defaults_used:
        import warnings
        warnings.warn(
            f"B={B} with degenerate bath: only one bonding combination "
            f"couples to the impurity, leaving {B - 1} bath orbital(s) per "
            f"impurity decoupled. Consider passing --bath-energies and/or "
            f"--bath-couplings (length M*B={M*B}) when B > 1.",
            stacklevel=2,
        )

    print("=" * 60)
    print("SKQD Solver")
    print(f"  experiment_id={experiment_id}")
    print(f"  AIM: {M} imp × {B} bath = {num_orbs} orbs, {num_qubits} qubits")
    print(f"  U={args['onsite']}, U'={args['U_prime']}, J_H={args['J_H']}, "
          f"mu={args['mu']}, V={args['hybridization']}")
    print(f"  nelec=({n_alpha}, {n_beta})  (total {n_alpha + n_beta})")
    if bath_energies is not None:
        print(f"  bath_energies={bath_energies}")
    if bath_couplings is not None:
        print(f"  bath_couplings={bath_couplings}")
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
    if persist and output_dir is not None:
        exp_meta = make_experiment_meta(
            name="skqd-aim",
            experiment_id=experiment_id,
        )
        write_experiment(output_dir, exp_meta, experiment_id=experiment_id)

        from math import comb
        nelec_total = n_alpha + n_beta
        hilbert_dim = comb(num_orbs, n_alpha) * comb(num_orbs, n_beta)
        is_half = (n_alpha == n_beta == num_orbs // 2)
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
            filling="half" if is_half else "custom",
            nelec=nelec_total,
            nelec_alpha=n_alpha,
            nelec_beta=n_beta,
            hilbert_dim=hilbert_dim,
            # --- Kanamori / AIM interactions ---
            U=args['onsite'],
            U_prime=eff_Up,
            J_H=eff_JH,
            mu=args['mu'],
            hybridization=args['hybridization'],
            bath_energies=bath_energies,
            bath_couplings=bath_couplings,
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
        bath_energies=bath_energies,
        bath_couplings=bath_couplings,
    )
    pre_time = time.time() - pre_start
    print(f"Hamiltonian setup in {pre_time:.2f}s")

    if persist and output_dir is not None:
        np.save(output_dir / 'h1e.npy', h1e)
        np.save(output_dir / 'h2e.npy', h2e)
        print("  Saved: h1e.npy, h2e.npy")

    # Step 2: Build Krylov circuits
    print(f"\n--- Step 2: Build Krylov Circuits ({num_qubits} qubits) ---")
    from step2_krylov import construct_krylov_siam, _resolve_dt
    import skqd_helpers
    build_start = time.time()
    dt, omega_h1e, omega_h2e, omega_total = _resolve_dt(
        h1e, h2e, args['dt_mult'], args['dt_scale_mode']
    )
    impurity_index = (num_orbs - 1) // 2
    print(
        f"  Trotter: order={args['trotter_order']}, "
        f"substeps={args['trotter_substeps']}, "
        f"dt_scale_mode={args['dt_scale_mode']}, "
        f"omega_h1e={omega_h1e:.4f}, omega_h2e={omega_h2e:.4f}, "
        f"omega_total={omega_total:.4f}, dt={dt:.4f}"
    )

    trotter_distance = None
    if args['check_trotter']:
        if num_orbs <= 6:
            try:
                trotter_distance = skqd_helpers.compute_trotter_distance(
                    h1e, h2e, dt,
                    num_imp_orbs=M,
                    order=args['trotter_order'],
                    substeps=args['trotter_substeps'],
                )
                fidelity_bound = max(0.0, 1.0 - trotter_distance ** 2)
                print(
                    f"  Trotter pre-flight: ||T(dt)-exp(-iHdt)||_2 = "
                    f"{trotter_distance:.4e}  (fidelity_per_step >= "
                    f"{fidelity_bound:.6f})"
                )
            except Exception as e:
                print(f"  Trotter pre-flight skipped: {e}")
        else:
            print(
                f"  Trotter pre-flight skipped: num_orbs={num_orbs} > 6 "
                f"(would need 2^{2*num_orbs} x 2^{2*num_orbs} matrices)"
            )

    circuits = construct_krylov_siam(
        num_orbs, impurity_index, (h1e, h2e), dt, args['krylov_dim'],
        num_imp_orbs=M,
        trotter_order=args['trotter_order'],
        trotter_substeps=args['trotter_substeps'],
        n_alpha=n_alpha, n_beta=n_beta,
    )
    for qc in circuits:
        qc.measure_all()
    build_time = time.time() - build_start
    print(f"Built {len(circuits)} circuits in {build_time:.2f}s")

    if persist and output_dir is not None:
        from qiskit import qpy
        with open(output_dir / 'circuits.qpy', 'wb') as f:
            qpy.dump(circuits, f)
        circuit_metadata = {
            'dt': dt,
            'impurity_index': impurity_index,
            'krylov_dim': args['krylov_dim'],
            'num_qubits': num_qubits,
            'trotter_order': args['trotter_order'],
            'trotter_substeps': args['trotter_substeps'],
            'dt_scale_mode': args['dt_scale_mode'],
            'omega_h1e': omega_h1e,
            'omega_h2e': omega_h2e,
            'omega_total': omega_total,
            'trotter_distance': trotter_distance,
            'nelec_alpha': n_alpha,
            'nelec_beta': n_beta,
            'bath_energies': bath_energies,
            'bath_couplings': bath_couplings,
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

    if persist and output_dir is not None:
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

    if persist and output_dir is not None:
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
        n_alpha=n_alpha, n_beta=n_beta,
    )
    postprocess_time = time.time() - postprocess_start

    if persist and output_dir is not None:
        energy_history = {
            'energies': result,
            'num_iterations': len(result),
        }
        with open(output_dir / 'energy_history.json', 'w', encoding='utf-8') as f:
            json.dump(energy_history, f, indent=2)
        print("  Saved: energy_history.json")

    # Compute exact energy for JSON output
    fci_start = time.time()
    exact_energy = exact_siam_energy(
        h1e, h2e, num_orbs, n_alpha=n_alpha, n_beta=n_beta
    )
    fci_time = time.time() - fci_start
    print(f"FCI exact energy computed in {fci_time:.2f}s")
    sqd_energy = result[-1]
    if abs(exact_energy) > 1e-15:
        error_pct = abs((sqd_energy - exact_energy) / exact_energy) * 100
    else:
        error_pct = 0.0

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
        'trotter_order': args['trotter_order'],
        'trotter_substeps': args['trotter_substeps'],
        'dt_scale_mode': args['dt_scale_mode'],
        'dt': dt,
        'omega_total': omega_total,
        'trotter_distance': trotter_distance,
        'nelec_alpha': n_alpha,
        'nelec_beta': n_beta,
        'bath_energies': bath_energies,
        'bath_couplings': bath_couplings,
    }
    print(f"SKQD_RESULT_JSON:{json.dumps(result_json)}")

    # --- Write exec_stats and results metadata fragments ---
    if persist and output_dir is not None:
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
