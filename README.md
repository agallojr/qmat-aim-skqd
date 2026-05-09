# SKQD-AIM: Sample-based Krylov Quantum Diagonalization for Anderson Impurity Models

Quantum-classical hybrid algorithm for computing ground state energies of Anderson Impurity Models (AIM) using Sample-based Krylov Quantum Diagonalization (SKQD).  Supports single-orbital and multi-orbital impurities with Kanamori interactions (U, U′, J_H), crystal-field splitting, and per-orbital bath hybridization.

## Overview

### Sample-based Quantum Diagonalization (SQD)

Traditional variational quantum eigensolvers (VQE) estimate ground state energies by measuring expectation values of the Hamiltonian, which requires many circuit executions and is sensitive to noise. **Sample-based Quantum Diagonalization (SQD)** takes a different approach:

1. **Sample bitstrings** from a quantum circuit that prepares a state with significant overlap with the ground state
2. **Build a subspace** from the sampled computational basis states
3. **Classically diagonalize** the Hamiltonian within this subspace

The key insight is that if the ground state has support on a relatively small number of computational basis states, we can identify those states through sampling and then solve the eigenvalue problem classically. This is particularly effective for:
- States with sparse structure in the computational basis
- Problems where noise corrupts expectation values but not the identity of sampled states
- Systems where classical diagonalization in the sampled subspace is tractable

### Krylov Subspace Enhancement (SKQD)

**SKQD** enhances SQD by using Krylov subspace methods to generate the quantum states for sampling. Instead of a single variational ansatz, SKQD:

1. Prepares an initial reference state |ψ₀⟩
2. Applies powers of the time evolution operator e^{-iHt} to generate Krylov basis states: |ψ₀⟩, e^{-iHt}|ψ₀⟩, e^{-2iHt}|ψ₀⟩, ...
3. Samples bitstrings from each Krylov state
4. Combines all samples to build a richer subspace for classical diagonalization

The Krylov approach systematically explores the relevant Hilbert space and often captures ground state components more effectively than a single ansatz.

### Application to the Anderson Impurity Model

The **Anderson Impurity Model** describes correlated impurity orbitals coupled to a bath of conduction electrons. It is a fundamental model in condensed matter physics for understanding:
- Kondo physics and heavy fermion systems
- Quantum dots coupled to leads
- Magnetic impurities in metals (e.g. Co 3d in Cu)
- f-electron systems (e.g. Ce 4f in CeCoIn₅)

All problems are specified via two structural parameters:
- **M** (`--num-imp-orbs`): Number of impurity (correlated) orbitals
- **B** (`--num-bath-per-imp`): Bath sites per impurity orbital

Total spatial orbitals = M × (1 + B), qubits = 2 × orbitals.  Single-orbital star geometry is the special case M=1.

The Hamiltonian supports full Kanamori interactions:
- **U** (`--onsite`): Intra-orbital Coulomb repulsion
- **U′** (`--U-prime`): Inter-orbital Coulomb (defaults to U − 2J_H)
- **J_H** (`--J-H`): Hund's exchange coupling
- **μ** (`--mu`): Chemical potential
- **V** (`--hybridization`): Impurity-bath coupling
- **ε_CF** (`--crystal-field`): Crystal-field splitting per impurity orbital

Star geometry: every bath site couples directly to its impurity orbital (no bath-bath hopping).

## Getting Started

### 1. Install uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it with:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### 2. Create and activate a virtual environment

```bash
cd qmat-aim-skqd
uv venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Run a single case directly

```bash
# Single-orbital: 1 imp + 3 bath = 4 orbs = 8 qubits
python src/run-skqd.py --num-imp-orbs 1 --num-bath-per-imp 3 --shots 1024

# Multi-orbital Ce 4f: 3 imp + 1 bath each = 6 orbs = 12 qubits
python src/run-skqd.py --num-imp-orbs 3 --num-bath-per-imp 1 \
    --onsite 6.0 --U-prime 4.6 --J-H 0.7 --mu 3.0 --hybridization 0.5 \
    --crystal-field -0.1,0.0,0.1 --shots 5000
```

### 5. Run a parameter sweep (via q8020-sweep)

```bash
q8020-sweep input/smoke-test.toml          # quick 8-qubit sanity check
q8020-sweep input/ce-4f.toml               # 12-qubit multi-orbital
q8020-sweep input/ce-4f-stretch.toml       # 20-qubit stretch case
q8020-sweep input/smoke-test.toml --dry-run
```

The sweeper expands list-valued parameters into a cross-product of cases, calls `run-skqd.py` for each, and collects results.

## Workflow

The algorithm proceeds in five steps:

1. **Step 1** (`src/step1_siam.py`): Build AIM Hamiltonian (Kanamori interactions, crystal field)
2. **Step 2** (`src/step2_krylov.py`): Construct Krylov circuits using Trotterized time evolution
3. **Step 3** (`src/step3_transpile.py`): Transpile circuits for the target backend
4. **Step 4** (`src/step4_execute.py`): Execute circuits and collect bitstring samples
5. **Step 5** (`src/step5_postprocess.py`): Classical SQD diagonalization to extract ground state energy

## CLI Reference

```
python src/run-skqd.py [OPTIONS]
```

### Krylov Circuit Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--krylov-dim` | int | 5 | Number of Krylov basis states to generate |
| `--dt-mult` | float | 1.0 | Time step multiplier for Trotter evolution |

### Execution Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--shots` | int | 1024 | Number of measurement shots per circuit |
| `--opt-level` | int | 1 | Qiskit transpiler optimization level (0-3) |

### Backend Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--backend` | str | none | Fake backend name for topology/noise (e.g. `manila`, `jakarta`, `brisbane`) |
| `--backend-type` | str | sim | Execution mode: `sim` (AerSimulator), `fake` (FakeBackendV2), `hardware` (IBM Quantum) |
| `--coupling-map` | str | default | Coupling map: `default` (backend native) or `all-to-all` (full connectivity) |
| `--t1` | float | none | T1 relaxation time in µs (overrides backend noise model) |
| `--t2` | float | none | T2 dephasing time in µs (overrides backend noise model) |

Backend modes:
- **No `--backend`**: Ideal statevector simulation, no noise, full connectivity
- **`--backend manila`**: AerSimulator with Manila's topology + calibrated noise model
- **`--backend manila --t1 50 --t2 70`**: Manila topology with custom thermal relaxation
- **`--backend manila --coupling-map all-to-all`**: Manila noise but full connectivity
- **`--backend-type hardware --backend ibm_brisbane`**: Real IBM Quantum hardware

### Hamiltonian Parameters

Unified AIM interface: single-orbital star geometry is `--num-imp-orbs 1`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-imp-orbs` | int | 1 | Number of impurity orbitals M |
| `--num-bath-per-imp` | int | 1 | Bath sites per impurity orbital B |
| `--onsite` | float | 4.0 | Intra-orbital Coulomb repulsion U (eV) |
| `--U-prime` | float | U−2J_H | Inter-orbital Coulomb U′ (eV) |
| `--J-H` | float | 0.0 | Hund's exchange coupling (eV) |
| `--mu` | float | 2.0 | Chemical potential (eV) |
| `--hybridization` | float | 0.8 | Impurity-bath coupling V (eV) |
| `--crystal-field` | str | none | Crystal-field energies, comma-separated (eV) |

Total orbitals = M × (1 + B).  Qubits = 2 × orbitals.  Half-filling assumed (N_elec = N_orbs).

### SQD Post-processing Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-iter` | int | 10 | Maximum SQD self-consistent iterations |
| `--num-batches` | int | 5 | Number of batches for subspace construction |
| `--samples-per-batch` | int | 200 | Samples drawn per batch |

### Output Control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | path | none | Directory for case output files |
| `--no-persist` | flag | false | Disable saving intermediate results |

## TOML Sweep Configuration

TOML files in `input/` are formatted for [q8020-sweep](https://github.com/Q8020-CFD/q8020-cfd-metautil). Example:

```toml
[global]
_output_dir = "~/output"
_script = "python src/run-skqd.py"
_inject_outdir = "--output-dir"

"--num-imp-orbs" = 1       # M=1 impurity orbital
"--num-bath-per-imp" = 3   # B=3 bath sites → 4 orbs, 8 qubits
"--krylov-dim" = 3
"--onsite" = 4.0           # U (eV)
"--mu" = 2.0               # chemical potential (eV)
"--hybridization" = 0.8    # V (eV)
"--backend-type" = "sim"

[statevector]
"--shots" = 0              # exact statevector sampling

[sampled]
"--shots" = [1024, 4096]
```

List values expand into a cross-product of cases. Group-level keys override globals.

### Available configurations

| File | System | Qubits | Hilbert dim | Purpose |
|------|--------|--------|-------------|---------|
| `smoke-test.toml` | Co 3d / Cu (1 imp × 3 bath) | 8 | 36 | Quick sanity check |
| `small-sys.toml` | Co 3d / Cu (1 imp × 5 bath) | 12 | 400 | Baseline single-orbital |
| `convergence-study.toml` | AIM (1 imp × 7 bath) | 16 | 4,900 | Shot-count sensitivity |
| `ce-4f.toml` | Ce 4f (3 imp × 1 bath) | 12 | 400 | Multi-orbital Kanamori |
| `ce-4f-stretch.toml` | Ce 4f (5 imp × 1 bath) | 20 | 63,504 | Stretch — ansatz-limited |

## Output

Each case emits a JSON result line on stdout:

```
SKQD_RESULT_JSON:{"sqd_energy": -12.345, "exact_energy": -12.346, "error_pct": 0.008, ...}
```

When `--output-dir` is provided, the solver writes:
- **Hamiltonian**: `h1e.npy`, `h2e.npy`
- **Circuits**: `circuits.qpy`, `transpiled_circuits.qpy`, `circuit_metadata.json`, `transpile_stats.json`
- **Execution**: `counts.json`, `bitstrings.npy`, `probabilities.npy`
- **Results**: `energy_history.json`

**Metadata fragments** (q8020 format) capture open-box reproducibility info:

| Fragment | Contents |
|----------|----------|
| `q8020_case_*.json` | Orbital geometry, Kanamori params (U, U′, J_H), crystal field, electron sector (nelec, Hilbert dim), solver config |
| `q8020_code_*.json` | Algorithm, entry point, full run_args, library versions |
| `q8020_backend_*.json` | Backend type, coupling map, noise model, basis gates |
| `q8020_exec_stats_*.json` | Per-step timing, circuit depths/gate counts, shots per circuit, total samples, unique bitstrings |
| `q8020_results_*.json` | SQD energy, FCI exact energy, error (eV, meV, %), convergence flag, full energy history |

## Dependencies

- **Qiskit** ≥2.0 (circuits, transpilation)
- **Qiskit Aer** (statevector/shot-based simulation)
- **qiskit-addon-sqd** (sample-based quantum diagonalization)
- **PySCF** (exact FCI reference energies)
- **ffsim** (fermionic simulation utilities)
- **NumPy**, **SciPy**
- **q8020-cfd-metautil** (sweep orchestration, metadata capture)
- **q8020-cfd-qutil** (backend rigging: sim/fake/hardware modes, coupling maps, noise models)
