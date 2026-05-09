# SKQD-SIAM: Sample-based Krylov Quantum Diagonalization for the Single Impurity Anderson Model

This project implements a quantum-classical hybrid algorithm for computing ground state energies of the Single Impurity Anderson Model (SIAM) using Sample-based Krylov Quantum Diagonalization (SKQD).

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

### Application to SIAM

The **Single Impurity Anderson Model** describes a magnetic impurity coupled to a bath of conduction electrons. It is a fundamental model in condensed matter physics for understanding:
- Kondo physics and heavy fermion systems
- Quantum dots coupled to leads
- Magnetic impurities in metals

The Hamiltonian includes:
- **Hopping** (t): Electron hopping between bath sites
- **Onsite interaction** (U): Coulomb repulsion on the impurity
- **Hybridization** (V): Coupling between impurity and bath
- **Chemical potential** (μ): Controls filling, computed as `filling_factor × U`

This implementation uses a momentum-space representation for circuit construction and a site-basis representation for classical post-processing.

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
cd qt04-skqd
uv venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Run a parameter study

```bash
python run-study-skqd.py input/study-example.toml
```

This will run the noise sweep defined in the example TOML and produce a summary table with ground state energies and errors compared to exact FCI.

### 5. Run a single case

```bash
python run-skqd.py --num-orbs 8 --krylov-dim 5 --shots 1024
```

See `python run-skqd.py --help` or the TOML reference below for all available options.

## Workflow

The algorithm proceeds in five steps:

1. **Step 1** (`step1_siam.py`): Build the SIAM Hamiltonian in momentum basis
2. **Step 2** (`step2_krylov.py`): Construct Krylov circuits using Trotterized time evolution
3. **Step 3** (`step3_transpile.py`): Transpile circuits for the target backend
4. **Step 4** (`step4_execute.py`): Execute circuits and collect bitstring samples
5. **Step 5** (`step5_postprocess.py`): Classical SQD diagonalization to extract ground state energy

## TOML Configuration Reference

### System Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_orbs` | int | 10 | Number of spatial orbitals (qubits = 2 × num_orbs) |

### Krylov Circuit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `krylov_dim` | int | 5 | Number of Krylov basis states to generate |
| `dt_mult` | float | 1.0 | Time step multiplier for Trotter evolution |

### Execution Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shots` | int | 1024 | Number of measurement shots per circuit |
| `noise` | float | 0.0 | Depolarizing noise rate on 2-qubit gates (0 = noiseless) |
| `opt_level` | int | 1 | Qiskit transpiler optimization level (0-3) |

### Hamiltonian Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hopping` | float | 1.0 | Hopping parameter (t) |
| `onsite` | float | 5.0 | Onsite Coulomb interaction (U) |
| `hybridization` | float | 1.0 | Impurity-bath hybridization strength (V) |
| `filling_factor` | float | -0.5 | Chemical potential = filling_factor × U |

### SQD Post-processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | int | 10 | Maximum SQD self-consistent iterations |
| `num_batches` | int | 5 | Number of batches for subspace construction |
| `samples_per_batch` | int | 200 | Samples drawn per batch |

## Example TOML

```toml
[global]
# System
num_orbs = 8

# Krylov
krylov_dim = 5
dt_mult = 1.0

# Execution
shots = 1024
opt_level = 1

# Hamiltonian
hopping = 1.0
onsite = 5.0
hybridization = 1.0
filling_factor = -0.5

# SQD
max_iter = 10
num_batches = 5
samples_per_batch = 200

[noise_sweep]
noise = [0.0, 0.005, 0.01]
```

Cases can override global parameters. List values expand into multiple subcases automatically.

## Output

The study runner produces a summary table showing all parameters and results:

```
STUDY SUMMARY
============================================================
Case                 Orbs Krylov    dt  Noise  Shots Opt Iter Bat Samp   hop     U   hyb   fill       Energy        Exact     Err%
------------------------------------------------------------------------------------------------------------------------
noise_sweep_0           8      5   1.0  0.000   1024   1   10   5  200   1.0   5.0   1.0  -0.50     -12.3456     -12.3456   0.0001
...
```

## Dependencies

- Qiskit
- Qiskit Aer
- qiskit-addon-sqd
- PySCF (for exact FCI reference)
- NumPy

## Future Work

- **Multiple backends**: Support for IBM Quantum hardware backends and other simulators
- **Expanded noise models**: Backend-specific noise models extracted from real device calibration data
- **Case parallelization**: Run multiple study cases in parallel across available cores
- **Post-processing parallelization**: Parallelize SQD batch processing within each case
- **Visualization**: Selective plotting of case parameters vs results (e.g., noise vs error, shots vs accuracy); CSV output of results
- **Uncertainty quantification**: Statistical analysis and visualization from multiple runs with error bars and confidence intervals
