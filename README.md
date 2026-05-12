# SKQD-AIM: Sample-based Krylov Quantum Diagonalization for Anderson Impurity Models

Quantum-classical hybrid solver for ground-state energies of the multi-orbital Anderson Impurity Model with rotationally-invariant Kanamori interactions. The quantum side prepares Krylov-evolved states and samples bitstrings; the classical side diagonalizes the Hamiltonian projected onto the sampled subspace (SCI). Implemented on top of [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) and [ffsim](https://github.com/qiskit-community/ffsim), with [PySCF](https://pyscf.org) used for FCI reference energies.

> *Nomenclature*: the upstream technique is **SQD** (Sample-based Quantum Diagonalization, Robledo-Moreno *et al.*). This repo's **SKQD** label specifically refers to the *Krylov-circuit* state-preparation variant; the classical SCI side is unchanged and is provided by [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd).

---

## Physics scope

### What this code computes

For a cluster of `M` correlated impurity orbitals and `M·B` bath orbitals in star geometry — each bath site `(m, b)` couples to impurity orbital `m` only, with no bath–bath, no impurity–impurity, and no cross-impurity-to-bath one-body hopping; impurity orbitals interact only through the Kanamori two-body terms — the code builds the second-quantized Hamiltonian

```
H = H_1 + H_imp_int

h1e[p, p] =  ε_CF[m] − μ   if p is impurity orbital m   (crystal field minus chem. pot.)
          =  ε_bath[m, b]  if p is bath site (m, b)     (per-bath on-site energy)
h1e[m, q] = −V_{m, b}      for q the bath site (m, b)   (impurity–bath hybridization)

H_1   = Σ_{pq, σ}  h1e[p, q]  c†_{pσ} c_{qσ}            (one-body part)
H_int = U   Σ_m   n_{m↑} n_{m↓}                          (intra-orbital Coulomb)
      + U′  Σ_{m<m'}  Σ_{σσ'}  n_{mσ} n_{m'σ'}           (inter-orbital, σ ≠ σ')
      + (U′ − J_H) Σ_{m<m', σ}  n_{mσ} n_{m'σ}           (inter-orbital, σ = σ')
      − J_H Σ_{m≠m'} (c†_{m↑} c_{m↓} c†_{m'↓} c_{m'↑}      spin-flip
                    + c†_{m↑} c†_{m↓} c_{m'↓} c_{m'↑})    pair-hopping
```
The two-body sums run over impurity orbitals only (m, m′ ∈ {0, …, M−1}); bath sites are non-interacting.

This is the rotationally-invariant Kanamori parameterization — the form typically used for t₂g / e_g subspaces of d-shells under cubic symmetry. It does not reproduce full Slater–Condon multiplet splittings for f-shells; see *Limitations* below.

The two-electron tensor is assembled with explicit 8-fold permutation symmetrization in the convention used by both ffsim and PySCF, then handed to ffsim's `MolecularHamiltonian` (Jordan–Wigner mapping) for circuit construction and to PySCF FCI for the reference energy. Numerical agreement of the two on small cases is one of the correctness checks.

### Inputs

| Quantity | Symbol | CLI | Notes |
|---|---|---|---|
| Impurity orbital count | M | `--num-imp-orbs` | Active correlated orbitals; the orbital basis (spherical, cubic, real-CF, …) is an external choice and is not constrained by the code |
| Bath sites per impurity | B | `--num-bath-per-imp` | Star geometry; bath sites of impurity m are independent |
| Intra-orbital Coulomb | U | `--onsite` (eV) | |
| Inter-orbital Coulomb | U′ | `--U-prime` (eV) | Defaults to `U − 2·J_H` (rotationally-invariant relation) |
| Hund's exchange | J_H | `--J-H` (eV) | |
| Chemical potential | μ | `--mu` (eV) | Subtracted from every diagonal h1e entry |
| Crystal-field eigenvalues | ε_CF | `--crystal-field` (eV, comma-sep, length M) | Diagonal one-body splitting in the chosen orbital basis |
| Bath energies | ε_p | `--bath-energies` (eV, comma-sep, length M·B) | Per-bath-site, row-major over (m, b) |
| Hybridization | V_p | `--bath-couplings` (eV, comma-sep, length M·B) | Per-bath-site. `--hybridization` is a scalar shortcut that broadcasts |
| Particle sector | (n↑, n↓) | `--n-electrons-alpha`, `--n-electrons-beta` | Hard-sets the (Nα, Nβ) sector of the FCI / SCI solve. Defaults to half-filling |

All energies in eV. Parameter values are inputs to the code; no defaults are pulled from a database. Typical sources include DFT+U, constrained-RPA, atomic Hartree–Fock, or experimental fits.

### Limitations

1. **Kanamori, not Slater–Condon.** The two-electron tensor uses three radial parameters `(U, U′, J_H)`, not the full `(F⁰, F², F⁴, F⁶)` Slater integrals. Adequate for d-shells in cubic environments and for f¹ / f¹³ systems (no two-electron multiplet structure). Loses accuracy on excited multiplet spacings for f² through f¹². Roadmapped in [#8](https://github.com/agallojr/qmat-aim-skqd/issues/8).
2. **No spin-orbit coupling.** Matters for L-S vs j-j coupling in actinides and for the Sm 4f⁵, Eu 4f⁶ ⁷F, and similar fine-structure manifolds. Not in scope; no issue filed yet.
3. **No Δ(ω) → {V_k, ε_k} fitting.** This is a standalone impurity solver, not a DMFT loop. Discretization of the hybridization function happens externally; the resulting bath parameters are passed via `--bath-energies` / `--bath-couplings`.
4. **Crystal field is diagonal.** Off-diagonal CF rotations (lower-than-cubic point groups, trigonal/tetragonal mixing of basis states) are not represented. Workaround: pre-diagonalize the CF block externally and pass the eigenvalues.
5. **Ground state only.** Excited-state energies require `nroots > 1` in the SCI solve — roadmapped in [#7](https://github.com/agallojr/qmat-aim-skqd/issues/7).
6. **Bath geometry is star-only.** No chain, no bath-bath hopping. Adequate for Anderson-style impurity problems; not adequate for, e.g., the bath of a CT-QMC reference solver that uses a different discretization.

### Validation status

| Test | Status |
|---|---|
| Single-orbital SIAM at half-filling vs `qc-dft-dmft` reference (`small-sys.toml`) | matches |
| Multi-orbital Hubbard at half-filling, FCI to 7 sig figs (M=5, B=1, ce-4f geometry, half-filled) | passes |
| Hand-built Kanamori H vs `multi_orbital_aim_hamiltonian` on M=2 (machine-precision eigenvalue match) | passes |
| 8-fold ERI symmetrization (numerical check on M=3) | passes |
| Per-bath enrichment regression (B=1 effective vs B=2 with distinct ε,V) | covered by `bath-richness-test.toml` |

---

## Algorithm

### Sample-based Krylov Quantum Diagonalization (SKQD)

1. Prepare an initial reference state |ψ₀⟩ in the (Nα, Nβ) sector — a single Slater determinant with the lowest Nα α-orbitals and lowest Nβ β-orbitals occupied (no HF iteration, no orbital optimization).
2. Generate Krylov circuits |ψₖ⟩ ≈ T(dt)ᵏ |ψ₀⟩ where `T(dt)` is a Trotterized real-time evolution under H. Available product formulas: 1st-order Lie, 2nd-order Strang (default), 4th-order Suzuki S4. Sub-stepping is also available (`--trotter-substeps`).
3. Sample computational-basis bitstrings from each |ψₖ⟩ on a simulator or on hardware.
4. Project H onto the support of all sampled bitstrings, and diagonalize classically via [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) (which calls PySCF FCI internally).

The classical SCI step performs the determinant selection; the quantum side only needs the ground-state determinants to be present in the sampled bitstring distribution. Because SKQD requires no expectation-value estimation, it can be more shot-efficient than VQE on problems where the ground state has sparse determinant support.

### Trotter accuracy and the `dt` knob

A pre-flight diagnostic (`--check-trotter`, restricted to ≤ 6 spatial orbitals) computes ‖T(dt) − e^{−iHdt}‖₂ on the half-filled sector and reports the rigorous fidelity bound `1 − ε²`. Two `dt`-scale modes:

- `--dt-scale-mode h1e` (default) — `dt = π · dt_mult / ‖h1e‖₂`. Places the Krylov states in the large-dt regime that produces good SCI subspace coverage in practice.
- `--dt-scale-mode full` — `dt = π · dt_mult / (‖h1e‖₂ + max|h2e|)`. Trotter-accurate but produces tightly clustered Krylov states; pair with `--trotter-substeps` and a larger `--dt-mult` for SQD use.

The two modes optimize different objectives: SCI subspace coverage benefits from spread-out Krylov states, while product-formula accuracy benefits from states close to the exact evolution. The default favors coverage.

### Five-step pipeline

| Step | Source | Output |
|---|---|---|
| 1. Build AIM Hamiltonian | `src/step1_siam.py` | `h1e.npy`, `h2e.npy` |
| 2. Build Krylov circuits (Trotter) | `src/step2_krylov.py` | `circuits.qpy`, `circuit_metadata.json` |
| 3. Transpile | `src/step3_transpile.py` | `transpiled_circuits.qpy`, `transpile_stats.json` |
| 4. Execute (sim / fake / hardware) | `src/step4_execute.py` | `counts.json`, `bitstrings.npy`, `probabilities.npy` |
| 5. Classical SCI diagonalization | `src/step5_postprocess.py` | `energy_history.json`, result JSON |

See [doc/pipeline.md](doc/pipeline.md) for a per-step description of what each module actually computes.

---

## Getting Started

### Install

```bash
# Recommended (uv)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
cd qmat-aim-skqd
uv venv && source .venv/bin/activate
uv sync

# Alternative (pip)
cd qmat-aim-skqd
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Run a single case

```bash
# Single-orbital Anderson, half-filling (default): 1 imp + 3 bath = 4 orbs = 8 qubits
python src/run-skqd.py --num-imp-orbs 1 --num-bath-per-imp 3 --shots 4096

# Ce 4f^1 — atomic filling (1 electron, spin-up)
python src/run-skqd.py --num-imp-orbs 5 --num-bath-per-imp 1 \
    --n-electrons-alpha 1 --n-electrons-beta 0 \
    --onsite 6.0 --U-prime 4.6 --J-H 0.7 --mu 3.0 --hybridization 0.4 \
    --crystal-field "-0.15,-0.05,0.0,0.05,0.15" \
    --shots 65536

# Pr 4f^2 — Sz = 0 component of the Hund's-rule S=1 triplet
# (degenerate by SU(2) with the fully spin-polarized (2, 0) sector)
python src/run-skqd.py --num-imp-orbs 5 --num-bath-per-imp 1 \
    --n-electrons-alpha 1 --n-electrons-beta 1 \
    --onsite 6.5 --U-prime 5.0 --J-H 0.75 --mu 3.0 --hybridization 0.3 \
    --crystal-field "-0.15,-0.05,0.0,0.05,0.15" \
    --shots 65536
```

### Run a parameter sweep (q8020-sweep)

```bash
q8020-sweep input/<config>.toml
q8020-sweep input/<config>.toml --dry-run
```

The sweeper expands list-valued TOML parameters into a cross-product of cases, calls `run-skqd.py` for each, and collects results. Example TOML configurations live under [`input/`](input/).

### CLI reference

The single source of truth for every flag (defaults, units, semantics, and the `q8020-sweep` integration notes) lives in the `run-skqd.py` module docstring. To view it:

```bash
python src/run-skqd.py --help
```

This README intentionally keeps the CLI table out — flag documentation should not drift between two files.

---

## Output

Each case emits one JSON line on stdout:

```
SKQD_RESULT_JSON:{"sqd_energy": -2.876, "exact_energy": -2.876, "error_pct": 1.5e-5, ...}
```

When `--output-dir` is supplied, the solver writes per-case artifacts and the [q8020](https://github.com/Q8020-CFD/q8020-cfd-metautil) metadata fragments below.

| Fragment | Contents |
|---|---|
| `q8020_case_*.json` | Orbital geometry; Kanamori (U, U′, J_H, μ, V, ε_p, ε_CF); electron sector (n↑, n↓, sector dim); solver config |
| `q8020_code_*.json` | Algorithm name, entry point, full run_args, library versions |
| `q8020_backend_*.json` | Backend type, coupling map, noise model, basis gates |
| `q8020_exec_stats_*.json` | Per-step timing, depths/gate counts, shots, total samples, unique bitstrings |
| `q8020_results_*.json` | SQD energy, FCI exact, error (eV / meV / %), convergence, full energy history, Trotter pre-flight distance |

---

## Dependencies

- **Qiskit** ≥ 2.0 (circuits, transpilation)
- **Qiskit Aer** (statevector / shot-based simulation)
- **qiskit-addon-sqd** (sample-based selected-CI)
- **PySCF** (FCI reference)
- **ffsim** (fermionic operators, Jordan–Wigner mapping, orbital rotations)
- **NumPy**, **SciPy**
- **q8020-cfd-metautil** (sweep orchestration, metadata)
- **q8020-cfd-qutil** (backend configuration)

---

## References

- *Sample-based Quantum Diagonalization* — [Robledo-Moreno et al., arXiv:2405.05068](https://arxiv.org/abs/2405.05068)
- *Krylov Variational Quantum Algorithm for First-Principles Materials Simulations* — [Baker et al., arXiv:2105.13298](https://arxiv.org/abs/2105.13298)
- *Hybrid Quantum-Classical Approach to Correlated Materials* — [Bauer et al., Phys. Rev. X 6, 031045 (2016)](https://link.aps.org/doi/10.1103/PhysRevX.6.031045)
- Companion / reference repo: [qc-dft-dmft](https://code.ornl.gov/se-qcwg/qc-dft-dmft) (single-orbital DMFT impurity solver with VQE / variational Lanczos)
