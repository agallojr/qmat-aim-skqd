# SKQD-AIM: Sample-based Krylov Quantum Diagonalization for Anderson Impurity Models

Quantum-classical hybrid solver for ground-state energies of the multi-orbital Anderson Impurity Model with rotationally-invariant Kanamori interactions. The quantum side prepares Krylov-evolved states and samples bitstrings; the classical side diagonalizes the Hamiltonian projected onto the sampled subspace (SCI). Implemented on top of [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) and [ffsim](https://github.com/qiskit-community/ffsim), with [PySCF](https://pyscf.org) used for FCI reference energies.

---

## Physics scope

### What this code computes

For a cluster of `M` correlated impurity orbitals and `M·B` bath orbitals (star geometry — every bath site couples directly to its impurity orbital, no bath-bath hopping), the code builds the second-quantized Hamiltonian

```
H = H_1 + H_imp_int

H_1   = Σ_{pq, σ}  h1e[p, q]  c†_{pσ} c_{qσ}        (one-body: ε_imp + crystal field, μ, V_{m,b})
H_int = U   Σ_m   n_{m↑} n_{m↓}                      (intra-orbital Coulomb)
      + U′  Σ_{m<m'}  Σ_{σσ'}  n_{mσ} n_{m'σ'}       (inter-orbital, σ ≠ σ')
      + (U′ − J_H) Σ_{m<m', σ}  n_{mσ} n_{m'σ}       (inter-orbital, σ = σ')
      − J_H Σ_{m≠m'} (c†_{m↑} c_{m↓} c†_{m'↓} c_{m'↑}   spin-flip
                    + c†_{m↑} c†_{m↓} c_{m'↓} c_{m'↑})  pair-hopping
```

This is the standard rotationally-invariant Kanamori parameterization. It is exact for two-orbital problems and is the standard cubic-symmetry approximation for t₂g / e_g subspaces of d-shells. For a full f-shell it captures Hund's-rule ordering correctly (validated on Pr 4f² ³H vs ¹G sectors) but does not reproduce Slater–Condon multiplet splittings — see *Limitations* below.

The two-electron tensor is assembled in chemist-ERI convention with explicit 8-fold permutation symmetrization, then handed to ffsim's `MolecularHamiltonian` (Jordan–Wigner mapping) for circuit construction and to PySCF FCI for the reference energy. **Both consumers see the same h2e.** Numerical agreement of ffsim and PySCF FCI on small cases is the primary correctness check.

### What you control (chemist-facing)

| Quantity | Symbol | CLI | Notes |
|---|---|---|---|
| Impurity orbital count | M | `--num-imp-orbs` | Active correlated orbitals; user chooses the basis (spherical, cubic, real-CF, etc.) |
| Bath sites per impurity | B | `--num-bath-per-imp` | Star geometry; bath sites of impurity m are independent |
| Intra-orbital Coulomb | U | `--onsite` (eV) | |
| Inter-orbital Coulomb | U′ | `--U-prime` (eV) | Defaults to `U − 2·J_H` (rotationally-invariant relation) |
| Hund's exchange | J_H | `--J-H` (eV) | |
| Chemical potential | μ | `--mu` (eV) | Subtracted from every diagonal h1e entry |
| Crystal-field eigenvalues | ε_CF | `--crystal-field` (eV, comma-sep, length M) | Diagonal one-body splitting in the chosen orbital basis |
| Bath energies | ε_p | `--bath-energies` (eV, comma-sep, length M·B) | Per-bath-site, row-major over (m, b) |
| Hybridization | V_p | `--bath-couplings` (eV, comma-sep, length M·B) | Per-bath-site. `--hybridization` is a scalar shortcut that broadcasts |
| Particle sector | (n↑, n↓) | `--n-electrons-alpha`, `--n-electrons-beta` | Hard-sets the (Nα, Nβ) sector of the FCI / SCI solve. Defaults to half-filling |

All energies in eV. The user owns the orbital basis and the parameter values — none are pulled from a database. Source them from DFT+U, constrained-RPA, atomic Hartree–Fock, or experimental fits.

### Limitations a chemist should know up front

1. **Kanamori, not Slater–Condon.** The two-electron tensor uses three radial parameters `(U, U′, J_H)`, not the full `(F⁰, F², F⁴, F⁶)` Slater integrals. Adequate for d-shells in cubic environments and for f¹ / f¹³ (one carrier — no two-electron multiplet structure to get wrong). Loses accuracy on excited multiplet spacings for f² through f¹². Roadmapped in [#8](https://github.com/agallojr/qmat-aim-skqd/issues/8).
2. **No spin-orbit coupling.** Matters for L-S vs j-j coupling in actinides and for the Sm 4f⁵, Eu 4f⁶ ⁷F, and similar fine-structure manifolds. Not in scope; no issue filed yet.
3. **No Δ(ω) → {V_k, ε_k} fitting.** This is a standalone impurity solver, not a DMFT loop. The user discretizes the hybridization function externally and passes the bath parameters via `--bath-energies` / `--bath-couplings`.
4. **Crystal field is diagonal.** Off-diagonal CF rotations (lower-than-cubic point groups, trigonal/tetragonal mixing of basis states) are not represented. Workaround: pre-diagonalize the CF block externally and pass the eigenvalues.
5. **Ground state only.** Excited-state energies require `nroots > 1` in the SCI solve — roadmapped in [#7](https://github.com/agallojr/qmat-aim-skqd/issues/7).
6. **Bath geometry is star-only.** No chain, no bath-bath hopping. Adequate for Anderson-style impurity problems; not adequate for, e.g., the bath of a CT-QMC reference solver that uses a different discretization.

### Validation status

| Test | Status |
|---|---|
| Single-orbital SIAM at half-filling vs `qc-dft-dmft` reference (`small-sys.toml`) | matches |
| Multi-orbital Hubbard at half-filling, FCI to 7 sig figs (M=5, B=1, ce-4f geometry, half-filled) | passes (post-Hund Trotter fix) |
| Hand-built Kanamori H vs `multi_orbital_aim_hamiltonian` on M=2 (machine-precision eigenvalue match) | passes |
| 8-fold ERI symmetrization (numerical check on M=3) | passes |
| Hund's-rule ordering S=1 < S=0 on Pr 4f² | covered by `pr-4f2.toml` `[singlet_check]` group |
| Per-bath enrichment regression (B=1 effective vs B=2 with distinct ε,V) | covered by `bath-richness-test.toml` |

---

## Algorithm

### Sample-based Krylov Quantum Diagonalization (SKQD)

1. Prepare an initial reference state |ψ₀⟩ in the (Nα, Nβ) sector — currently a Hartree-Fock-like occupation-number state.
2. Generate Krylov circuits |ψₖ⟩ ≈ T(dt)ᵏ |ψ₀⟩ where `T(dt)` is a Trotterized real-time evolution under H. Available product formulas: 1st-order Lie, 2nd-order Strang (default), 4th-order Suzuki S4. Sub-stepping is also available (`--trotter-substeps`).
3. Sample computational-basis bitstrings from each |ψₖ⟩ on a simulator or on hardware.
4. Project H onto the support of all sampled bitstrings, and diagonalize classically via [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) (which calls PySCF FCI under the hood).

The classical SCI step *selects the determinants* — the quantum side only needs to make sure the ground-state determinants are *visible* in the bitstring distribution. SKQD does not require expectation-value estimation, so it is more shot-efficient than VQE for problems where the ground state has sparse determinant support.

### Trotter accuracy and the `dt` knob

A pre-flight diagnostic (`--check-trotter`, restricted to ≤ 6 spatial orbitals) computes ‖T(dt) − e^{−iHdt}‖₂ on the half-filled sector and reports the rigorous fidelity bound `1 − ε²`. Two `dt`-scale modes:

- `--dt-scale-mode h1e` (default) — `dt = π · dt_mult / ‖h1e‖₂`. Empirically-good large-dt regime for SCI subspace coverage.
- `--dt-scale-mode full` — `dt = π · dt_mult / (‖h1e‖₂ + max|h2e|)`. Trotter-accurate but produces tightly clustered Krylov states; pair with `--trotter-substeps` and a larger `--dt-mult` for SQD use.

These are different objectives. SCI subspace coverage wants the Krylov states *spread*; product-formula accuracy wants them *close to exact*. The default favors coverage.

### Five-step pipeline

| Step | Source | Output |
|---|---|---|
| 1. Build AIM Hamiltonian | `src/step1_siam.py` | `h1e.npy`, `h2e.npy`, `case_info.json` |
| 2. Build Krylov circuits (Trotter) | `src/step2_krylov.py` | `circuits.qpy`, `circuit_metadata.json` |
| 3. Transpile | `src/step3_transpile.py` | `transpiled_circuits.qpy`, `transpile_stats.json` |
| 4. Execute (sim / fake / hardware) | `src/step4_execute.py` | `counts.json`, `bitstrings.npy`, `probabilities.npy` |
| 5. Classical SCI diagonalization | `src/step5_postprocess.py` | `energy_history.json`, result JSON |

Each step is independently runnable from a case directory.

---

## Getting Started

### Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
cd qmat-aim-skqd
uv venv && source .venv/bin/activate
uv sync
```

### Run a single case

```bash
# Single-orbital Anderson, half-filling (default): 1 imp + 3 bath = 4 orbs = 8 qubits
python src/run-skqd.py --num-imp-orbs 1 --num-bath-per-imp 3 --shots 4096

# Ce 4f^1 — proper atomic filling (1 electron, spin-up)
python src/run-skqd.py --num-imp-orbs 5 --num-bath-per-imp 1 \
    --n-electrons-alpha 1 --n-electrons-beta 0 \
    --onsite 6.0 --U-prime 4.6 --J-H 0.7 --mu 3.0 --hybridization 0.4 \
    --crystal-field "-0.15,-0.05,0.0,0.05,0.15" \
    --shots 65536

# Pr 4f^2 — Hund's-rule high-spin sector
python src/run-skqd.py --num-imp-orbs 5 --num-bath-per-imp 1 \
    --n-electrons-alpha 2 --n-electrons-beta 0 \
    --onsite 6.5 --U-prime 5.0 --J-H 0.75 --mu 3.0 --hybridization 0.3 \
    --crystal-field "-0.15,-0.05,0.0,0.05,0.15" \
    --shots 65536
```

### Run a parameter sweep (q8020-sweep)

```bash
q8020-sweep input/smoke-test.toml             # 8q SIAM sanity check
q8020-sweep input/small-sys.toml              # 12q SIAM, comparable to qc-dft-dmft reference
q8020-sweep input/bath-richness-test.toml     # 12q regression for per-bath ε/V plumbing
q8020-sweep input/ce-4f1.toml                 # 20q Ce 4f^1
q8020-sweep input/pr-4f2.toml                 # 20q Pr 4f^2 (Hund's-rule check)
q8020-sweep input/smoke-test.toml --dry-run
```

The sweeper expands list-valued TOML parameters into a cross-product of cases, calls `run-skqd.py` for each, and collects results.

### Available configurations

| File | System | Filling | Qubits | Sector dim | Purpose |
|---|---|---|---|---|---|
| [`smoke-test.toml`](input/smoke-test.toml) | SIAM (Co 3d / Cu, M=1, B=3) | half-filled | 8 | 36 | Pipeline regression |
| [`small-sys.toml`](input/small-sys.toml) | SIAM (M=1, B=5) | half-filled | 12 | 400 | Cross-check vs `qc-dft-dmft` |
| [`bath-richness-test.toml`](input/bath-richness-test.toml) | M=2, B=2 with three bath configs | half-filled | 12 | 225 | Per-bath ε/V plumbing regression |
| [`ce-4f1.toml`](input/ce-4f1.toml) | Ce 4f¹ (M=5, B=1) | (n↑, n↓) = (1, 0) | 20 | 10 | Atomic Ce |
| [`pr-4f2.toml`](input/pr-4f2.toml) | Pr 4f² (M=5, B=1) | (n↑, n↓) = (2, 0) high-spin; singlet check too | 20 | 45 | Hund's-rule ordering test |

Sector dim is the FCI dimension `C(N_orb, n↑) · C(N_orb, n↓)` — qubit count alone is misleading for non-half-filling.

---

## CLI Reference

### Hamiltonian

| Flag | Type | Default | Description |
|---|---|---|---|
| `--num-imp-orbs` | int | 1 | Impurity orbitals M |
| `--num-bath-per-imp` | int | 1 | Bath sites per impurity B |
| `--onsite` | float | 4.0 | Intra-orbital Coulomb U (eV) |
| `--U-prime` | float | U − 2·J_H | Inter-orbital Coulomb U′ (eV) |
| `--J-H` | float | 0.0 | Hund's exchange J_H (eV) |
| `--mu` | float | 2.0 | Chemical potential μ (eV) |
| `--hybridization` | float | 0.8 | Scalar V (broadcast across all M·B baths) |
| `--bath-energies` | str | zeros | Per-bath ε, comma-sep length M·B (eV), row-major (m, b) |
| `--bath-couplings` | str | broadcast `--hybridization` | Per-bath V, comma-sep length M·B (eV) |
| `--crystal-field` | str | none | Diagonal CF eigenvalues, comma-sep length M (eV) |
| `--n-electrons` | int | N_orb (half-filling) | Total electrons; split evenly between spins |
| `--n-electrons-alpha` | int | n_electrons // 2 | α (spin-↑) electrons |
| `--n-electrons-beta` | int | n_electrons − n_α | β (spin-↓) electrons |
| `--system-label` | str | none | Free-form label for output |

### Krylov / Trotter

| Flag | Type | Default | Description |
|---|---|---|---|
| `--krylov-dim` | int | 5 | Number of Krylov basis states |
| `--dt-mult` | float | 1.0 | Time-step multiplier |
| `--dt-scale-mode` | str | `h1e` | `h1e` (legacy) or `full` (omega_h1e + max\|h2e\|) |
| `--trotter-order` | int | 2 | 1 (Lie), 2 (Strang), 4 (Suzuki S4) |
| `--trotter-substeps` | int | 1 | Sub-divisions per Krylov increment |
| `--check-trotter` | flag | off | Pre-flight: ‖T(dt) − e^{−iHdt}‖₂ on N≤6 sector |

### Execution / backend

| Flag | Type | Default | Description |
|---|---|---|---|
| `--shots` | int | 1024 | Measurement shots per circuit |
| `--opt-level` | int | 1 | Qiskit transpiler optimization level |
| `--backend` | str | none | Fake backend name (e.g. `manila`, `brisbane`) |
| `--backend-type` | str | `sim` | `sim` (AerSimulator), `fake` (FakeBackendV2), `hardware` (IBM Quantum) |
| `--coupling-map` | str | `default` | `default` (backend native) or `all-to-all` |
| `--t1`, `--t2` | float | none | Override T1/T2 (µs) on top of backend noise model |

### SQD post-processing

| Flag | Type | Default | Description |
|---|---|---|---|
| `--max-iter` | int | 10 | Max SQD self-consistent iterations |
| `--num-batches` | int | 5 | Batches for subspace construction |
| `--samples-per-batch` | int | 200 | Samples per batch |

### Output

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output-dir` | path | none | Directory for case output files |
| `--no-persist` | flag | off | Skip writing intermediates |

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
- **q8020-cfd-qutil** (backend rigging)

---

## References

- *Sample-based Quantum Diagonalization* — [Robledo-Moreno et al., arXiv:2405.05068](https://arxiv.org/abs/2405.05068)
- *Krylov Variational Quantum Algorithm for First-Principles Materials Simulations* — [Baker et al., arXiv:2105.13298](https://arxiv.org/abs/2105.13298)
- *Hybrid Quantum-Classical Approach to Correlated Materials* — [Bauer et al., Phys. Rev. X 6, 031045 (2016)](https://link.aps.org/doi/10.1103/PhysRevX.6.031045)
- Companion / reference repo: [qc-dft-dmft](https://code.ornl.gov/se-qcwg/qc-dft-dmft) (single-orbital DMFT impurity solver with VQE / variational Lanczos)
