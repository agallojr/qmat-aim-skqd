# SKQD pipeline — what each step does

The SKQD solver is a five-step quantum-classical hybrid for ground-state
energies of the multi-orbital Anderson Impurity Model. Steps 1, 2, 3, 5 are
classical; step 4 is the only one that touches a (real or simulated) quantum
backend. The orchestrator is [src/run-skqd.py](../src/run-skqd.py); each
step lives in its own module.

## Step 1 — Build the AIM Hamiltonian
[src/step1_siam.py](../src/step1_siam.py)

Given the user's choice of M correlated impurity orbitals and B bath sites
per orbital, this assembles the second-quantised AIM in the standard
chemistry-style `(h1e, h2e)` representation: `h1e` is the one-body matrix on
`num_orbs = M*(1+B)` spatial orbitals (impurity on-site energies, bath
on-site energies, hybridisation V coupling each impurity orbital to its
star-geometry bath sites, optional crystal-field splitting, and a uniform
`-mu` chemical potential on the diagonal); `h2e` is the four-index two-body
tensor carrying the rotationally-invariant Kanamori interaction (intra-orbital
U, inter-orbital U′, Hund's J_H — including the spin-flip and pair-hopping
pieces that make it more than a density–density approximation). All of this
is purely classical — numpy arrays — and feeds every later step that needs a
Hamiltonian (Trotter circuits, FCI exact reference, classical SQD
diagonaliser).

## Step 2 — Build Krylov-evolved quantum circuits
[src/step2_krylov.py](../src/step2_krylov.py)

This produces a list of `krylov_dim` circuits where circuit `k` prepares a
Hartree-Fock-like reference state on `2*num_orbs` qubits (one per
spin-orbital, populating `n_alpha + n_beta` of them with X gates) and then
applies `k` repetitions of a Trotterised real-time evolution
`T(dt) ≈ exp(-i H dt)` — Suzuki order 1/2/4, optionally subdivided into
`trotter_substeps` sub-steps. The time-step `dt = π * dt_mult / scale` is
chosen by `_resolve_dt` from either `||h1e||₂` (the legacy Krylov-SQD
heuristic that spreads samples broadly across Fock space) or
`||h1e||₂ + max|h2e|` (a Trotter-accurate but tightly-clustered regime). An
optional pre-flight diagnostic computes `||T(dt) - exp(-iHdt)||₂` exactly on
small systems to bound per-step fidelity. Every circuit ends with
`measure_all` so we can sample bitstrings.

## Step 3 — Transpile circuits to the target backend
[src/step3_transpile.py](../src/step3_transpile.py)

`transpile_circuits` runs Qiskit's preset pass-manager at the requested
optimisation level (0–3), which decomposes the high-level gates in the
Krylov circuits into the backend's native basis gates and routes them onto
the backend's coupling map (or full connectivity if no fake/hardware backend
was selected). The output is a parallel list of physically realisable
circuits whose depth and 2-qubit-gate count are recorded as the chief
noise-cost proxies; the orchestrator persists per-circuit
`{depth, num_qubits, gate_counts}` plus the 2q-gate aggregate so the
noise-vs-accuracy tradeoff can be inspected per case after a sweep.

## Step 4 — Execute the transpiled circuits and collect bitstring counts
[src/step4_execute.py](../src/step4_execute.py)

This is the only step that touches a real or simulated quantum backend. For
`shots > 0` it loops over the circuits and delegates each to
`q8020_cfd_qutil.execute_circuit_counts`, which dispatches to `backend.run`
for AerSimulator and to `SamplerV2` for real IBM hardware (so the same code
path covers noiseless sim, fake-backend noise models, and real devices) and
returns normalised single-string counts plus per-circuit timing/job-id
telemetry. Counts from all Krylov circuits are merged into one combined
dictionary and converted via `qiskit_addon_sqd.counts_to_arrays` into a
`(bitstrings, probabilities)` pair. The `shots == 0` branch instead computes
each circuit's exact `Statevector` and draws 10 000 samples from the ideal
probability distribution — useful for algorithmic debugging, but it bypasses
the backend entirely and warns loudly if a noise-configured backend was
about to be ignored.

## Step 5 — Classical SQD post-processing to recover the ground-state energy
[src/step5_postprocess.py](../src/step5_postprocess.py)

The bitstring counts from step 4 define a "sampled" subspace of the full
Fock space; this step packs them into a `BitArray`, hands them to
`qiskit_addon_sqd`'s self-consistent diagonaliser
(`classically_diagonalize`), which iterates: (i) split the samples into
`num_batches` batches of `samples_per_batch` each, (ii) project the original
`(h1e, h2e)` Hamiltonian onto each batch's particle-number-resolved subspace
and run a small CASCI/SCI solve, (iii) keep the orbital-occupation
information from the best batch as a "carryover" prior, (iv) repeat until
the energy stops moving (`energy_tol`) or `max_iterations` is hit.
`symmetrize_spin` is auto-disabled when `n_alpha != n_beta` since SQD's
spin-symmetrisation is only valid in the Sz=0 sector. The function also
computes the exact FCI energy via pyscf as a reference, prints the energy
history and percent error, and returns the per-iteration energy list — the
final entry is the SKQD estimate that goes into the `SKQD_RESULT_JSON` line
the sweeper consumes.
