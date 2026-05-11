
from typing import Sequence

from json.encoder import JSONEncoder
from json.decoder import JSONDecoder
import time

from functools import partial
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)

import ffsim
import scipy

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.circuit.library import CPhaseGate, PauliEvolutionGate, XGate, XXPlusYYGate
from qiskit.primitives import BitArray
from qiskit.quantum_info import Pauli, SparsePauliOp


def exact_ground_state_energy(
    H_spo: SparsePauliOp, eigvals: np.ndarray | None = None
) -> float:
    """Calculates the exact ground state energy of the given Hamiltonian.

    Args:
        H_spo: Hamiltonian as a SparsePauliOp
        eigvals: precomputed eigenvalues; if None, diagonalize H_spo

    Returns:
        Ground state energy of the Hamiltonian.
    """
    if eigvals is None:
        eigvals = np.linalg.eigvalsh(H_spo.to_matrix())
    return float(np.min(eigvals).real)


def estimate_hamiltonian_scale(
    h1e: np.ndarray, h2e: np.ndarray
) -> tuple[float, float, float]:
    """Per-Pauli-term energy scale used to set the Trotter time step.

    Returns (omega_h1e, omega_h2e, omega_total) where:
        omega_h1e   = ||h1e||_2  (one-body spectral radius)
        omega_h2e   = max |h2e[p,q,r,s]|  (largest single-pair interaction)
        omega_total = omega_h1e + omega_h2e

    omega_total is a per-Pauli-term scale, NOT the operator norm ||H||. It
    bounds the rotation angle of any single Pauli term per Trotter step,
    which controls per-step product-formula error magnitude. For an N-electron
    system <H> grows with N but per-step Trotter error is governed by per-term
    rotation magnitude.

    Note for Krylov-SQD users: minimizing per-step Trotter error and maximizing
    SQD subspace coverage are different goals. SQD wants Krylov states that
    spread across the Hilbert space, which means *large* dt; product-formula
    accuracy wants *small* dt. The 'h1e' dt-scale mode (using only omega_h1e)
    sits in the empirically-good large-dt regime for SQD. Use omega_total when
    you need an accurate evolution operator (and pair it with substeps and/or
    a larger dt_mult to recover SQD subspace coverage).
    """
    omega_h1e = float(np.linalg.norm(h1e, ord=2))
    omega_h2e = float(np.max(np.abs(h2e))) if h2e.size else 0.0
    return omega_h1e, omega_h2e, omega_h1e + omega_h2e


def dt_from_spectral_norm(H_spo: SparsePauliOp) -> float:
    """Calculate dt based on Hamiltonian spectral norm.

    Note that this is a theoretical optimal dt. Heuristically, the optimal can
    be 6-10x the theoretical value.

    Args:
        H_spo: Hamiltonian represented as a SparsePauliOp

    Returns:
        An optimal Trotter timestep
    """
    num_qubits = H_spo.num_qubits
    assert num_qubits is not None
    single_particle_H = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        for j in range(i + 1):
            for p, coeff in H_spo.to_list():
                p_x = Pauli(p).x
                p_z = Pauli(p).z
                if all(
                    p_x[k] == ((i == k) + (j == k)) % 2 for k in range(num_qubits)
                ):
                    sgn = (
                        (-1j) ** sum(p_z[k] and p_x[k] for k in range(num_qubits))
                    ) * ((-1) ** p_z[i])
                else:
                    sgn = 0
                single_particle_H[i, j] += sgn * coeff
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            single_particle_H[i, j] = np.conj(single_particle_H[j, i])

    # dt based on spectral norm
    return float(np.pi / np.linalg.norm(single_particle_H, ord=2))


def siam_hamiltonian(
    num_baths: int,
    onsite: float,
    chemical_potential: float,
    hybridization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hamiltonian for the single-impurity Anderson model (star geometry).

    Impurity at site 0 couples to every bath site with the same
    hybridization V.  No bath-bath hopping.  Matches the
    parameterization used by qc-dft-dmft: ``aim_hamiltonian(U, mu, V, num_baths)``.

    Args:
        num_baths: Number of bath sites (num_orbs = num_baths + 1)
        onsite: On-site Coulomb repulsion U on impurity
        chemical_potential: Chemical potential mu on impurity
        hybridization: Impurity-bath coupling V (uniform)

    Returns:
        One- and two-body terms of Hamiltonian (h1e, h2e) with
        num_orbs = num_baths + 1.
    """
    num_orbs = num_baths + 1
    impurity_orb = 0

    # One-body: impurity chemical potential + impurity-bath hybridization
    h1e = np.zeros((num_orbs, num_orbs))
    h1e[impurity_orb, impurity_orb] = chemical_potential
    for b in range(1, num_orbs):
        h1e[impurity_orb, b] = -hybridization
        h1e[b, impurity_orb] = -hybridization

    # Two-body: on-site U on impurity only
    h2e = np.zeros((num_orbs, num_orbs, num_orbs, num_orbs))
    h2e[impurity_orb, impurity_orb, impurity_orb, impurity_orb] = onsite

    return h1e, h2e


def multi_orbital_aim_hamiltonian(
    num_imp_orbs: int,
    num_bath_per_imp: int,
    U: float,
    U_prime: float,
    J_H: float,
    mu: float,
    hybridization: float | list[float],
    crystal_field: list[float] | None = None,
    bath_energies: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-orbital Anderson impurity model Hamiltonian (star geometry).

    Models *num_imp_orbs* correlated impurity orbitals (e.g. f-shell), each
    coupled to its own set of *num_bath_per_imp* bath orbitals.  Includes
    intra-orbital U, inter-orbital U', Hund's coupling J_H, crystal-field
    splitting, and per-bath-site hybridization V and on-site energy ε.

    Orbital ordering (spatial): imp_0, imp_1, ..., imp_{M-1},
        bath_0_0, ..., bath_0_{B-1}, bath_1_0, ..., bath_{M-1}_{B-1}
    Bath site (m, b) lives at spatial-orbital index M + m*B + b.

    Total spatial orbitals = M + M*B = M*(1+B).
    Total qubits = 2 * M * (1 + B).

    Two-body integrals use physicist (chemist-1) convention:
        h2e[p,q,r,s] corresponds to 1/2 * a†_p a†_r a_s a_q  (physicists')

    Args:
        num_imp_orbs: Number of impurity (correlated) orbitals M
        num_bath_per_imp: Number of bath sites per impurity orbital B
        U: Intra-orbital Coulomb repulsion (same orbital, opposite spin)
        U_prime: Inter-orbital Coulomb repulsion (different orbitals)
        J_H: Hund's exchange coupling
        mu: Chemical potential (applied to all impurity orbitals)
        hybridization: Impurity-bath coupling V.  Scalar (uniform across all
            M*B bath sites), list of length M (per-orbital, broadcast across
            B baths), or list of length M*B (per-bath-site, row-major over
            (m, b)).
        crystal_field: Crystal-field energies for each impurity orbital.
            List of length M.  Defaults to zero (degenerate).
        bath_energies: On-site energy ε for each bath site, list of length
            M*B (row-major over (m, b)).  Defaults to zeros.

    Returns:
        (h1e, h2e) with shape (num_orbs, num_orbs) and
        (num_orbs, num_orbs, num_orbs, num_orbs).
    """
    M = num_imp_orbs
    B = num_bath_per_imp
    num_orbs = M * (1 + B)

    # Broadcast hybridization to length M*B
    if isinstance(hybridization, (int, float)):
        V = [float(hybridization)] * (M * B)
    else:
        V_in = list(hybridization)
        if len(V_in) == M and B != 1:
            V = [float(V_in[m]) for m in range(M) for _ in range(B)]
        elif len(V_in) == M * B:
            V = [float(v) for v in V_in]
        elif len(V_in) == M and B == 1:
            V = [float(v) for v in V_in]
        else:
            raise ValueError(
                f"hybridization length {len(V_in)} must be scalar, M={M}, "
                f"or M*B={M*B}"
            )

    # Bath energies (default zero)
    if bath_energies is None:
        eps = [0.0] * (M * B)
    else:
        eps = [float(e) for e in bath_energies]
        if len(eps) != M * B:
            raise ValueError(
                f"bath_energies length {len(eps)} must equal M*B={M*B}"
            )

    # Crystal field
    if crystal_field is None:
        cf = [0.0] * M
    else:
        cf = list(crystal_field)
        assert len(cf) == M

    # --- One-body terms ---
    h1e = np.zeros((num_orbs, num_orbs))

    # Impurity on-site: crystal field - mu
    for m in range(M):
        h1e[m, m] = cf[m] - mu

    # Impurity-bath hybridization and bath on-site energies.
    # NOTE: -mu is applied only to impurity orbitals (canonical convention,
    # consistent with prior behavior); bath sites carry only eps[site].
    for m in range(M):
        for b in range(B):
            bath_idx = M + m * B + b
            site = m * B + b
            h1e[bath_idx, bath_idx] = eps[site]
            h1e[m, bath_idx] = -V[site]
            h1e[bath_idx, m] = -V[site]

    # Two-body integrals — standard rotationally-invariant Kanamori in
    # physicist convention (matches ffsim.MolecularHamiltonian and PySCF):
    #
    #   H_2 = (1/2) sum_{pqrs,sigma sigma'} h2e[p,q,r,s]
    #             a^dagger_{p sigma} a^dagger_{r sigma'} a_{s sigma'} a_{q sigma}
    #
    # For the textbook Kanamori target Hamiltonian
    #   H_K = U sum_m n_{m up} n_{m dn}
    #       + U' sum_{m<m', sigma!=sigma'} n_{m sigma} n_{m' sigma'}
    #       + (U' - J_H) sum_{m<m',sigma} n_{m sigma} n_{m' sigma}
    #       - J_H sum_{m<m'} (S+_m S-_{m'} + h.c.)
    #       + J_H sum_{m<m'} (a^dag_{m up} a^dag_{m dn} a_{m' dn} a_{m' up} + h.c.)
    # the unique 8-fold-symmetric h2e entries per impurity-orbital pair {m,m'} are:
    #   direct  : h2e[m,m,m',m'] = h2e[m',m',m,m] = U'
    #   exchange: 4 perm-equivalent positions of (m,m',m',m) all = +J_H
    #
    # Sign sanity: a^dag_{m up} a^dag_{m' dn} a_{m dn} a_{m' up} = -S+_m S-_{m'},
    # so a +J_H exchange ERI gives -J_H spin-flip (textbook).
    h2e = np.zeros((num_orbs, num_orbs, num_orbs, num_orbs))

    for m in range(M):
        h2e[m, m, m, m] = U

    for m in range(M):
        for mp in range(M):
            if m == mp:
                continue
            h2e[m, m, mp, mp] += U_prime  # direct Coulomb
            h2e[m, mp, mp, m] += J_H       # exchange (Hund's)
            h2e[m, mp, m, mp] += J_H       # exchange (pair-hopping partner)

    return h1e, h2e


def _canonicalize_real_eigvec_signs(V: np.ndarray, anchor_row: int = 0) -> np.ndarray:
    V = V.copy()
    signs = np.sign(V[anchor_row, :])
    # fallback for exact zeros: use the largest-magnitude element as anchor
    zero = np.isclose(signs, 0.0)
    if np.any(zero):
        js = np.where(zero)[0]
        i_max = np.argmax(np.abs(V[:, js]), axis=0)
        signs[js] = np.sign(V[i_max, js])
    signs[signs == 0] = 1.0
    return V * signs

def momentum_basis(norb: int) -> np.ndarray:
    """Get the orbital rotation to change from the position to the momentum basis.

    Args:
        norb: Number of spatial orbitals

    Returns:
        Matrix corresponding to orbital rotation for position -> momentum basis.
    """
    n_bath = norb - 1

    # Orbital rotation that diagonalizes the bath (non-interacting system)
    hopping_matrix = np.zeros((n_bath, n_bath))
    np.fill_diagonal(hopping_matrix[:, 1:], -1)
    np.fill_diagonal(hopping_matrix[1:, :], -1)
    _, vecs = np.linalg.eigh(hopping_matrix)

    # >>> deterministic sign convention <<<
    vecs = _canonicalize_real_eigvec_signs(vecs, anchor_row=0)

    # Expand to include impurity
    orbital_rotation = np.zeros((norb, norb))
    # Impurity is on the first site
    orbital_rotation[0, 0] = 1
    orbital_rotation[1:, 1:] = vecs

    # Move the impurity to the center
    new_index = n_bath // 2
    perm = np.r_[1: (new_index + 1), 0, (new_index + 1): norb]
    orbital_rotation = orbital_rotation[:, perm]

    return orbital_rotation

def rotated(
    h1e: np.ndarray, h2e: np.ndarray, orbital_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the orbital basis of a Hamiltonian.

    Args:
        h1e: One-body term
        h2e: Two-body term
        orbital_rotation: Position-to-momentum basis rotation matrix

    Returns:
        One- and two-body Hamiltonian rotated from position to momentum basis.
    """
    h1e_rotated = np.einsum(
        "ab,Aa,Bb->AB",
        h1e,
        orbital_rotation,
        orbital_rotation.conj(),
        optimize="greedy",
    )
    h2e_rotated = np.einsum(
        "abcd,Aa,Bb,Cc,Dd->ABCD",
        h2e,
        orbital_rotation,
        orbital_rotation.conj(),
        orbital_rotation,
        orbital_rotation.conj(),
        optimize="greedy",
    )
    return h1e_rotated, h2e_rotated


def prepare_initial_state(
    qubits: Sequence[Qubit], num_orbs: int, n_alpha: int, n_beta: int
):
    """Prepare a Sz-resolved product initial state on (alpha, beta) qubits.

    Places X gates on the first ``n_alpha`` alpha qubits (indices 0..n_alpha-1)
    and the first ``n_beta`` beta qubits (indices num_orbs..num_orbs+n_beta-1),
    producing a single Slater determinant in the (n_alpha, n_beta) particle-
    number sector.

    A small XXPlusYY mixing layer is applied only at half-filling
    (n_alpha == n_beta == num_orbs // 2) AND num_orbs >= 4 — it relies on
    adjacent occupied/empty pairing which breaks at extreme fillings.

    Args:
        qubits: Ordered collection of qubits (length 2*num_orbs)
        num_orbs: Number of spatial orbitals
        n_alpha: Number of spin-up electrons
        n_beta: Number of spin-down electrons

    Returns:
        Initial state generator.
    """
    x_gate = XGate()
    for i in range(n_alpha):
        yield CircuitInstruction(x_gate, [qubits[i]])
    for i in range(n_beta):
        yield CircuitInstruction(x_gate, [qubits[num_orbs + i]])

    half = num_orbs // 2
    if not (n_alpha == n_beta == half and num_orbs >= 4):
        return

    rot = XXPlusYYGate(0.5 * np.pi, -0.5 * np.pi)
    occ_num = half
    # The inner loop below always runs at least once for i=0; the explicit
    # init silences the static-analysis "possibly unbound" warning on the
    # post-loop `j + 2 < num_orbs` check.
    j = 0
    for i in range(3):
        for j in range(occ_num - i - 1, occ_num + i, 2):
            if j + 1 < num_orbs:
                yield CircuitInstruction(rot, [qubits[j], qubits[j + 1]])
                yield CircuitInstruction(
                    rot, [qubits[num_orbs + j], qubits[num_orbs + j + 1]]
                )
    # Final rotation pair - only if indices are valid
    if j + 2 < num_orbs:
        yield CircuitInstruction(rot, [qubits[j + 1], qubits[j + 2]])
        yield CircuitInstruction(
            rot, [qubits[num_orbs + j + 1], qubits[num_orbs + j + 2]]
        )


def _emit_diag(
    qubits: Sequence[Qubit],
    h2e: np.ndarray,
    time: float,
    num_imp_orbs: int,
    num_orbs: int,
):
    """Yield CPhaseGate instructions for e^{-i H_diag * time}.

    Standard rotationally-invariant Kanamori, per impurity-orbital pair {m, mp}:
        intra:      U n_{m up} n_{m dn}
        same-spin:  (U' - J_H) n_{m sigma} n_{mp sigma}
        cross-spin: U' n_{m sigma} n_{mp sigma'}

    The same-spin (-J_H) shift comes from the diagonal (sigma=sigma') part of
    the +J_H exchange ERI entries h2e[m,mp,mp,m] = h2e[mp,m,m,mp]; cross-spin
    is unmodified by exchange. The off-diagonal Hund's spin-flip + pair-hop
    is handled separately by _emit_offdiag.
    """
    if abs(time) < 1e-15:
        return
    no = num_orbs
    for m in range(num_imp_orbs):
        u_mm = h2e[m, m, m, m]
        if abs(u_mm) > 1e-12:
            yield CircuitInstruction(
                CPhaseGate(-time * u_mm),
                [qubits[m], qubits[no + m]],
            )
    for m in range(num_imp_orbs):
        for mp in range(m + 1, num_imp_orbs):
            u_prime = h2e[m, m, mp, mp]
            j_H = h2e[m, mp, mp, m]  # = +J_H in standard Kanamori convention
            if abs(u_prime) < 1e-12 and abs(j_H) < 1e-12:
                continue
            c_ss = -time * (u_prime - j_H)
            c_xs = -time * u_prime
            yield CircuitInstruction(CPhaseGate(c_ss), [qubits[m], qubits[mp]])
            yield CircuitInstruction(
                CPhaseGate(c_ss), [qubits[no + m], qubits[no + mp]]
            )
            yield CircuitInstruction(CPhaseGate(c_xs), [qubits[m], qubits[no + mp]])
            yield CircuitInstruction(CPhaseGate(c_xs), [qubits[no + m], qubits[mp]])


def _emit_offdiag(qubit_list, offdiag_op, time):
    """Yield PauliEvolutionGate for e^{-i H_offdiag * time} (spin-flip + pair-hop)."""
    if offdiag_op is None or abs(time) < 1e-15:
        return
    yield CircuitInstruction(PauliEvolutionGate(offdiag_op, time=time), qubit_list)


def _emit_h1e(qubits, h1e: np.ndarray, num_orbs: int, time: float):
    """Yield ffsim's OrbitalRotationJW gate for e^{-i H_1 * time}."""
    if abs(time) < 1e-15:
        return
    h1e_u = scipy.linalg.expm(-1j * time * h1e)
    yield CircuitInstruction(
        ffsim.qiskit.OrbitalRotationJW(num_orbs, h1e_u), qubits
    )


def _hund_offdiag_op(
    h2e: np.ndarray, num_imp_orbs: int, num_orbs: int
) -> SparsePauliOp | None:
    """Build the JW-mapped SparsePauliOp for the off-diagonal Kanamori terms.

    For each impurity-orbital pair {m, mp} (standard rotationally-invariant Kanamori):
        spin-flip:    -J_H (S+_m S-_{mp} + h.c.)
        pair-hopping: +J_H (a^dagger_{m up} a^dagger_{m dn} a_{mp dn} a_{mp up} + h.c.)

    Implemented as J_H * sum of (a^dag a^dag a a) operator products. The
    sign-flip on the spin-flip side comes from the operator algebra:
    a^dag_{m up} a^dag_{mp dn} a_{m dn} a_{mp up} = -S+_m S-_{mp},
    so coef +J_H here yields the textbook -J_H S+S- in the Hamiltonian.
    """
    terms: dict = {}
    any_nonzero = False
    for m in range(num_imp_orbs):
        for mp in range(m + 1, num_imp_orbs):
            j_H = h2e[m, mp, mp, m]
            if abs(j_H) < 1e-12:
                continue
            any_nonzero = True
            coef = j_H
            terms[(ffsim.cre_a(m), ffsim.cre_b(mp),
                   ffsim.des_b(m), ffsim.des_a(mp))] = coef
            terms[(ffsim.cre_a(mp), ffsim.cre_b(m),
                   ffsim.des_b(mp), ffsim.des_a(m))] = coef
            terms[(ffsim.cre_a(m), ffsim.cre_b(m),
                   ffsim.des_b(mp), ffsim.des_a(mp))] = coef
            terms[(ffsim.cre_a(mp), ffsim.cre_b(mp),
                   ffsim.des_b(m), ffsim.des_a(m))] = coef
    if not any_nonzero:
        return None
    return ffsim.qiskit.jordan_wigner(ffsim.FermionOperator(terms), norb=num_orbs)


def _lie_step(qubits, qubit_list, dt, h1e, h2e, offdiag_op, num_imp_orbs, num_orbs):
    """Order-1 Lie-Trotter: e^{-iHdt} ~ e^{-iH_off dt} e^{-iH_diag dt} e^{-iH_1 dt}."""
    yield from _emit_offdiag(qubit_list, offdiag_op, dt)
    yield from _emit_diag(qubits, h2e, dt, num_imp_orbs, num_orbs)
    yield from _emit_h1e(qubits, h1e, num_orbs, dt)


def _strang_step(qubits, qubit_list, dt, h1e, h2e, offdiag_op, num_imp_orbs, num_orbs):
    """Suzuki S2 (Strang). Outer e^{-iH_2 dt/2} e^{-iH_1 dt} e^{-iH_2 dt/2},
    inner Strang for H_2 = H_diag + H_offdiag.
    """
    yield from _emit_offdiag(qubit_list, offdiag_op, dt / 4)
    yield from _emit_diag(qubits, h2e, dt / 2, num_imp_orbs, num_orbs)
    yield from _emit_offdiag(qubit_list, offdiag_op, dt / 4)
    yield from _emit_h1e(qubits, h1e, num_orbs, dt)
    yield from _emit_offdiag(qubit_list, offdiag_op, dt / 4)
    yield from _emit_diag(qubits, h2e, dt / 2, num_imp_orbs, num_orbs)
    yield from _emit_offdiag(qubit_list, offdiag_op, dt / 4)


# Suzuki 4th-order (5-step) coefficient. Reference:
#   M. Suzuki, "General theory of fractal path integrals with applications to
#   many-body theories and statistical physics", J. Math. Phys. 32, 400 (1991).
_SUZUKI_S = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))


def _suzuki4_step(qubits, qubit_list, dt, h1e, h2e, offdiag_op, num_imp_orbs, num_orbs):
    """Suzuki S4 (order 4): five Strang sub-steps with weights [s, s, 1-4s, s, s],
    s = 1/(4 - 4^{1/3}). Middle weight is negative (back-evolution).
    """
    s = _SUZUKI_S
    for w in (s, s, 1.0 - 4.0 * s, s, s):
        yield from _strang_step(
            qubits, qubit_list, w * dt, h1e, h2e, offdiag_op, num_imp_orbs, num_orbs
        )


_TROTTER_STEP_FNS = {1: _lie_step, 2: _strang_step, 4: _suzuki4_step}


def trotter_step(
    qubits: Sequence[Qubit],
    time_step: float,
    h1e: np.ndarray,
    h2e: np.ndarray,
    impurity_index: int,
    num_orbs: int,
    num_imp_orbs: int = 1,
    order: int = 2,
    substeps: int = 1,
):
    """Apply e^{-iH * time_step} approximately via product-formula Trotterization.

    H is split as H = H_1 (one-body) + H_diag (diagonal Kanamori 2-body) +
    H_offdiag (Hund's spin-flip + pair-hopping). Order-2 uses an outer Strang
    on H_2 vs H_1 with an inner Strang on H_diag vs H_offdiag; order-4 wraps
    five order-2 sub-steps (Suzuki S4).

    Args:
        qubits: Ordered collection of qubits (length 2 * num_orbs).
        time_step: Total evolution time t for this step.
        h1e: One-body Hamiltonian (num_orbs x num_orbs).
        h2e: Two-body Hamiltonian (num_orbs^4), 8-fold symmetric.
        impurity_index: Center-orbital index (retained for API compatibility).
        num_orbs: Number of spatial orbitals.
        num_imp_orbs: Number of correlated impurity orbitals (1 for SIAM).
        order: Suzuki product-formula order. 1 (Lie), 2 (Strang), 4 (Suzuki S4).
        substeps: Subdivide [0, time_step] into N equal sub-steps. Per-sub-step
            local Trotter error scales as (time_step/substeps)^{order+1}.

    Yields:
        CircuitInstruction objects implementing the chosen product formula.
    """
    step_fn = _TROTTER_STEP_FNS.get(order)
    if step_fn is None:
        raise ValueError(
            f"Unsupported trotter order: {order} (supported: 1, 2, 4)"
        )
    qubit_list = list(qubits)
    offdiag_op = _hund_offdiag_op(h2e, num_imp_orbs, num_orbs)
    sub_dt = time_step / substeps
    for _ in range(substeps):
        yield from step_fn(
            qubits, qubit_list, sub_dt, h1e, h2e,
            offdiag_op, num_imp_orbs, num_orbs,
        )


def _half_filled_sector_indices(num_orbs: int, n_per_spin: int) -> np.ndarray:
    """Computational-basis indices in the (n_per_spin, n_per_spin) particle sector.

    Uses ffsim convention: alpha (spin-up) on qubits 0..N-1, beta on N..2N-1.
    """
    nq = 2 * num_orbs
    alpha_mask = (1 << num_orbs) - 1
    beta_mask = alpha_mask << num_orbs
    out = []
    for i in range(1 << nq):
        if (bin(i & alpha_mask).count('1') == n_per_spin
                and bin(i & beta_mask).count('1') == n_per_spin):
            out.append(i)
    return np.array(out, dtype=np.int64)


def _full_hamiltonian_pauli(
    h1e: np.ndarray, h2e: np.ndarray, num_orbs: int
) -> SparsePauliOp:
    """Build the full second-quantized H as a JW-mapped SparsePauliOp.

    Convention (matches ffsim.MolecularHamiltonian):
        H = sum_{pq,sigma} h1e[p,q] a^dagger_{p,sigma} a_{q,sigma}
          + (1/2) sum_{pqrs,sigma sigma'} h2e[p,q,r,s]
              a^dagger_{p,sigma} a^dagger_{r,sigma'} a_{s,sigma'} a_{q,sigma}
    """
    from collections import defaultdict
    terms: dict = defaultdict(complex)
    spin_ops = ((ffsim.cre_a, ffsim.des_a), (ffsim.cre_b, ffsim.des_b))

    for p in range(num_orbs):
        for q in range(num_orbs):
            v = h1e[p, q]
            if abs(v) < 1e-14:
                continue
            for cre, des in spin_ops:
                terms[(cre(p), des(q))] += complex(v)

    for p in range(num_orbs):
        for q in range(num_orbs):
            for r in range(num_orbs):
                for s in range(num_orbs):
                    v = h2e[p, q, r, s]
                    if abs(v) < 1e-14:
                        continue
                    half_v = complex(v) / 2
                    for cre1, des1 in spin_ops:
                        for cre2, des2 in spin_ops:
                            terms[(cre1(p), cre2(r), des2(s), des1(q))] += half_v

    fop_terms = {k: complex(v) for k, v in terms.items() if abs(v) > 1e-14}
    return ffsim.qiskit.jordan_wigner(
        ffsim.FermionOperator(fop_terms), norb=num_orbs
    )


def compute_trotter_distance(
    h1e: np.ndarray,
    h2e: np.ndarray,
    dt: float,
    num_imp_orbs: int = 1,
    order: int = 2,
    substeps: int = 1,
    max_orbs: int = 6,
) -> float:
    """Operator-norm distance between Trotter and exact unitaries on the
    half-filled (N/2, N/2) particle-number sector.

    Diagnostic: a single number that bounds per-step product-formula error.
    For unitaries A, B we have |<psi|A^dagger B|psi>|^2 >= 1 - ||A - B||_2^2,
    so a per-step distance epsilon implies fidelity >= 1 - epsilon^2 for any
    state in the sector.

    Restricted to small systems (num_orbs <= max_orbs, default 6) since it
    constructs the full 2^{2N} x 2^{2N} Trotter unitary and exponentiates
    the JW-mapped Hamiltonian.

    Returns:
        ||T(dt) - exp(-i H dt)||_2 evaluated on the (N/2, N/2) sector.
    """
    from qiskit.quantum_info import Operator
    from qiskit import QuantumCircuit, QuantumRegister

    num_orbs = h1e.shape[0]
    if num_orbs > max_orbs:
        raise ValueError(
            f"compute_trotter_distance restricted to num_orbs <= {max_orbs} "
            f"(got {num_orbs}); needs 2^{{2N}} x 2^{{2N}} matrices."
        )

    nq = 2 * num_orbs
    qreg = QuantumRegister(nq, 'q')
    qc = QuantumCircuit(qreg)
    for instr in trotter_step(
        qreg, dt, h1e, h2e, (num_orbs - 1) // 2, num_orbs,
        num_imp_orbs=num_imp_orbs, order=order, substeps=substeps,
    ):
        qc.append(instr)
    T = Operator(qc).data

    H_pauli = _full_hamiltonian_pauli(h1e, h2e, num_orbs)
    H_dense = H_pauli.to_matrix()
    U_exact = scipy.linalg.expm(-1j * dt * H_dense)

    sector = _half_filled_sector_indices(num_orbs, num_orbs // 2)
    T_sec = T[np.ix_(sector, sector)]
    U_sec = U_exact[np.ix_(sector, sector)]

    return float(np.linalg.norm(T_sec - U_sec, ord=2))


def plot_comparison(
    history: list[list[SCIResult]],
    result: SCIResult,
    hamiltonian: tuple[np.ndarray, np.ndarray],
):
    """Plot SKQD energy per iteration and reference (DMRG) energy.

    Args:
        history: list of per-iteration result batches; each batch is the list
            of SCIResults across the parallel subsamples for that iteration
        result: Final SCIResult from diagonalize_fermionic_hamiltonian
        hamiltonian: (h1e, h2e) used to recompute the energy via 1- and 2-RDMs
    """
    ref_energy = -28.70659686

    min_es = [
        min(batch, key=lambda res: res.energy).energy
        for batch in history
    ]
    _, min_e = min(enumerate(min_es), key=lambda x: x[1])

    # Data for energies plot
    x1 = range(len(history))

    # Data for avg spatial orbital occupancy
    y2 = np.sum(result.orbital_occupancies, axis=0)
    x2 = range(len(y2))

    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot energies
    axs[0].plot(x1, min_es, label="energy", marker="o")
    axs[0].set_xticks(x1)
    axs[0].set_xticklabels(x1)
    axs[0].axhline(
        y=ref_energy, color="#BF5700", linestyle="--", label="DMRG energy"
    )
    axs[0].set_title("SKQD Approximated Ground State Energy")
    axs[0].set_xlabel("Iteration", fontdict={"fontsize": 12})
    axs[0].set_ylabel("Approximate Energy", fontdict={"fontsize": 12})
    axs[0].legend()

    # Plot orbital occupancy
    axs[1].bar(x2, y2, width=0.8)
    axs[1].set_xticks(x2)
    axs[1].set_xticklabels(x2)
    axs[1].set_title("Average Occupancy")
    axs[1].set_xlabel("Spatial Orbital", fontdict={"fontsize": 12})
    axs[1].set_ylabel("Average Occupancy", fontdict={"fontsize": 12})

    print(f"Reference (DMRG) Energy: {ref_energy:.5f}")
    print(f"SQD Energy: {min_e:.5f}")
    print(f"Absolute Error: {abs(min_e - ref_energy):.5f}")
    plt.tight_layout()
    plt.show()

    rdm1 = result.sci_state.rdm(rank=1, spin_summed=True)
    rdm2 = result.sci_state.rdm(rank=2, spin_summed=True)
    energy = np.sum(hamiltonian[0] * rdm1) + 0.5 * np.sum(hamiltonian[1] * rdm2)
    print(f"Verified recomputed energy: {energy:.5f}")


def classically_diagonalize(
    bit_array: BitArray | None = None,
    hcore: np.ndarray | None = None,
    eri: np.ndarray | None = None,
    num_orbitals: int | None = None,
    nelec: int | None = None,
    num_elec_a: int | None = None,
    num_elec_b: int | None = None,
    job_id: str | None = None,
    client=None,
    energy_tol: float = 1e-4,
    occupancies_tol: float = 1e-3,
    max_iterations: int = 12,
    num_batches: int = 8,
    samples_per_batch: int = 300,
    symmetrize_spin: bool = False,
    carryover_threshold: float = 1e-5,
    max_cycle: int = 200,
    mem: int = 64,
    local: bool = True,
):
    """Classical Diagonalization Engine sent to HPC.

    Args:
        bit_array: Bit string array; only needed if locally processing data
        hcore: 1-electron hamiltonian integrals
        eri: 2-electron hamiltonian integrals
        num_orbitals: Number of spatial orbitals
        nelec: Number of electrons
        num_elec_a: Alpha orbitals
        num_elec_b: Beta orbitals
        job_id: QPU bitstring Job ID
        client: Diagonalization engine worker
        energy_tol: SQD option
        occupancies_tol: SQD option
        max_iterations: SQD option
        num_batches: Eigenstate solver option
        samples_per_batch: Eigenstate solver option
        symmetrize_spin: Eigenstate solver option
        carryover_threshold: Eigenstate solver option
        max_cycle: Eigenstate solver option
        mem: distributed task memory, in GiB
        local: True to run locally, False for remote

    Returns:
        Serverless job result outputs as a tuple.
    """

    print(">>>>> Starting Diagonalization Engine...")
    # Pass options to the built-in eigensolver. To use defaults, omit the
    # sci_solver argument from the diagonalize_fermionic_hamiltonian call.
    if local:
        assert hcore is not None, "local mode requires hcore"
        assert eri is not None, "local mode requires eri"
        assert bit_array is not None, "local mode requires bit_array"
        assert num_orbitals is not None, "local mode requires num_orbitals"
        # Accept either nelec (back-compat: half-filled split) or an explicit
        # (num_elec_a, num_elec_b) pair (Sz-resolved).
        if num_elec_a is not None and num_elec_b is not None:
            nelec_tuple = (int(num_elec_a), int(num_elec_b))
        else:
            assert nelec is not None, (
                "local mode requires nelec or (num_elec_a, num_elec_b)"
            )
            nelec_tuple = (nelec // 2, nelec // 2)

        sz_target = 0.5 * (nelec_tuple[0] - nelec_tuple[1])
        spin_sq_target = sz_target * (sz_target + 1.0)
        sci_solver = partial(
            solve_sci_batch, spin_sq=spin_sq_target, max_cycle=max_cycle
        )

        # List to capture intermediate results
        result_history: list[list[SCIResult]] = []

        def callback(results: list[SCIResult]):
            result_history.append(results)
            iteration = len(result_history)
            print(f"Iteration {iteration}")
            for i, sub in enumerate(results):
                dim = np.prod(sub.sci_state.amplitudes.shape)
                print(f"\tSubsample {i}")
                print(f"\t\tEnergy: {sub.energy}")
                print(f"\t\tSubspace dimension: {dim}")

        diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=samples_per_batch,
            norb=num_orbitals,
            nelec=nelec_tuple,
            num_batches=num_batches,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            max_iterations=max_iterations,
            sci_solver=sci_solver,
            symmetrize_spin=symmetrize_spin,
            carryover_threshold=carryover_threshold,
            callback=callback,
            seed=12345,
        )

        min_es = [
            min(batch, key=lambda res: res.energy).energy
            for batch in result_history
        ]
        return [float(e) for e in min_es]

    # Serverless Logic
    assert hcore is not None, "remote mode requires hcore"
    assert eri is not None, "remote mode requires eri"
    assert num_orbitals is not None, "remote mode requires num_orbitals"
    assert num_elec_a is not None, "remote mode requires num_elec_a"
    assert num_elec_b is not None, "remote mode requires num_elec_b"
    assert client is not None, "remote mode requires client"

    print(f">>>>> Sending job {job_id} to Serverless...")

    data = [
        job_id,
        hcore.tolist(),
        eri.tolist(),
        int(num_orbitals),
        int(num_elec_a),
        int(num_elec_b),
    ]

    # Encode the execution dependencies with the JSONEncoder
    data_e = JSONEncoder().encode(data)

    # Send to Serverless
    worker = client.load("diagonalization_engine")
    serverless_job = worker.run(
        data=data_e,
        mem=mem,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        num_batches=num_batches,
        samples_per_batch=samples_per_batch,
        max_cycle=max_cycle,
    )

    # Wait for the job to execute
    print(f">>>>> Serverless status: {serverless_job.job_id}")
    result = None
    timer = 0
    while timer < 10000:
        status = serverless_job.status()
        if status in ("QUEUED", "INITIALIZING", "RUNNING"):
            print(
                f">>>>> [{timer}s] Serverless job "
                f"{serverless_job.job_id}: {status}"
            )
            time.sleep(10)
            timer += 10
        elif status == "ERROR":
            print(f">>>>> Serverless job {serverless_job.job_id}: {status}")
            print(">>>>> Logs:")
            print(serverless_job.logs())
            break
        elif status == "DONE":
            print(f">>>>> Serverless job {serverless_job.job_id}: {status}")
            o_data = JSONDecoder().decode(serverless_job.result()["outputs"])
            result = o_data[0]
            print(f">>>>>>>>>> Energies/iteration: {result}")
            break
        else:
            break

    return result
