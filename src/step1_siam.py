"""
Step 1: Construct AIM Hamiltonian.

Unified interface via multi_orbital_aim_hamiltonian.  Single-impurity star
geometry is the special case num_imp_orbs=1 (with U'=0, J_H=0).
"""

import numpy as np

import skqd_helpers


def run_step1(
    num_imp_orbs: int = 1,
    num_bath_per_imp: int = 1,
    onsite: float = 4.0,
    hybridization: float | list[float] = 0.8,
    mu: float = 2.0,
    U_prime: float | None = None,
    J_H: float | None = None,
    crystal_field: list[float] | None = None,
    bath_energies: list[float] | None = None,
    bath_couplings: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct AIM Hamiltonian.

    Args:
        num_imp_orbs: Number of impurity (correlated) orbitals M
        num_bath_per_imp: Bath sites per impurity orbital B
        onsite: Intra-orbital Coulomb repulsion U
        hybridization: Impurity-bath coupling V (used only if bath_couplings
            is None)
        mu: Chemical potential
        U_prime: Inter-orbital Coulomb (defaults to U - 2*J_H)
        J_H: Hund's exchange coupling (defaults to 0)
        crystal_field: Crystal-field energies per impurity orbital
        bath_energies: Per-bath-site on-site energy ε, length M*B (row-major
            over (m, b)).  None → zeros.
        bath_couplings: Per-bath-site coupling V, length M*B (row-major over
            (m, b)).  None → broadcast ``hybridization``.

    Returns:
        Tuple of (h1e, h2e) arrays.
    """
    if J_H is None:
        J_H = 0.0
    if U_prime is None:
        U_prime = onsite - 2.0 * J_H

    V_arg = bath_couplings if bath_couplings is not None else hybridization

    h1e, h2e = skqd_helpers.multi_orbital_aim_hamiltonian(
        num_imp_orbs=num_imp_orbs,
        num_bath_per_imp=num_bath_per_imp,
        U=onsite,
        U_prime=U_prime,
        J_H=J_H,
        mu=mu,
        hybridization=V_arg,
        crystal_field=crystal_field,
        bath_energies=bath_energies,
    )
    num_orbs = num_imp_orbs * (1 + num_bath_per_imp)
    print(f"h1e shape: {h1e.shape}")
    print(f"h2e shape: {h2e.shape}")
    v_str = bath_couplings if bath_couplings is not None else hybridization
    eps_str = bath_energies if bath_energies is not None else "0"
    print(
        f"Step 1 passed: AIM "
        f"({num_imp_orbs} imp × {num_bath_per_imp} bath, "
        f"{num_orbs} orbs, {2*num_orbs} qubits, "
        f"U={onsite}, U'={U_prime}, J_H={J_H}, mu={mu}, "
        f"V={v_str}, eps={eps_str})."
    )
    return h1e, h2e
