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
    hybridization: float = 0.8,
    mu: float = 2.0,
    U_prime: float | None = None,
    J_H: float | None = None,
    crystal_field: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct AIM Hamiltonian.

    Args:
        num_imp_orbs: Number of impurity (correlated) orbitals M
        num_bath_per_imp: Bath sites per impurity orbital B
        onsite: Intra-orbital Coulomb repulsion U
        hybridization: Impurity-bath coupling V
        mu: Chemical potential
        U_prime: Inter-orbital Coulomb (defaults to U - 2*J_H)
        J_H: Hund's exchange coupling (defaults to 0)
        crystal_field: Crystal-field energies per impurity orbital

    Returns:
        Tuple of (h1e, h2e) arrays.
    """
    if J_H is None:
        J_H = 0.0
    if U_prime is None:
        U_prime = onsite - 2.0 * J_H

    h1e, h2e = skqd_helpers.multi_orbital_aim_hamiltonian(
        num_imp_orbs=num_imp_orbs,
        num_bath_per_imp=num_bath_per_imp,
        U=onsite,
        U_prime=U_prime,
        J_H=J_H,
        mu=mu,
        hybridization=hybridization,
        crystal_field=crystal_field,
    )
    num_orbs = num_imp_orbs * (1 + num_bath_per_imp)
    print(f"h1e shape: {h1e.shape}")
    print(f"h2e shape: {h2e.shape}")
    print(
        f"Step 1 passed: AIM "
        f"({num_imp_orbs} imp × {num_bath_per_imp} bath, "
        f"{num_orbs} orbs, {2*num_orbs} qubits, "
        f"U={onsite}, U'={U_prime}, J_H={J_H}, mu={mu}, V={hybridization})."
    )
    return h1e, h2e
