"""
Step 4: Construct SIAM Hamiltonian in momentum basis.
"""

import numpy as np

import skqd_helpers


def siam_hamiltonian_momentum(
    num_orbs: int,
    hopping: float,
    onsite: float,
    hybridization: float,
    chemical_potential: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct SIAM Hamiltonian in momentum basis.
    
    Args:
        num_orbs: Number of spatial orbitals
        hopping: Hopping parameter
        onsite: Onsite energy
        hybridization: Hybridization strength
        chemical_potential: Chemical potential
        
    Returns:
        Tuple of (h1e, h2e) arrays in momentum basis.
    """
    h1e, h2e = skqd_helpers.siam_hamiltonian(
        num_orbs, hopping, onsite, hybridization, chemical_potential
    )
    
    orbital_rotation = skqd_helpers.momentum_basis(num_orbs)
    
    # Transpose orbital_rotation to get proper U.H @ H @ U transformation
    h1e_momentum, h2e_momentum = skqd_helpers.rotated(h1e, h2e, orbital_rotation.T)
    
    return h1e_momentum, h2e_momentum


def run_step1(
    num_orbs: int = 10,
    hopping: float = 1.0,
    onsite: float = 5.0,
    hybridization: float = 1.0,
    filling_factor: float = -0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run step 1: construct SIAM Hamiltonian in momentum basis.
    
    Args:
        num_orbs: Number of spatial orbitals
        hopping: Hopping parameter
        onsite: Onsite energy (U)
        hybridization: Hybridization strength
        filling_factor: Multiplier for chemical potential
            (chemical_potential = filling_factor * onsite)
        
    Returns:
        Tuple of (h1e, h2e) arrays in momentum basis.
    """
    chemical_potential = filling_factor * onsite

    result = siam_hamiltonian_momentum(num_orbs, hopping, onsite,
        hybridization, chemical_potential)
    
    print(f"h1e shape: {result[0].shape}")
    print(f"h2e shape: {result[1].shape}")
    print(f"Step 1 passed: SIAM Hamiltonian ({num_orbs} orbitals).")
    
    return result

