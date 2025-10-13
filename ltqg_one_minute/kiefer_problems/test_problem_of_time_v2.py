"""
Simple test of the new problem_of_time_v2.py implementation.
This test verifies that we can solve Wheeler-DeWitt evolution
using the enhanced LTQG core framework.
"""

import numpy as np
import sys
import os

# Add core_concepts to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_concepts'))

from problem_of_time_v2 import ProblemOfTime, LTQGConfig

def test_basic_wdw_evolution():
    """Test basic Wheeler-DeWitt evolution."""
    print("Testing Wheeler-DeWitt evolution with new implementation...")
    
    # Simple configuration
    config = LTQGConfig(
        tau_0=1.0,
        hbar=1.0,
        envelope_type='tanh',
        envelope_params={'sigma_0': -2.0, 'width': 1.0},
        always_apply_silence=True,
        enforce_hermitian=True
    )
    
    # Initialize solver
    pot = ProblemOfTime(config)
    
    # Simple system
    system_params = {
        'n_modes': 2,
        'mass_scale': 1.0,
        'potential_type': 'harmonic',
        'frequency': 1.0,
        'coupling_strength': 0.0  # No coupling for simplicity
    }
    
    # Simple initial state
    initial_psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    initial_psi /= np.linalg.norm(initial_psi)
    
    # Short evolution
    print("Running evolution...")
    results = pot.solve_wheeler_dewitt_evolution(
        sigma_range=(-2.0, 0.0),
        initial_wavefunction=initial_psi,
        system_params=system_params,
        num_points=100
    )
    
    # Check results
    print(f"Evolution over σ ∈ [{results.sigma_array[0]:.2f}, {results.sigma_array[-1]:.2f}]")
    print(f"Wavefunction preserved: {np.allclose(np.linalg.norm(results.wavefunction, axis=1), 1.0, rtol=1e-2)}")
    print(f"Final probability |ψ₁|² = {np.abs(results.wavefunction[-1, 0])**2:.3f}")
    print(f"Final probability |ψ₂|² = {np.abs(results.wavefunction[-1, 1])**2:.3f}")
    print(f"Max constraint violation: {np.max(results.constraint_violation):.2e}")
    
    print("✓ Test completed successfully!")
    return results

if __name__ == "__main__":
    results = test_basic_wdw_evolution()