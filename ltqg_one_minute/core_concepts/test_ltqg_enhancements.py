#!/usr/bin/env python3
"""
Quick verification test for LTQG framework enhancements.

This script runs a minimal test to verify all four enhancements are working:
1. First-class silence in H_eff
2. Early Hermiticity enforcement  
3. Centralized configuration
4. Polished reporting
"""

import numpy as np
from ltqg_core import LTQGFramework, LTQGConfig


def test_enhancements():
    """Quick test of all four enhancements."""
    
    print("Testing LTQG Framework Enhancements...")
    
    # Test 1: First-class silence
    config_silence = LTQGConfig(always_apply_silence=True)
    config_no_silence = LTQGConfig(always_apply_silence=False)
    
    ltqg_yes = LTQGFramework(config_silence)
    ltqg_no = LTQGFramework(config_no_silence)
    
    def H_test(tau):
        return np.array([[1.0, 0.1], [0.1, -1.0]], dtype=complex)
    
    sigma = -2.0  # Deep in silence region
    H_eff_no_silence = ltqg_no.effective_hamiltonian(sigma, H_test)
    H_eff_with_silence = ltqg_yes.effective_hamiltonian(sigma, H_test)
    
    silence_ratio = np.linalg.norm(H_eff_with_silence) / np.linalg.norm(H_eff_no_silence)
    
    # Debug: print the actual silence factor
    silence_factor = ltqg_yes.silence_envelope(sigma)
    print(f"   Silence factor at σ={sigma}: {silence_factor:.6f}")
    print(f"   Expected tanh((σ + σ₀)/w) = tanh((-2 + 2)/1) = tanh(0) = 0")
    print(f"   So envelope = 0.5 * (1 + tanh(0)) = 0.5")
    
    print(f"✓ First-class silence: {silence_ratio:.3f} ≈ 0.5 (silence active)")
    assert abs(silence_ratio - 0.5) < 0.1, f"Silence should reduce H_eff to ~0.5, got {silence_ratio}"
    
    # Test 2: Hermiticity enforcement
    def H_non_hermitian(tau):
        return np.array([[1.0, 0.1+1e-4j], [0.1-1e-4j+1e-6j, -1.0]], dtype=complex)
    
    config_hermitian = LTQGConfig(enforce_hermitian=True)
    ltqg_herm = LTQGFramework(config_hermitian)
    
    H_eff_raw = ltqg_herm.effective_hamiltonian(0.0, H_non_hermitian, enforce_hermitian=False)
    H_eff_herm = ltqg_herm.effective_hamiltonian(0.0, H_non_hermitian, enforce_hermitian=True)
    
    hermiticity_error_raw = np.linalg.norm(H_eff_raw - H_eff_raw.conj().T)
    hermiticity_error_herm = np.linalg.norm(H_eff_herm - H_eff_herm.conj().T)
    
    print(f"✓ Hermiticity enforcement: {hermiticity_error_herm:.2e} < {hermiticity_error_raw:.2e}")
    assert hermiticity_error_herm < hermiticity_error_raw * 0.1 or hermiticity_error_herm < 1e-14, "Hermiticity should be improved"
    
    # Test 3: Centralized configuration
    custom_config = LTQGConfig(tau_0=2.5, envelope_type='exponential')
    ltqg_custom = LTQGFramework(custom_config)
    
    # Verify config propagation
    assert ltqg_custom.tau_0 == 2.5, "tau_0 should match config"
    assert ltqg_custom.config.envelope_type == 'exponential', "envelope_type should match config"
    
    print(f"✓ Centralized config: τ₀={ltqg_custom.tau_0}, envelope={ltqg_custom.config.envelope_type}")
    
    # Test 4: Polished reporting
    psi_test = np.array([0.6 + 1e-16j, 0.8], dtype=complex)
    measures = ltqg_custom.quantum_state_measures(psi_test)
    
    # Check that all measures are real floats (no complex artifacts)
    for key, value in measures.items():
        assert isinstance(value, float), f"{key} should be real float, got {type(value)}"
        assert np.isreal(value), f"{key} should be real, got {value}"
    
    print(f"✓ Polished reporting: all measures are clean floats")
    print(f"   Norm: {measures['norm']:.6f} (no 'j' suffix)")
    
    # Integration test: Evolution with all features
    config_full = LTQGConfig(
        always_apply_silence=True,
        enforce_hermitian=True,
        tau_0=1.0
    )
    ltqg_full = LTQGFramework(config_full)
    
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    result = ltqg_full.sigma_unitary_evolution((-1.0, 1.0), psi_0, H_test, steps=50)
    
    final_measures = ltqg_full.quantum_state_measures(result['psi_final'])
    
    print(f"✓ Integration test: Final norm = {final_measures['norm']:.6f}")
    assert abs(final_measures['norm'] - 1.0) < 1e-10, "Norm should be preserved"
    
    print("\nAll enhancements verified successfully! 🎉")
    return True


if __name__ == "__main__":
    test_enhancements()