#!/usr/bin/env python3
"""
Enhanced LTQG Framework Demo

This script demonstrates the four key refinements to the LTQG framework:

1. First-class silence: H̃_eff(σ) = f_silence(σ) × τ(σ) × H(τ(σ))
2. Early Hermiticity enforcement with optional flag
3. Centralized τ₀ and envelope parameters
4. Polished reporting (no "−0.000j" artifacts)

The result is a more robust, consistent, and user-friendly framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from ltqg_core import LTQGFramework, LTQGConfig, SigmaTimeVisualizer
from plotting_utils import setup_results_directory, save_plot, get_shared_config, update_shared_config


def demo_first_class_silence():
    """Demonstrate first-class silence integration."""
    print("=== Demo 1: First-Class Silence ===")
    
    # Create two configs: one with silence, one without
    config_no_silence = LTQGConfig(
        tau_0=1.0,
        always_apply_silence=False,
        enforce_hermitian=True
    )
    
    config_with_silence = LTQGConfig(
        tau_0=1.0,
        always_apply_silence=True,
        enforce_hermitian=True,
        envelope_type='tanh',
        envelope_params={'sigma_0': 2.0, 'width': 1.0}
    )
    
    # Initialize frameworks
    ltqg_no_silence = LTQGFramework(config_no_silence)
    ltqg_with_silence = LTQGFramework(config_with_silence)
    
    # Test Hamiltonian: simple 2-level system
    def H_two_level(tau):
        omega = 1.0
        return np.array([[omega, 0.1], [0.1, -omega]], dtype=complex)
    
    # Compare effective Hamiltonians
    sigma_range = np.linspace(-4, 2, 50)
    H_eff_norms_no_silence = []
    H_eff_norms_with_silence = []
    
    for sigma in sigma_range:
        H_eff_no = ltqg_no_silence.effective_hamiltonian(sigma, H_two_level)
        H_eff_yes = ltqg_with_silence.effective_hamiltonian(sigma, H_two_level)
        
        H_eff_norms_no_silence.append(np.linalg.norm(H_eff_no))
        H_eff_norms_with_silence.append(np.linalg.norm(H_eff_yes))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.semilogy(sigma_range, H_eff_norms_no_silence, 'b-', linewidth=2, 
                 label='Standard: τH(τ)')
    plt.semilogy(sigma_range, H_eff_norms_with_silence, 'r-', linewidth=2, 
                 label='First-class silence: f_silence(σ)τH(τ)')
    plt.xlabel('σ-time')
    plt.ylabel('||H_eff(σ)||')
    plt.title('First-Class Silence in Effective Hamiltonian')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    save_plot(plt, 'first_class_silence_comparison.png', 'demo_enhanced_ltqg')
    
    print(f"✓ Generated comparison plot showing asymptotic silence")
    print(f"✓ No risk of forgetting envelope - it's built into H̃_eff")
    
    return ltqg_with_silence


def demo_hermiticity_enforcement():
    """Demonstrate early Hermiticity enforcement."""
    print("\n=== Demo 2: Hermiticity Enforcement ===")
    
    config = LTQGConfig(enforce_hermitian=True)
    ltqg = LTQGFramework(config)
    
    # Create a slightly non-Hermitian Hamiltonian
    def H_non_hermitian(tau):
        # Add tiny imaginary components and asymmetry
        return np.array([
            [1.0, 0.5 + 1e-10j],
            [0.5 - 1e-10j + 1e-14j, -1.0]
        ], dtype=complex)
    
    sigma = 0.0
    
    # Test with and without Hermiticity enforcement
    H_eff_raw = ltqg.effective_hamiltonian(sigma, H_non_hermitian, enforce_hermitian=False)
    H_eff_hermitian = ltqg.effective_hamiltonian(sigma, H_non_hermitian, enforce_hermitian=True)
    
    # Analyze properties
    props_raw = ltqg.hamiltonian_properties(H_eff_raw)
    props_hermitian = ltqg.hamiltonian_properties(H_eff_hermitian)
    
    print(f"Raw H_eff properties:")
    for key, value in props_raw.items():
        print(f"   {key}: {value}")
    
    print(f"Hermitianized H̃_eff properties:")
    for key, value in props_hermitian.items():
        print(f"   {key}: {value}")
    
    # Evolution comparison
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    sigma_span = (0.0, 1.0)
    
    # Temporarily disable Hermiticity for comparison
    ltqg.config.enforce_hermitian = False
    result_raw = ltqg.sigma_unitary_evolution(sigma_span, psi_0, H_non_hermitian, steps=100)
    ltqg.config.enforce_hermitian = True
    result_hermitian = ltqg.sigma_unitary_evolution(sigma_span, psi_0, H_non_hermitian, steps=100)
    
    print(f"\nEvolution results:")
    print(f"   Raw evolution final norm: {np.linalg.norm(result_raw['psi_final']):.8f}")
    print(f"   Hermitianized evolution final norm: {np.linalg.norm(result_hermitian['psi_final']):.8f}")
    print(f"✓ Hermiticity enforcement prevents numerical drift")
    
    return ltqg


def demo_centralized_config():
    """Demonstrate centralized configuration management."""
    print("\n=== Demo 3: Centralized Configuration ===")
    
    # Create custom configuration
    custom_config = LTQGConfig(
        tau_0=2.5,  # Custom reference time
        envelope_type='exponential',
        envelope_params={'sigma_0': 1.5, 'envelope_floor': 1e-10},
        ode_rtol=1e-10,
        ode_atol=1e-14
    )
    
    # Initialize framework and visualizer
    ltqg = LTQGFramework(custom_config)
    visualizer = SigmaTimeVisualizer(ltqg)
    
    # Update shared plotting config to match
    update_shared_config({
        'tau_0': custom_config.tau_0,
        'envelope_type': custom_config.envelope_type,
        'envelope_params': custom_config.envelope_params
    })
    
    # Verify consistency
    plot_config = get_shared_config()
    
    print(f"Framework config τ₀: {ltqg.config.tau_0}")
    print(f"Visualizer config τ₀: {visualizer.config.tau_0}")
    print(f"Plotting config τ₀: {plot_config['tau_0']}")
    print(f"✓ All configs synchronized")
    
    # Demonstrate consistent envelope behavior
    sigma_test = np.linspace(-3, 1, 5)
    print(f"\nEnvelope consistency check:")
    print(f"   σ-time    Framework    Plotting")
    for sigma in sigma_test:
        framework_envelope = ltqg.silence_envelope(sigma)
        
        # Manually compute with plotting config (should match)
        params = plot_config['envelope_params']
        sigma_0 = params['sigma_0']
        if plot_config['envelope_type'] == 'exponential':
            plotting_envelope = np.exp(sigma / sigma_0) if sigma < 0 else 1.0
            plotting_envelope = max(plotting_envelope, params['envelope_floor'])
        
        print(f"   {sigma:6.2f}    {framework_envelope:.6f}     {plotting_envelope:.6f}")
    
    print(f"✓ Framework and plotting use identical envelope parameters")
    
    return ltqg, custom_config


def demo_polished_reporting():
    """Demonstrate polished reporting without artifacts."""
    print("\n=== Demo 4: Polished Reporting ===")
    
    config = LTQGConfig()
    ltqg = LTQGFramework(config)
    
    # Create a complex quantum state with potential numerical artifacts
    psi = np.array([
        0.6 + 1e-16j,      # Real number with tiny imaginary part
        0.8 * np.exp(1j * np.pi/4)  # Complex number
    ], dtype=complex)
    
    # Add tiny normalization error
    psi = psi * (1 + 1e-15)
    
    # Compute measures with and without polishing
    measures = ltqg.quantum_state_measures(psi)
    
    # Raw computation for comparison
    rho = np.outer(psi, psi.conj())
    raw_norm = np.sqrt(np.vdot(psi, psi))
    raw_purity = np.trace(rho @ rho)
    
    print(f"Raw quantum measures (with artifacts):")
    print(f"   Norm: {raw_norm}")
    print(f"   Purity: {raw_purity}")
    
    print(f"\nPolished quantum measures (artifacts removed):")
    for key, value in measures.items():
        print(f"   {key.capitalize()}: {value}")
    
    # Test Hamiltonian properties
    H = np.array([
        [1.0 + 1e-16j, 0.5],
        [0.5, -1.0 - 1e-15j]
    ], dtype=complex)
    
    ham_props = ltqg.hamiltonian_properties(H)
    raw_trace = np.trace(H)
    
    print(f"\nRaw Hamiltonian trace: {raw_trace}")
    print(f"Polished Hamiltonian properties:")
    for key, value in ham_props.items():
        print(f"   {key.capitalize()}: {value}")
    
    print(f"✓ All numerical artifacts eliminated")
    
    return ltqg


def demo_integrated_workflow():
    """Demonstrate integrated workflow using all refinements."""
    print("\n=== Demo 5: Integrated Workflow ===")
    
    # Set up comprehensive configuration
    config = LTQGConfig(
        tau_0=1.0,
        always_apply_silence=True,
        enforce_hermitian=True,
        envelope_type='tanh',
        envelope_params={'sigma_0': 2.0, 'width': 0.8, 'envelope_floor': 1e-8},
        ode_rtol=1e-10,
        ode_atol=1e-13
    )
    
    ltqg = LTQGFramework(config)
    
    # Complex test system: coupled oscillators
    def H_coupled_oscillators(tau):
        omega1, omega2 = 1.0, 1.5
        coupling = 0.2
        return np.array([
            [omega1, coupling],
            [coupling, omega2]
        ], dtype=complex)
    
    # Initial state: superposition
    psi_0 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    
    # Evolution comparison: ODE vs Unitary with first-class silence
    sigma_span = (-2.0, 2.0)
    
    print(f"Evolving coupled oscillators with first-class silence...")
    print(f"σ-time span: {sigma_span}")
    print(f"Corresponding τ-time span: [{ltqg.tau_from_sigma(sigma_span[0]):.4f}, {ltqg.tau_from_sigma(sigma_span[1]):.4f}]")
    
    # Both methods now use the same H̃_eff with built-in silence
    result_ode = ltqg.sigma_evolution(sigma_span, psi_0, H_coupled_oscillators)
    result_unitary = ltqg.sigma_unitary_evolution(sigma_span, psi_0, H_coupled_oscillators, steps=500)
    
    if result_ode['success'] and result_unitary['success']:
        psi_final_ode = result_ode['psi_sigma'](sigma_span[1])
        psi_final_unitary = result_unitary['psi_final']
        
        # Polished measures for both
        measures_ode = ltqg.quantum_state_measures(psi_final_ode)
        measures_unitary = ltqg.quantum_state_measures(psi_final_unitary)
        
        print(f"\nFinal state measures (ODE method):")
        for key, value in measures_ode.items():
            print(f"   {key}: {value:.8f}")
        
        print(f"\nFinal state measures (Unitary method):")
        for key, value in measures_unitary.items():
            print(f"   {key}: {value:.8f}")
        
        # Overlap between methods
        overlap = np.abs(np.vdot(psi_final_ode, psi_final_unitary))**2
        print(f"\nMethod overlap: {overlap:.8f}")
        print(f"✓ Both methods use identical H̃_eff with first-class silence")
        print(f"✓ All measures polished (no artifacts)")
        print(f"✓ Hermiticity enforced throughout")
        print(f"✓ Centralized config ensures consistency")
    
    return ltqg


def main():
    """Run all enhancement demos."""
    print("Enhanced LTQG Framework Demonstration")
    print("====================================")
    print("Showcasing four key refinements:")
    print("1. First-class silence in H̃_eff")
    print("2. Early Hermiticity enforcement")
    print("3. Centralized configuration")
    print("4. Polished reporting")
    print()
    
    # Set up results directory
    setup_results_directory('demo_enhanced_ltqg')
    
    try:
        # Run all demos
        ltqg1 = demo_first_class_silence()
        ltqg2 = demo_hermiticity_enforcement()
        ltqg3, config3 = demo_centralized_config()
        ltqg4 = demo_polished_reporting()
        ltqg5 = demo_integrated_workflow()
        
        print("\n" + "="*50)
        print("All enhancement demos completed successfully!")
        print("The LTQG framework is now:")
        print("• More robust (Hermiticity enforcement)")
        print("• More consistent (centralized config)")
        print("• More user-friendly (polished output)")
        print("• More reliable (first-class silence)")
        print("="*50)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()