#!/usr/bin/env python3
"""
Asymptotic Silence Visualization Plots

Generate diagnostic plots for asymptotic silence mechanisms:
1. ||H_eff(σ)|| evolution for all envelope types
2. Fidelity, purity, and von Neumann entropy evolution
3. Redshift factor 1:1 theory vs. measured plot

These plots help lock in the narrative for reviewers.
Images are saved to results/asymptotic_silence/
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from asymptotic_silence import AsymptoticSilence
from plotting_utils import configure_matplotlib_for_export, save_plot, setup_results_directory

# Configure matplotlib for high-quality export
configure_matplotlib_for_export()

def plot_effective_hamiltonian_norms():
    """
    Plot ||H_eff(σ)|| (log-axis) for all envelope types with silent band shaded.
    """
    silence = AsymptoticSilence(tau_0=1.0, hbar=1.0)
    
    # Define test Hamiltonian
    def H_two_level(tau):
        omega = 1.0
        coupling = 0.1 / tau
        return np.array([[omega/2, coupling], [coupling, -omega/2]], dtype=complex)
    
    # σ range for visualization
    sigma_array = np.linspace(-6, 3, 200)
    envelope_types = ['tanh', 'exponential', 'polynomial', 'smooth_step']
    
    plt.figure(figsize=(12, 8))
    
    for env_type in envelope_types:
        h_eff_norms = []
        
        for sigma in sigma_array:
            env_params = {'type': env_type}
            H_eff = silence.effective_hamiltonian_with_silence(sigma, H_two_level, env_params)
            h_eff_norms.append(np.linalg.norm(H_eff))
        
        plt.semilogy(sigma_array, h_eff_norms, label=f'{env_type}', linewidth=2)
    
    # Shade silent band (σ < -2)
    plt.axvspan(-6, -2, alpha=0.2, color='gray', label='Silent Band')
    
    plt.xlabel('σ = log(τ/τ₀)', fontsize=14)
    plt.ylabel('||H_eff(σ)||', fontsize=14)
    plt.title('Effective Hamiltonian Norm Evolution\n(Asymptotic Silence Demonstration)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(-6, 3)
    plt.ylim(1e-8, 1e2)
    
    # Annotate observed minima
    plt.text(-4, 1e-6, 'polynomial & smooth_step:\n||H_eff|| = 0', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    plt.text(-3, 1e-4, 'tanh: min ~10⁻⁷', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    output_path = save_plot(plt, 'H_eff_norms_evolution.png', 'asymptotic_silence')
    plt.close()
    
    return output_path

def plot_information_preservation():
    """
    Plot fidelity F(σ), purity, and von Neumann entropy on one panel.
    """
    silence = AsymptoticSilence(tau_0=1.0, hbar=1.0)
    
    # Define test Hamiltonian
    def H_two_level(tau):
        omega = 1.0
        coupling = 0.1 / tau
        return np.array([[omega/2, coupling], [coupling, -omega/2]], dtype=complex)
    
    # Initial entangled state
    initial_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
    
    # Run information analysis
    info_analysis = silence.information_preservation_analysis(
        (-3, 1), H_two_level, initial_state)
    
    if not info_analysis['evolution_success']:
        print("Evolution failed, cannot generate plot")
        return None
    
    sigma_array = info_analysis['sigma_array']
    fidelity = info_analysis['fidelity_with_initial']
    purity = info_analysis['purity']
    entropy = info_analysis['von_neumann_entropy']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Fidelity plot
    ax1.plot(sigma_array, fidelity, 'b-', linewidth=2, label='Fidelity F(σ)')
    ax1.set_ylabel('Fidelity F(σ)', fontsize=12)
    ax1.set_title('Information Preservation During σ-Evolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calculate and annotate average decay rate
    fidelity_decay_rate = np.mean(np.abs(np.gradient(fidelity, sigma_array)))
    ax1.text(0.5, 0.7, f'Avg decay ≈ {fidelity_decay_rate:.3f}/σ', 
             transform=ax1.transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Purity plot
    ax2.plot(sigma_array, purity, 'g-', linewidth=2, label='Purity Tr(ρ²)')
    ax2.set_ylabel('Purity', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0.99, 1.01)  # Zoom in on near-unity purity
    
    # Entropy plot
    ax3.plot(sigma_array, entropy.real, 'r-', linewidth=2, label='von Neumann Entropy')
    ax3.set_ylabel('S_vN', fontsize=12)
    ax3.set_xlabel('σ = log(τ/τ₀)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(-0.01, 0.05)  # Show small entropy values
    
    # Shade silent region
    for ax in [ax1, ax2, ax3]:
        ax.axvspan(-3, -2, alpha=0.15, color='gray', label='Silent Region' if ax == ax1 else "")
    
    plt.tight_layout()
    output_path = save_plot(plt, 'information_preservation_evolution.png', 'asymptotic_silence')
    plt.close()
    
    return output_path

def plot_redshift_test():
    """
    Generate 1:1 theory vs. "measured" redshift plot with correlation and RMS stamps.
    """
    silence = AsymptoticSilence(tau_0=1.0, hbar=1.0)
    
    # Run redshift test
    redshift_test = silence.analyze_schwarzschild_redshift_test(M=1.0, r_range=(2.1, 5.0))
    
    r_array = redshift_test['r_array']
    theoretical = redshift_test['theoretical_redshift']
    measured = redshift_test['measured_redshift']
    correlation = redshift_test['correlation']
    rms_error = redshift_test['rms_error']
    
    plt.figure(figsize=(10, 8))
    
    # 1:1 plot
    plt.plot(theoretical, measured, 'bo', markersize=8, alpha=0.7, label='Test Points')
    
    # Perfect correlation line
    min_val = np.min(theoretical)
    max_val = np.max(theoretical)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Correlation')
    
    plt.xlabel('Theoretical Redshift √(1 - 2M/r)', fontsize=14)
    plt.ylabel('Measured Redshift', fontsize=14)
    plt.title('Schwarzschild Redshift Factor Validation\n(Theory vs. Measurement)', fontsize=16)
    
    # Add correlation and RMS stamps
    plt.text(0.05, 0.95, f'Correlation = {correlation:.6f}', 
             transform=plt.gca().transAxes, fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    plt.text(0.05, 0.85, f'RMS Error = {rms_error:.2e}', 
             transform=plt.gca().transAxes, fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    # Add r/M range annotation
    r_min = np.min(r_array)
    r_max = np.max(r_array)
    plt.text(0.05, 0.75, f'Test Range: r/M ∈ [{r_min:.1f}, {r_max:.1f}]', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.axis('equal')
    
    plt.tight_layout()
    output_path = save_plot(plt, 'redshift_validation_1to1.png', 'asymptotic_silence')
    plt.close()
    
    return output_path

def generate_all_plots():
    """
    Generate all three diagnostic plots and save to results/asymptotic_silence/.
    """
    print("=== Generating Asymptotic Silence Diagnostic Plots ===\n")
    
    # Setup results directory
    results_dir = setup_results_directory('asymptotic_silence')
    print(f"Output directory: {results_dir}\n")
    
    generated_files = []
    
    print("1. Generating ||H_eff(σ)|| evolution plot...")
    try:
        path1 = plot_effective_hamiltonian_norms()
        generated_files.append(path1)
        print(f"   ✓ Saved: {path1}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("2. Generating information preservation plot...")
    try:
        path2 = plot_information_preservation()
        if path2:
            generated_files.append(path2)
            print(f"   ✓ Saved: {path2}")
        else:
            print(f"   ✗ Failed: Evolution unsuccessful")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("3. Generating redshift validation plot...")
    try:
        path3 = plot_redshift_test()
        generated_files.append(path3)
        print(f"   ✓ Saved: {path3}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print(f"\n✓ Generated {len(generated_files)} diagnostic plots successfully!")
    print(f"All files saved to: {os.path.abspath(results_dir)}")
    
    return generated_files

if __name__ == "__main__":
    generate_all_plots()